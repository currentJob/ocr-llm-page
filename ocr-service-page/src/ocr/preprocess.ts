/**
 * 각 모델의 이미지 전처리 (PaddleOCR / PaddleX inference.yml 기준)
 *  - det·rec 은 BGR 로 학습됐다(DecodeImage img_mode: BGR). 방향 분류 두 모델은 RGB.
 */
import type { Pt } from './dbPostprocess'

export type Src = HTMLImageElement | HTMLCanvasElement | OffscreenCanvas | ImageBitmap

export function dims(src: Src): [number, number] {
  return src instanceof HTMLImageElement ? [src.naturalWidth, src.naturalHeight] : [src.width, src.height]
}

function draw(src: Src, w: number, h: number, sx = 0, sy = 0, sw?: number, sh?: number): Uint8ClampedArray {
  const c = new OffscreenCanvas(w, h)
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingQuality = 'high'
  const [iw, ih] = dims(src)
  ctx.drawImage(src, sx, sy, sw ?? iw, sh ?? ih, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h).data
}

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225]

/** RGBA → CHW float, ImageNet 정규화. bgr 이면 채널 순서를 B,G,R 로 넣는다(평균·표준편차는 그 순서 그대로). */
function imagenetCHW(px: Uint8ClampedArray, hw: number, bgr: boolean, out = new Float32Array(3 * hw), off = 0): Float32Array {
  for (let c = 0; c < 3; c++) {
    const srcC = bgr ? 2 - c : c, m = MEAN[c], s = STD[c], base = off + c * hw
    for (let i = 0; i < hw; i++) out[base + i] = (px[i * 4 + srcC] / 255 - m) / s
  }
  return out
}

// ── doc_ori: resize_short 256 → center crop 224 ─────────────────────────────
export function preprocessDocOri(src: Src): Float32Array {
  const [w, h] = dims(src)
  const scale = 256 / Math.min(w, h)
  const cw = 224 / scale  // 원본 좌표에서 잘라 낼 크기
  return imagenetCHW(draw(src, 224, 224, (w - cw) / 2, (h - cw) / 2, cw, cw), 224 * 224, false)
}

// ── det: 긴 변을 maxSide 이하로(작은 이미지는 minSide 까지 키움), 32 배수 ──────
export interface DetInput { tensor: Float32Array; newH: number; newW: number; ratioH: number; ratioW: number }

export function preprocessDet(src: Src, maxSide: number, minSide: number): DetInput {
  const [ow, oh] = dims(src)
  const long = Math.max(ow, oh)
  const ratio = long > maxSide ? maxSide / long : long < minSide ? minSide / long : 1
  const newH = Math.max(32, Math.round((oh * ratio) / 32) * 32)
  const newW = Math.max(32, Math.round((ow * ratio) / 32) * 32)
  const tensor = imagenetCHW(draw(src, newW, newH), newH * newW, true)
  return { tensor, newH, newW, ratioH: newH / oh, ratioW: newW / ow }
}

// ── textline_ori: 160×80 로 늘림, RGB ─────────────────────────────────────────
export function preprocessTextlineOri(crops: OffscreenCanvas[]): Float32Array {
  const hw = 160 * 80, out = new Float32Array(crops.length * 3 * hw)
  crops.forEach((c, i) => imagenetCHW(draw(c, 160, 80), hw, false, out, i * 3 * hw))
  return out
}

// ── rec: 높이 48, 비율 유지, 배치 안 가장 긴 비율에 맞춰 오른쪽 패딩, BGR, [-1,1] ─
export const REC_H = 48, REC_MAX_W = 3200

export function recWidth(crops: OffscreenCanvas[]): number {
  const maxRatio = Math.max(320 / REC_H, ...crops.map(c => c.width / c.height))
  // 32 배수로 올려 WebGPU 가 같은 모양의 셰이더를 다시 쓰게 한다
  return Math.min(REC_MAX_W, Math.ceil((REC_H * maxRatio) / 32) * 32)
}

export function preprocessRec(crops: OffscreenCanvas[], W: number): Float32Array {
  const plane = REC_H * W, out = new Float32Array(crops.length * 3 * plane)  // 0 = 회색 패딩
  crops.forEach((c, i) => {
    const w = Math.max(1, Math.min(W, Math.ceil((REC_H * c.width) / c.height)))
    const px = draw(c, w, REC_H)
    for (let ch = 0; ch < 3; ch++) {
      const base = i * 3 * plane + ch * plane, srcC = 2 - ch
      for (let y = 0; y < REC_H; y++) for (let x = 0; x < w; x++) out[base + y * W + x] = px[(y * w + x) * 4 + srcC] / 127.5 - 1
    }
  })
  return out
}

// ── 원근 보정 크롭 (get_rotate_crop_image) ─────────────────────────────────────
export interface Pixels { data: Uint8ClampedArray; w: number; h: number }

export function readPixels(src: Src): Pixels {
  const [w, h] = dims(src)
  return { data: draw(src, w, h), w, h }
}

function homography(src: Pt[], dst: Pt[]): number[] {
  // dst → src 로 가는 3x3 (역매핑용). 8x8 가우스 소거.
  const A: number[][] = []
  for (let i = 0; i < 4; i++) {
    const [x, y] = dst[i], [u, v] = src[i]
    A.push([x, y, 1, 0, 0, 0, -x * u, -y * u, u])
    A.push([0, 0, 0, x, y, 1, -x * v, -y * v, v])
  }
  for (let c = 0; c < 8; c++) {
    let p = c
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r
    ;[A[c], A[p]] = [A[p], A[c]]
    for (let r = 0; r < 8; r++) {
      if (r === c || !A[c][c]) continue
      const f = A[r][c] / A[c][c]
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k]
    }
  }
  return [...A.map((row, i) => row[8] / (row[i] || 1)), 1]
}

/** box: [tl, tr, br, bl]. 세로로 긴 박스(h/w ≥ 1.5)는 반시계 90° 돌려 가로 줄로 만든다. */
export function cropByBox(px: Pixels, box: Pt[]): OffscreenCanvas | null {
  const [tl, tr, br, bl] = box
  const w = Math.round(Math.max(Math.hypot(tr[0] - tl[0], tr[1] - tl[1]), Math.hypot(br[0] - bl[0], br[1] - bl[1])))
  const h = Math.round(Math.max(Math.hypot(bl[0] - tl[0], bl[1] - tl[1]), Math.hypot(br[0] - tr[0], br[1] - tr[1])))
  if (w < 2 || h < 2) return null

  const H = homography(box, [[0, 0], [w, 0], [w, h], [0, h]])
  const { data: s, w: iw, h: ih } = px
  const img = new ImageData(w, h), d = img.data
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const q = H[6] * x + H[7] * y + H[8]
      // 가장자리는 복제(BORDER_REPLICATE)
      const sx = Math.min(iw - 1, Math.max(0, (H[0] * x + H[1] * y + H[2]) / q))
      const sy = Math.min(ih - 1, Math.max(0, (H[3] * x + H[4] * y + H[5]) / q))
      const x0 = Math.floor(sx), y0 = Math.floor(sy)
      const x1 = Math.min(iw - 1, x0 + 1), y1 = Math.min(ih - 1, y0 + 1)
      const fx = sx - x0, fy = sy - y0
      const a = (y0 * iw + x0) * 4, b = (y0 * iw + x1) * 4, c = (y1 * iw + x0) * 4, e = (y1 * iw + x1) * 4
      const o = (y * w + x) * 4
      for (let k = 0; k < 3; k++) {
        d[o + k] = (s[a + k] * (1 - fx) + s[b + k] * fx) * (1 - fy) + (s[c + k] * (1 - fx) + s[e + k] * fx) * fy
      }
      d[o + 3] = 255
    }
  }
  const out = new OffscreenCanvas(w, h)
  out.getContext('2d')!.putImageData(img, 0, 0)
  return h / w >= 1.5 ? rotate(out, 270) : out
}

/** 시계 방향 angle 도 회전 */
export function rotate(src: Src, angle: 90 | 180 | 270): OffscreenCanvas {
  const [w, h] = dims(src)
  const [nw, nh] = angle === 180 ? [w, h] : [h, w]
  const c = new OffscreenCanvas(nw, nh)
  const ctx = c.getContext('2d')!
  ctx.translate(nw / 2, nh / 2)
  ctx.rotate((angle * Math.PI) / 180)
  ctx.drawImage(src, -w / 2, -h / 2)
  return c
}
