/**
 * 각 모델의 이미지 전처리
 */

const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD  = [0.229, 0.224, 0.225]

function imgToCanvas(
  src: HTMLImageElement | OffscreenCanvas,
  w: number,
  h: number,
): OffscreenCanvas {
  const c = new OffscreenCanvas(w, h)
  c.getContext('2d')!.drawImage(src as CanvasImageSource, 0, 0, w, h)
  return c
}

function imagenetNormCHW(data: Uint8ClampedArray, hw: number): Float32Array {
  const out = new Float32Array(3 * hw)
  for (let i = 0; i < hw; i++) {
    out[0 * hw + i] = (data[i*4]   / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    out[1 * hw + i] = (data[i*4+1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    out[2 * hw + i] = (data[i*4+2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
  }
  return out
}

// ── doc_ori ─────────────────────────────────────────────────────────────────
// ResizeImage(resize_short=256) → CenterCrop(224) → ImageNet norm

export function preprocessDocOri(img: HTMLImageElement): Float32Array {
  const { naturalWidth: ow, naturalHeight: oh } = img
  const scale = 256 / Math.min(ow, oh)
  const nw = Math.round(ow * scale)
  const nh = Math.round(oh * scale)

  const tmp = new OffscreenCanvas(nw, nh)
  tmp.getContext('2d')!.drawImage(img, 0, 0, nw, nh)

  const x0 = (nw - 224) >> 1
  const y0 = (nh - 224) >> 1
  const cropped = new OffscreenCanvas(224, 224)
  cropped.getContext('2d')!.drawImage(tmp, x0, y0, 224, 224, 0, 0, 224, 224)

  const { data } = cropped.getContext('2d')!.getImageData(0, 0, 224, 224)
  return imagenetNormCHW(data, 224 * 224)
}

// ── det ─────────────────────────────────────────────────────────────────────
// DetResizeForTest(resize_long=960) → 32배수 → ImageNet norm

export interface DetPreprocessResult {
  tensor: Float32Array
  newH: number
  newW: number
  ratioH: number   // newH / origH
  ratioW: number   // newW / origW
}

export function preprocessDet(img: HTMLImageElement, resizeLong = 960): DetPreprocessResult {
  const ow = img.naturalWidth, oh = img.naturalHeight
  const ratio = Math.min(resizeLong / Math.max(ow, oh), 1)
  const newH  = Math.max(Math.round((oh * ratio) / 32) * 32, 32)
  const newW  = Math.max(Math.round((ow * ratio) / 32) * 32, 32)

  const c    = imgToCanvas(img, newW, newH)
  const { data } = c.getContext('2d')!.getImageData(0, 0, newW, newH)
  const tensor = imagenetNormCHW(data, newH * newW)

  return { tensor, newH, newW, ratioH: newH / oh, ratioW: newW / ow }
}

// ── textline_ori ─────────────────────────────────────────────────────────────
// ResizeImage(size=[160, 80]) → ImageNet norm

export function preprocessTextlineOri(src: OffscreenCanvas): Float32Array {
  const c = imgToCanvas(src, 160, 80)
  const { data } = c.getContext('2d')!.getImageData(0, 0, 160, 80)
  return imagenetNormCHW(data, 160 * 80)
}

// ── rec ──────────────────────────────────────────────────────────────────────
// RecResizeImg(image_shape=[3,48,320]): H=48, 비율유지, max_w=320, 오른쪽 패딩, [-1,1] 정규화

export function preprocessRec(src: OffscreenCanvas, maxW = 320): Float32Array {
  const sw = src.width, sh = src.height
  const newW = Math.max(1, Math.min(Math.ceil(48 * sw / sh), maxW))

  const tmp = new OffscreenCanvas(newW, 48)
  tmp.getContext('2d')!.drawImage(src, 0, 0, newW, 48)
  const { data } = tmp.getContext('2d')!.getImageData(0, 0, newW, 48)

  const out = new Float32Array(3 * 48 * maxW)  // 패딩 포함
  for (let ch = 0; ch < 3; ch++) {
    for (let y = 0; y < 48; y++) {
      for (let x = 0; x < newW; x++) {
        const v = data[(y * newW + x) * 4 + ch] / 255
        out[ch * 48 * maxW + y * maxW + x] = (v - 0.5) / 0.5
      }
    }
  }
  return out
}

// ── warpPerspective 기반 크롭 ────────────────────────────────────────────────
// Python cv2.warpPerspective 와 동일한 호모그래피 역매핑 + 이중선형 보간

/** 8x8 선형계 가우스 소거법 */
function solveLinear8(A: number[][], b: number[]): number[] {
  const n = 8
  const aug = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    let maxRow = col
    for (let r = col + 1; r < n; r++)
      if (Math.abs(aug[r][col]) > Math.abs(aug[maxRow][col])) maxRow = r
    ;[aug[col], aug[maxRow]] = [aug[maxRow], aug[col]]

    const piv = aug[col][col]
    if (Math.abs(piv) < 1e-12) continue

    for (let r = col + 1; r < n; r++) {
      const f = aug[r][col] / piv
      for (let j = col; j <= n; j++) aug[r][j] -= f * aug[col][j]
    }
  }

  const x = new Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    x[r] = aug[r][n]
    for (let c = r + 1; c < n; c++) x[r] -= aug[r][c] * x[c]
    x[r] /= aug[r][r]
  }
  return x
}

/** src 4점 → dst 4점 호모그래피 계산 (DLT) */
function computeHomography(
  src: [number,number][],
  dst: [number,number][],
): Float64Array {
  const A: number[][] = [], b: number[] = []
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i]
    A.push([x, y, 1, 0, 0, 0, -x*u, -y*u])
    A.push([0, 0, 0, x, y, 1, -x*v, -y*v])
    b.push(u, v)
  }
  const h = solveLinear8(A, b)
  return new Float64Array([h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7],1])
}

/** 3x3 행렬 역행렬 */
function invertH(H: Float64Array): Float64Array {
  const [a,b,c,d,e,f,g,hi,i] = H
  const det = a*(e*i-f*hi) - b*(d*i-f*g) + c*(d*hi-e*g)
  const inv = new Float64Array(9)
  inv[0] =  (e*i-f*hi)/det; inv[1] = -(b*i-c*hi)/det; inv[2] =  (b*f-c*e)/det
  inv[3] = -(d*i-f*g)/det;  inv[4] =  (a*i-c*g)/det;  inv[5] = -(a*f-c*d)/det
  inv[6] =  (d*hi-e*g)/det; inv[7] = -(a*hi-b*g)/det; inv[8] =  (a*e-b*d)/det
  return inv
}

/** 4점 정렬: [tl, tr, br, bl] — Python _order_points 와 동일 */
function orderPoints(pts: [number,number][]): [[number,number],[number,number],[number,number],[number,number]] {
  const sum  = pts.map(([x,y]) => x + y)
  const diff = pts.map(([x,y]) => y - x)
  return [
    pts[sum.indexOf(Math.min(...sum))],
    pts[diff.indexOf(Math.min(...diff))],
    pts[sum.indexOf(Math.max(...sum))],
    pts[diff.indexOf(Math.max(...diff))],
  ]
}

/**
 * cv2.warpPerspective 와 동일한 원근 변환 크롭
 * src: HTMLImageElement, box: 4 corner points (원본 좌표)
 */
export function cropByBox(
  img: HTMLImageElement,
  box: [number,number][],
): OffscreenCanvas | null {
  const [tl, tr, br, bl] = orderPoints(box)

  const w = Math.max(
    1,
    Math.round(Math.max(
      Math.hypot(br[0]-bl[0], br[1]-bl[1]),
      Math.hypot(tr[0]-tl[0], tr[1]-tl[1]),
    )),
  )
  const h = Math.max(
    1,
    Math.round(Math.max(
      Math.hypot(tr[0]-br[0], tr[1]-br[1]),
      Math.hypot(tl[0]-bl[0], tl[1]-bl[1]),
    )),
  )
  if (w <= 0 || h <= 0) return null

  // 소스 이미지 → ImageData
  const iw = img.naturalWidth, ih = img.naturalHeight
  const srcCanvas = new OffscreenCanvas(iw, ih)
  srcCanvas.getContext('2d')!.drawImage(img, 0, 0)
  const srcPx = srcCanvas.getContext('2d')!.getImageData(0, 0, iw, ih).data

  // 호모그래피: src(박스 좌표) → dst(0,0 ~ w,h)
  const srcPts: [number,number][] = [tl, tr, br, bl]
  const dstPts: [number,number][] = [[0,0],[w,0],[w,h],[0,h]]
  const H     = computeHomography(srcPts, dstPts)
  const H_inv = invertH(H)

  // 역매핑 + 이중선형 보간
  const dst    = new OffscreenCanvas(w, h)
  const dstCtx = dst.getContext('2d')!
  const dstImg = dstCtx.createImageData(w, h)
  const dstPx  = dstImg.data

  for (let dy = 0; dy < h; dy++) {
    for (let dx = 0; dx < w; dx++) {
      const ww = H_inv[6]*dx + H_inv[7]*dy + H_inv[8]
      const sx = (H_inv[0]*dx + H_inv[1]*dy + H_inv[2]) / ww
      const sy = (H_inv[3]*dx + H_inv[4]*dy + H_inv[5]) / ww

      const x0 = Math.floor(sx), x1 = x0 + 1
      const y0 = Math.floor(sy), y1 = y0 + 1
      if (x0 < 0 || x1 >= iw || y0 < 0 || y1 >= ih) continue

      const fx = sx - x0, fy = sy - y0
      const di  = (dy * w + dx) * 4

      for (let c = 0; c < 3; c++) {
        const v00 = srcPx[(y0*iw + x0)*4 + c]
        const v10 = srcPx[(y0*iw + x1)*4 + c]
        const v01 = srcPx[(y1*iw + x0)*4 + c]
        const v11 = srcPx[(y1*iw + x1)*4 + c]
        dstPx[di+c] = Math.round(v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy)
      }
      dstPx[di+3] = 255
    }
  }

  dstCtx.putImageData(dstImg, 0, 0)
  return dst
}
