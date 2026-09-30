/**
 * DB (Differentiable Binarization) 후처리 — PaddleOCR DBPostProcess 와 같은 순서:
 *   이진화 → 연결 영역 → minAreaRect → 박스 점수 → unclip(변마다 같은 거리만큼 확장) → 원본 좌표
 */

export type Pt = [number, number]

export interface DetBox {
  box:   Pt[]   // [tl, tr, br, bl] 원본 좌표
  score: number
}

// 8-연결 영역. 볼록 껍질에는 행마다 양 끝 픽셀만 있으면 되므로 그것만 모은다.
function collectComponents(binary: Uint8Array, w: number, h: number, maxCount: number): Pt[][] {
  const visited = new Uint8Array(w * h)
  const queue   = new Int32Array(w * h)
  const result: Pt[][] = []

  for (let start = 0; start < w * h && result.length < maxCount; start++) {
    if (!binary[start] || visited[start]) continue
    visited[start] = 1
    let head = 0, tail = 0
    queue[tail++] = start
    const rowMin = new Map<number, number>(), rowMax = new Map<number, number>()

    while (head < tail) {
      const cur = queue[head++]
      const cx = cur % w, cy = (cur / w) | 0
      if (!(rowMin.get(cy)! <= cx)) rowMin.set(cy, cx)
      if (!(rowMax.get(cy)! >= cx)) rowMax.set(cy, cx)
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy
        if (ny < 0 || ny >= h) continue
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx
          if (nx < 0 || nx >= w || (dx === 0 && dy === 0)) continue
          const ni = ny * w + nx
          if (binary[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni }
        }
      }
    }
    if (tail < 4) continue
    const pts: Pt[] = []
    for (const [y, x0] of rowMin) { pts.push([x0, y]); const x1 = rowMax.get(y)!; if (x1 !== x0) pts.push([x1, y]) }
    result.push(pts)
  }
  return result
}

function convexHull(pts: Pt[]): Pt[] {
  if (pts.length < 3) return pts.slice()
  const s = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: Pt[] = [], upper: Pt[] = []
  for (const p of s) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop()
    lower.push(p)
  }
  for (let i = s.length - 1; i >= 0; i--) {
    const p = s[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop()
    upper.push(p)
  }
  lower.pop(); upper.pop()
  return [...lower, ...upper]
}

interface Rect { cx: number; cy: number; ux: number; uy: number; w: number; h: number }  // u: 너비 방향 단위벡터

// cv2.minAreaRect (rotating calipers)
function minAreaRect(pts: Pt[]): Rect {
  const hull = convexHull(pts)
  let best: Rect = { cx: hull[0][0], cy: hull[0][1], ux: 1, uy: 0, w: 0, h: 0 }
  let bestArea = Infinity
  for (let i = 0; i < hull.length; i++) {
    const [ax, ay] = hull[i], [bx, by] = hull[(i + 1) % hull.length]
    const len = Math.hypot(bx - ax, by - ay)
    if (len < 1e-9) continue
    const ux = (bx - ax) / len, uy = (by - ay) / len
    let p0 = Infinity, p1 = -Infinity, n0 = Infinity, n1 = -Infinity
    for (const [px, py] of hull) {
      const p = px * ux + py * uy, n = -px * uy + py * ux
      if (p < p0) p0 = p; if (p > p1) p1 = p
      if (n < n0) n0 = n; if (n > n1) n1 = n
    }
    const area = (p1 - p0) * (n1 - n0)
    if (area < bestArea) {
      bestArea = area
      const pc = (p0 + p1) / 2, nc = (n0 + n1) / 2
      best = { cx: pc * ux - nc * uy, cy: pc * uy + nc * ux, ux, uy, w: p1 - p0, h: n1 - n0 }
    }
  }
  // 긴 변을 너비로 (방향만 바꾼다)
  if (best.h > best.w) best = { ...best, ux: -best.uy, uy: best.ux, w: best.h, h: best.w }
  return best
}

// 네 꼭짓점을 [tl, tr, br, bl] 로 — PaddleOCR get_mini_boxes 와 같은 규칙
function corners(r: Rect): Pt[] {
  const hw = r.w / 2, hh = r.h / 2, vx = -r.uy, vy = r.ux
  const pts: Pt[] = [
    [r.cx - r.ux * hw - vx * hh, r.cy - r.uy * hw - vy * hh],
    [r.cx + r.ux * hw - vx * hh, r.cy + r.uy * hw - vy * hh],
    [r.cx + r.ux * hw + vx * hh, r.cy + r.uy * hw + vy * hh],
    [r.cx - r.ux * hw + vx * hh, r.cy - r.uy * hw + vy * hh],
  ]
  pts.sort((a, b) => a[0] - b[0])
  const [l0, l1] = pts[1][1] > pts[0][1] ? [pts[0], pts[1]] : [pts[1], pts[0]]
  const [r0, r1] = pts[3][1] > pts[2][1] ? [pts[2], pts[3]] : [pts[3], pts[2]]
  return [l0, r0, r1, l1]
}

// box_score_fast: 박스 다각형 안의 확률 평균
function boxScore(prob: Float32Array, pw: number, ph: number, pts: Pt[]): number {
  const ys = pts.map(p => p[1]), xs = pts.map(p => p[0])
  const x1 = Math.max(0, Math.floor(Math.min(...xs))), x2 = Math.min(pw - 1, Math.ceil(Math.max(...xs)))
  const y1 = Math.max(0, Math.floor(Math.min(...ys))), y2 = Math.min(ph - 1, Math.ceil(Math.max(...ys)))
  let sum = 0, count = 0
  for (let y = y1; y <= y2; y++) {
    const cut: number[] = []
    for (let i = 0; i < 4; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[(i + 1) % 4]
      if ((ay <= y && by > y) || (by <= y && ay > y)) cut.push(ax + (y - ay) / (by - ay) * (bx - ax))
    }
    cut.sort((a, b) => a - b)
    for (let j = 0; j + 1 < cut.length; j += 2) {
      for (let x = Math.max(x1, Math.ceil(cut[j])); x <= Math.min(x2, Math.floor(cut[j + 1])); x++) { sum += prob[y * pw + x]; count++ }
    }
  }
  if (count) return sum / count
  // 한 줄짜리 얇은 박스: 중심 픽셀
  const cx = Math.round((x1 + x2) / 2), cy = Math.round((y1 + y2) / 2)
  return prob[cy * pw + cx] ?? 0
}

export function dbPostprocess(
  pred: Float32Array, probH: number, probW: number,
  ratioH: number, ratioW: number,          // 검출 입력 / 원본
  origH: number, origW: number,
  thresh = 0.3, boxThresh = 0.6, unclipRatio = 1.5, maxCandidates = 1000,
): DetBox[] {
  const binary = new Uint8Array(probH * probW)
  for (let i = 0; i < binary.length; i++) binary[i] = pred[i] > thresh ? 1 : 0

  const boxes: DetBox[] = []
  for (const pts of collectComponents(binary, probW, probH, maxCandidates)) {
    const r = minAreaRect(pts)
    if (Math.min(r.w, r.h) < 3) continue
    const score = boxScore(pred, probW, probH, corners(r))
    if (score < boxThresh) continue

    // pyclipper 라운드 오프셋 후 minAreaRect 와 같다: 네 변을 d 만큼 밀어낸다
    const d = (r.w * r.h * unclipRatio) / (2 * (r.w + r.h))
    const e: Rect = { ...r, w: r.w + 2 * d, h: r.h + 2 * d }
    if (Math.min(e.w, e.h) < 5) continue

    const box = corners(e).map(([x, y]) => [
      Math.min(origW, Math.max(0, Math.round(x / ratioW))),
      Math.min(origH, Math.max(0, Math.round(y / ratioH))),
    ] as Pt)
    boxes.push({ box, score })
  }
  return boxes
}
