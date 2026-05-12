/**
 * DB (Differentiable Binarization) 후처리
 * Python 구현과 알고리즘을 정확히 일치시킴:
 *   findContours → minAreaRect → boxScore(fillPoly) → unclip → 좌표 역변환
 */

export interface DetBox {
  box: [number, number][]  // 4 corner points [[x1,y1],...] 원본 좌표
  score: number
}

// ── 연결 영역 + 컨투어 픽셀 수집 ────────────────────────────────────────────

function collectComponents(
  binary: Uint8Array,
  w: number,
  h: number,
): [number, number][][] {
  const visited  = new Uint8Array(w * h)
  const result: [number, number][][] = []

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      if (!binary[idx] || visited[idx]) continue

      const pixels: [number, number][] = []
      const queue = [idx]
      visited[idx] = 1
      let qi = 0

      while (qi < queue.length) {
        const cur = queue[qi++]
        const cx  = cur % w
        const cy  = (cur / w) | 0
        pixels.push([cx, cy])

        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]] as const) {
          const nx = cx + dx, ny = cy + dy
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
          const ni = ny * w + nx
          if (binary[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni) }
        }
      }

      if (pixels.length >= 4) result.push(pixels)
    }
  }
  return result
}

// ── 볼록 껍질 (Andrew's monotone chain) ────────────────────────────────────

function convexHull(pts: [number, number][]): [number, number][] {
  if (pts.length < 3) return pts.slice()
  const sorted = [...pts].sort((a, b) => a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1])

  const cross = (O: [number,number], A: [number,number], B: [number,number]) =>
    (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0])

  const lower: [number, number][] = []
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop()
    lower.push(p)
  }
  const upper: [number, number][] = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop()
    upper.push(p)
  }
  lower.pop(); upper.pop()
  return [...lower, ...upper]
}

// ── 최소 면적 회전 박스 (Rotating calipers) ─────────────────────────────────
// Python cv2.minAreaRect + cv2.boxPoints 에 해당

function minAreaRect(pixels: [number, number][]): [number, number][] {
  const hull = convexHull(pixels)
  if (hull.length < 2) {
    const xs = pixels.map(p => p[0]), ys = pixels.map(p => p[1])
    const x1 = Math.min(...xs), x2 = Math.max(...xs)
    const y1 = Math.min(...ys), y2 = Math.max(...ys)
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
  }

  let minArea = Infinity
  let best: [number,number][] = []

  for (let i = 0; i < hull.length; i++) {
    const [ax, ay] = hull[i]
    const [bx, by] = hull[(i + 1) % hull.length]
    const len = Math.hypot(bx - ax, by - ay)
    if (len < 1e-10) continue

    const ux = (bx - ax) / len   // 에지 방향 단위벡터
    const uy = (by - ay) / len
    const nx = -uy, ny = ux       // 법선 단위벡터

    let minP = Infinity, maxP = -Infinity
    let minN = Infinity, maxN = -Infinity

    for (const [px, py] of hull) {
      const proj = px*ux + py*uy
      const norm = px*nx + py*ny
      if (proj < minP) minP = proj
      if (proj > maxP) maxP = proj
      if (norm < minN) minN = norm
      if (norm > maxN) maxN = norm
    }

    const area = (maxP - minP) * (maxN - minN)
    if (area < minArea) {
      minArea = area
      // 4 모서리 좌표 복원
      best = [
        [minP*ux + minN*nx, minP*uy + minN*ny],
        [maxP*ux + minN*nx, maxP*uy + minN*ny],
        [maxP*ux + maxN*nx, maxP*uy + maxN*ny],
        [minP*ux + maxN*nx, minP*uy + maxN*ny],
      ] as [number, number][]
    }
  }
  return best
}

// ── 박스 스코어 (polygon 스캔라인 채움) ────────────────────────────────────
// Python cv2.fillPoly + (prob * mask).sum() / mask.sum()

function boxScore(
  prob: Float32Array,
  probW: number,
  probH: number,
  pts: [number, number][],
): number {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1])
  const x1 = Math.max(0, Math.floor(Math.min(...xs)))
  const x2 = Math.min(probW - 1, Math.ceil(Math.max(...xs)))
  const y1 = Math.max(0, Math.floor(Math.min(...ys)))
  const y2 = Math.min(probH - 1, Math.ceil(Math.max(...ys)))

  let sum = 0, count = 0
  const n = pts.length

  for (let y = y1; y <= y2; y++) {
    // 스캔라인 y에서 폴리곤 에지와의 교점 계산
    const xs_int: number[] = []
    for (let i = 0; i < n; i++) {
      const [p1x, p1y] = pts[i]
      const [p2x, p2y] = pts[(i + 1) % n]
      if ((p1y <= y && p2y > y) || (p2y <= y && p1y > y)) {
        xs_int.push(p1x + (y - p1y) / (p2y - p1y) * (p2x - p1x))
      }
    }
    xs_int.sort((a, b) => a - b)

    for (let j = 0; j < xs_int.length - 1; j += 2) {
      const xa = Math.max(x1, Math.ceil(xs_int[j]))
      const xb = Math.min(x2, Math.floor(xs_int[j + 1]))
      for (let x = xa; x <= xb; x++) {
        sum += prob[y * probW + x]
        count++
      }
    }
  }
  return count > 0 ? sum / count : 0
}

// ── 언클립 (중심 기준 확장) ─────────────────────────────────────────────────
// Python _unclip 과 동일

function unclip4(pts: [number, number][], ratio: number): [number, number][] {
  const cx = (pts[0][0]+pts[1][0]+pts[2][0]+pts[3][0]) / 4
  const cy = (pts[0][1]+pts[1][1]+pts[2][1]+pts[3][1]) / 4
  return pts.map(([x, y]) => [
    Math.round((x - cx) * ratio + cx),
    Math.round((y - cy) * ratio + cy),
  ] as [number, number])
}

// ── 메인 후처리 함수 ─────────────────────────────────────────────────────────

export function dbPostprocess(
  predData: Float32Array,
  probH: number,
  probW: number,
  ratioH: number,   // new / orig
  ratioW: number,
  origH: number,
  origW: number,
  thresh       = 0.3,
  boxThresh    = 0.5,
  unclipRatio  = 1.5,
): DetBox[] {
  // 이진화
  const binary = new Uint8Array(probH * probW)
  for (let i = 0; i < binary.length; i++) binary[i] = predData[i] > thresh ? 1 : 0

  // 연결 영역 탐색
  const components = collectComponents(binary, probW, probH)

  const boxes: DetBox[] = []

  for (const pixels of components) {
    // 최소 면적 회전 박스
    const corners = minAreaRect(pixels)
    if (!corners.length) continue

    // 짧은 변이 너무 짧으면 제거
    const side0 = Math.hypot(corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])
    const side1 = Math.hypot(corners[2][0]-corners[1][0], corners[2][1]-corners[1][1])
    if (Math.min(side0, side1) < 3) continue

    // 박스 스코어
    const score = boxScore(predData, probW, probH, corners)
    if (score < boxThresh) continue

    // 언클립
    const expanded = unclip4(corners, unclipRatio)

    // 원본 좌표로 변환  (ratio = new/orig → orig = new/ratio)
    const box = expanded.map(([x, y]) => [
      Math.min(origW - 1, Math.max(0, Math.round(x / ratioW))),
      Math.min(origH - 1, Math.max(0, Math.round(y / ratioH))),
    ] as [number, number])

    // 최종 크기 검사
    const xs = box.map(p => p[0]), ys = box.map(p => p[1])
    if (Math.max(...xs) - Math.min(...xs) < 4) continue
    if (Math.max(...ys) - Math.min(...ys) < 2) continue

    boxes.push({ box, score })
  }

  return boxes
}
