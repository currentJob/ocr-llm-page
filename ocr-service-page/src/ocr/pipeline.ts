import { loadOrt, type Ort, type Backend } from './ort'
import {
  dims, preprocessDocOri, preprocessDet, preprocessTextlineOri, preprocessRec, recWidth,
  readPixels, cropByBox, rotate, type Src,
} from './preprocess'
import { dbPostprocess, type Pt } from './dbPostprocess'

// ── 타입 ─────────────────────────────────────────────────────────────────────

export interface OcrItem {
  text:     string
  recScore: number
  detScore: number
  box:      number[][]
}

export interface LoadProgress {
  step:  string
  done:  number
  total: number
}

export interface OcrOptions {
  backend?:     Backend   // 기본: WebGPU 가 되면 webgpu
  detMaxSide?:  number    // 검출 입력 긴 변 상한
  detMinSide?:  number    // 작은 이미지는 이 크기까지 키워 검출
  boxThresh?:   number
  docOrientation?: boolean
  textlineOrientation?: boolean
}

interface CharDict { charTable: (string | null)[]; blankIdx: number }
type Session = Awaited<ReturnType<Ort['InferenceSession']['create']>>

const MODELS = ['PP-LCNet_x1_0_doc_ori', 'PP-OCRv5_mobile_det', 'PP-LCNet_x1_0_textline_ori', 'korean_PP-OCRv5_mobile_rec'] as const

// ── CTC 디코딩 ────────────────────────────────────────────────────────────────

function ctcDecode(data: Float32Array, off: number, T: number, C: number, dict: CharDict): { text: string; score: number } {
  let text = '', sum = 0, n = 0, prev = -1
  for (let t = 0; t < T; t++) {
    let best = 0, bestV = -Infinity
    const o = off + t * C
    for (let c = 0; c < C; c++) if (data[o + c] > bestV) { bestV = data[o + c]; best = c }
    if (best !== prev && best !== dict.blankIdx) {
      const ch = dict.charTable[best]
      if (ch) { text += ch; sum += bestV; n++ }
    }
    prev = best
  }
  // 사전에 자모가 섞여 있어 조합형으로 나올 수 있다
  return { text: text.normalize('NFC').trim(), score: n ? sum / n : 0 }
}

function softmax(v: ArrayLike<number>, off: number, n: number): number[] {
  const xs = Array.from({ length: n }, (_, i) => v[off + i])
  if (xs.every(x => x >= 0 && x <= 1) && Math.abs(xs.reduce((a, b) => a + b, 0) - 1) < 1e-3) return xs  // 이미 확률
  const m = Math.max(...xs), e = xs.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0)
  return e.map(x => x / s)
}

/** 읽는 순서: 위→아래, 같은 줄(세로로 절반 이상 겹침)이면 왼→오 */
function readingOrder<T extends { box: Pt[] }>(items: T[]): T[] {
  const top = (b: Pt[]) => Math.min(...b.map(p => p[1]))
  const bottom = (b: Pt[]) => Math.max(...b.map(p => p[1]))
  const left = (b: Pt[]) => Math.min(...b.map(p => p[0]))
  const s = [...items].sort((a, b) => top(a.box) - top(b.box) || left(a.box) - left(b.box))
  for (let i = 0; i < s.length - 1; i++) {
    for (let j = i; j >= 0; j--) {
      const a = s[j].box, b = s[j + 1].box
      const overlap = Math.min(bottom(a), bottom(b)) - Math.max(top(a), top(b))
      const minH = Math.min(bottom(a) - top(a), bottom(b) - top(b))
      if (overlap > minH * 0.5 && left(b) < left(a)) [s[j], s[j + 1]] = [s[j + 1], s[j]]
      else break
    }
  }
  return s
}

// ── 메인 파이프라인 ───────────────────────────────────────────────────────────

export class KoreanOCR {
  private ort: Ort
  readonly backend: Backend
  private docOri: Session
  private det: Session
  private textlineOri: Session
  private rec: Session
  private dict: CharDict
  private opts: Required<Omit<OcrOptions, 'backend'>>

  private constructor(ort: Ort, backend: Backend, s: Session[], dict: CharDict, opts: Required<Omit<OcrOptions, 'backend'>>) {
    this.ort = ort; this.backend = backend
    ;[this.docOri, this.det, this.textlineOri, this.rec] = s
    this.dict = dict; this.opts = opts
  }

  static async create(onProgress?: (p: LoadProgress) => void, options: OcrOptions = {}): Promise<KoreanOCR> {
    const total = MODELS.length + 2
    let done = 0
    const tick = (step: string) => onProgress?.({ step, done: ++done, total })
    onProgress?.({ step: '런타임 준비 중...', done: 0, total })

    const { ort, backend } = await loadOrt(options.backend)
    const base = import.meta.env.BASE_URL
    const eps = backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
    const get = (path: string) => fetch(base + path).then(r => { if (!r.ok) throw new Error(`${path} 로드 실패 (${r.status})`); return r })
    // 내려받기는 동시에, 세션 생성은 차례로(WebGPU EP 는 동시 생성을 막는다)
    const [dict, ...bytes] = await Promise.all([
      get('models/charDict.json').then(r => r.json() as Promise<CharDict>),
      ...MODELS.map(m => get(`models/${m}.onnx`).then(r => r.arrayBuffer()).then(b => { tick(m); return new Uint8Array(b) })),
    ])
    const sessions: Session[] = []
    for (const b of bytes) sessions.push(await ort.InferenceSession.create(b, { executionProviders: eps, graphOptimizationLevel: 'all' }))
    const gpu = backend === 'webgpu'
    const ocr = new KoreanOCR(ort, backend, sessions, dict as CharDict, {
      detMaxSide: options.detMaxSide ?? (gpu ? 1920 : 1600),
      detMinSide: options.detMinSide ?? 960,
      boxThresh: options.boxThresh ?? 0.6,
      docOrientation: options.docOrientation ?? true,
      textlineOrientation: options.textlineOrientation ?? true,
    })
    if (gpu) {
      // WebGPU 는 첫 추론 때 셰이더를 컴파일한다(약 10초). 첫 사진이 느리지 않게 여기서 한 번 돌린다.
      tick('GPU 준비 중...')
      const c = new OffscreenCanvas(320, 96), g = c.getContext('2d')!
      g.fillStyle = '#fff'; g.fillRect(0, 0, 320, 96); g.fillStyle = '#000'; g.font = '32px sans-serif'; g.fillText('가나다 ABC 123', 16, 60)
      await ocr.predict(c)
    }
    onProgress?.({ step: '완료', done: total, total })
    return ocr
  }

  /** 이미지에서 OCR 수행. 박스 좌표는 입력 이미지 기준. */
  async predict(img: Src): Promise<OcrItem[]> {
    const { angle, image } = this.opts.docOrientation ? await this.docOrientation(img) : { angle: 0 as const, image: img }
    const [W, H] = dims(image)

    let found = await this.detect(image)
    // 검출이 없으면 이미 잘라 낸 한 줄 이미지로 보고 통째로 인식
    if (!found.length) found = [{ box: [[0, 0], [W, 0], [W, H], [0, H]], score: 1 }]
    found = readingOrder(found)

    const px = readPixels(image)
    const lines = found.map(f => ({ ...f, crop: cropByBox(px, f.box) })).filter(l => l.crop) as
      { box: Pt[]; score: number; crop: OffscreenCanvas }[]
    const crops = lines.map(l => l.crop)
    // 줄 방향 분류기가 뒤집혔다고 본 줄만 180° 돌린 것도 인식해 보고, 점수가 높은 쪽을 쓴다(분류기 오판 방지)
    const flip = this.opts.textlineOrientation ? await this.upsideDownCandidates(crops) : []
    const all = await this.recognize([...crops, ...flip.map(i => rotate(crops[i], 180))])
    const texts = all.slice(0, crops.length)
    flip.forEach((i, j) => { if (all[crops.length + j].score > texts[i].score) texts[i] = all[crops.length + j] })

    const [iw, ih] = dims(img)
    return lines.flatMap((l, i) => texts[i].text ? [{
      text: texts[i].text, recScore: texts[i].score, detScore: l.score,
      box: l.box.map(p => unrotate(p, angle, iw, ih)),
    }] : [])
  }

  // ── 내부 단계 ──────────────────────────────────────────────────────────────

  private async run(sess: Session, data: Float32Array, shape: number[]) {
    const out = await sess.run({ [sess.inputNames[0]]: new this.ort.Tensor('float32', data, shape) })
    const t = out[sess.outputNames[0]]
    const res = { data: await t.getData() as Float32Array, dims: t.dims as number[] }
    t.dispose()
    return res
  }

  private async docOrientation(img: Src): Promise<{ angle: 0 | 90 | 180 | 270; image: Src }> {
    const { data } = await this.run(this.docOri, preprocessDocOri(img), [1, 3, 224, 224])
    const p = softmax(data, 0, 4)
    const cls = p.indexOf(Math.max(...p))
    // 오판 하나가 문서 전체를 망치므로 확신할 때만 돌린다
    if (cls === 0 || p[cls] < 0.85) return { angle: 0, image: img }
    const angle = ([0, 90, 180, 270] as const)[cls]
    // 분류 결과 = 반시계로 돌아간 각도 → 시계 방향으로 되돌린다
    return { angle, image: rotate(img, angle as 90 | 180 | 270) }
  }

  private async detect(img: Src) {
    const [ow, oh] = dims(img)
    const { tensor, newH, newW, ratioH, ratioW } = preprocessDet(img, this.opts.detMaxSide, this.opts.detMinSide)
    const { data } = await this.run(this.det, tensor, [1, 3, newH, newW])
    return dbPostprocess(data, newH, newW, ratioH, ratioW, oh, ow, 0.3, this.opts.boxThresh)
  }

  private async upsideDownCandidates(crops: OffscreenCanvas[]): Promise<number[]> {
    const out: number[] = []
    for (let i = 0; i < crops.length; i += 16) {
      const batch = crops.slice(i, i + 16)
      const { data } = await this.run(this.textlineOri, preprocessTextlineOri(batch), [batch.length, 3, 80, 160])
      batch.forEach((_, j) => { if (softmax(data, j * 2, 2)[1] > 0.5) out.push(i + j) })
    }
    return out
  }

  private async recognize(crops: OffscreenCanvas[]): Promise<{ text: string; score: number }[]> {
    const res: { text: string; score: number }[] = new Array(crops.length)
    // 비슷한 비율끼리 묶어 패딩을 줄인다
    const order = crops.map((c, i) => [c.width / c.height, i] as const).sort((a, b) => a[0] - b[0]).map(x => x[1])
    const B = this.backend === 'webgpu' ? 16 : 6
    for (let i = 0; i < order.length; i += B) {
      const idx = order.slice(i, i + B), batch = idx.map(k => crops[k])
      const W = recWidth(batch)
      const { data, dims: [, T, C] } = await this.run(this.rec, preprocessRec(batch, W), [batch.length, 3, 48, W])
      idx.forEach((k, j) => { res[k] = ctcDecode(data, j * T * C, T, C, this.dict) })
    }
    return res
  }
}

/** 방향 보정으로 돌린 이미지 좌표 → 원래 이미지 좌표 */
function unrotate([x, y]: Pt, angle: 0 | 90 | 180 | 270, w: number, h: number): Pt {
  if (angle === 90) return [y, h - x]
  if (angle === 180) return [w - x, h - y]
  if (angle === 270) return [w - y, x]
  return [x, y]
}
