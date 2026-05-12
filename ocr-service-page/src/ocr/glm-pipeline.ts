import * as ort from 'onnxruntime-web'
import { preprocessDet, cropByBox } from './preprocess'
import { dbPostprocess } from './dbPostprocess'
import type { OcrItem, LoadProgress } from './pipeline'

// ── 타입 ──────────────────────────────────────────────────────────────────────

interface CharDict {
  charTable: (string | null)[]
  blankIdx:  number
}

// ── GLM-OCR 전처리 ────────────────────────────────────────────────────────────
// 기본값: H=64, W=256, [-1,1] 정규화
// 모델의 실제 입력 크기에 맞춰 targetH / maxW 조정 가능

function preprocessGlmRec(
  src:     OffscreenCanvas,
  maxW:    number = 256,
  targetH: number = 64,
): Float32Array {
  const sw  = src.width
  const sh  = src.height
  const newW = Math.max(1, Math.min(Math.ceil(targetH * sw / sh), maxW))

  const tmp = new OffscreenCanvas(newW, targetH)
  tmp.getContext('2d')!.drawImage(src, 0, 0, newW, targetH)
  const { data } = tmp.getContext('2d')!.getImageData(0, 0, newW, targetH)

  const out = new Float32Array(3 * targetH * maxW)
  for (let ch = 0; ch < 3; ch++) {
    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < newW; x++) {
        const v = data[(y * newW + x) * 4 + ch] / 255
        out[ch * targetH * maxW + y * maxW + x] = (v - 0.5) / 0.5
      }
    }
  }
  return out
}

// ── CTC 디코딩 ────────────────────────────────────────────────────────────────

function ctcDecode(
  data:      Float32Array,
  timeSteps: number,
  numCls:    number,
  charTable: (string | null)[],
  blankIdx:  number,
): { text: string; score: number } {
  const chars: string[] = []
  const confs: number[] = []
  let prev = -1

  for (let t = 0; t < timeSteps; t++) {
    let maxIdx = 0
    let maxVal = -Infinity
    const off  = t * numCls
    for (let c = 0; c < numCls; c++) {
      if (data[off + c] > maxVal) { maxVal = data[off + c]; maxIdx = c }
    }
    if (maxIdx === prev || maxIdx === blankIdx) { prev = maxIdx; continue }
    const ch = charTable[maxIdx]
    if (ch !== null && ch !== undefined) {
      chars.push(ch)
      confs.push(maxVal)
    }
    prev = maxIdx
  }

  const text  = chars.join('')
  const score = confs.length ? confs.reduce((a, b) => a + b) / confs.length : 0
  return { text, score }
}

// ── GLM-OCR 파이프라인 ────────────────────────────────────────────────────────
//
// 구조: PP-OCR 검출(PP-OCRv5_mobile_det) + GLM-OCR 인식(glm_ocr_rec.onnx)
//
// 모델 파일 위치:  public/models/glm_ocr_rec.onnx
// 어휘 파일 위치:  public/models/glm_vocab.json  (없으면 charDict.json 사용)
//
// glm_vocab.json 형식: { "charTable": ["a", "b", ...], "blankIdx": 0 }
//
// 인식 모델 입력 스펙 (기본값):
//   - shape : [1, 3, 64, 256]  → targetH=64, maxW=256
//   - dtype : float32
//   - 정규화: (pixel/255 - 0.5) / 0.5  →  [-1, 1]
// 인식 모델 출력 스펙:
//   - shape : [1, T, num_classes]  (CTC)
// 다를 경우 아래 _recognize() 내 주석 지점을 수정하세요.

export class GlmOCR {
  private det:      ort.InferenceSession
  private rec:      ort.InferenceSession
  private charDict: CharDict
  private recH:     number
  private recMaxW:  number

  private constructor(
    det:      ort.InferenceSession,
    rec:      ort.InferenceSession,
    charDict: CharDict,
    recH:     number,
    recMaxW:  number,
  ) {
    this.det      = det
    this.rec      = rec
    this.charDict = charDict
    this.recH     = recH
    this.recMaxW  = recMaxW
  }

  static async create(
    onProgress?: (p: LoadProgress) => void,
    recH    = 64,
    recMaxW = 256,
  ): Promise<GlmOCR> {
    const base  = import.meta.env.BASE_URL
    const total = 3   // charDict + det + rec
    const notify = (step: string, done: number) =>
      onProgress?.({ step, done, total })

    // 어휘 로드: glm_vocab.json 우선, 없으면 charDict.json 폴백
    notify('어휘 사전 로드 중...', 0)
    let charDict: CharDict
    const vocabRes = await fetch(`${base}models/glm_vocab.json`)
    if (vocabRes.ok) {
      charDict = await vocabRes.json()
    } else {
      const fallback = await fetch(`${base}models/charDict.json`)
      if (!fallback.ok) throw new Error('어휘 사전 로드 실패')
      charDict = await fallback.json()
    }

    notify('텍스트 검출 모델 로드 중...', 1)
    const det = await ort.InferenceSession.create(
      `${base}models/PP-OCRv5_mobile_det.onnx`,
      { executionProviders: ['wasm'] },
    )

    notify('GLM-OCR 인식 모델 로드 중...', 2)
    const recUrl = `${base}models/glm_ocr_rec.onnx`
    const recCheck = await fetch(recUrl, { method: 'HEAD' })
    if (!recCheck.ok) {
      throw new Error(
        `GLM-OCR 모델 파일을 찾을 수 없습니다.\n` +
        `public/models/glm_ocr_rec.onnx 파일을 추가해 주세요.`
      )
    }
    const rec = await ort.InferenceSession.create(
      recUrl,
      { executionProviders: ['wasm'] },
    )

    notify('완료', total)
    return new GlmOCR(det, rec, charDict, recH, recMaxW)
  }

  /** 이미지에서 OCR 수행 */
  async predict(img: HTMLImageElement): Promise<OcrItem[]> {
    const rW = img.naturalWidth
    const rH = img.naturalHeight

    // 검출 (PP-OCR 검출기 재사용)
    const detectedBoxes = await this._detect(img, rW, rH)
    const boxes = detectedBoxes.length > 0
      ? detectedBoxes
      : [{ box: [[0,0],[rW,0],[rW,rH],[0,rH]] as [number,number][], score: 1.0 }]

    const items: OcrItem[] = []
    for (const { box, score: detScore } of boxes) {
      const crop = cropByBox(img, box as [number,number][])
      if (!crop) continue

      const { text, score: recScore } = await this._recognize(crop)
      if (!text) continue

      items.push({ text, recScore, detScore, box: box as [number,number][] })
    }
    return items
  }

  // ── 내부 단계 ──────────────────────────────────────────────────────────────

  private async _detect(
    img:   HTMLImageElement,
    origW: number,
    origH: number,
  ) {
    const { tensor: data, newH, newW, ratioH, ratioW } = preprocessDet(img)
    const tensor = new ort.Tensor('float32', data, [1, 3, newH, newW])
    const out    = await this.det.run({ [this.det.inputNames[0]]: tensor })
    const pred   = out[this.det.outputNames[0]].data as Float32Array
    return dbPostprocess(pred, newH, newW, ratioH, ratioW, origH, origW)
  }

  private async _recognize(
    crop: OffscreenCanvas,
  ): Promise<{ text: string; score: number }> {
    // ── 전처리 ───────────────────────────────────────────────────────────────
    // 모델 입력 크기가 다를 경우 preprocessGlmRec 의 maxW / targetH 를 변경하세요.
    const data   = preprocessGlmRec(crop, this.recMaxW, this.recH)
    const tensor = new ort.Tensor('float32', data, [1, 3, this.recH, this.recMaxW])
    const out    = await this.rec.run({ [this.rec.inputNames[0]]: tensor })

    // ── 후처리 ───────────────────────────────────────────────────────────────
    // CTC 출력: shape [1, T, num_classes]
    const probs    = out[this.rec.outputNames[0]].data as Float32Array
    const outShape = out[this.rec.outputNames[0]].dims
    const T        = outShape[1]
    const numCls   = outShape[2]

    return ctcDecode(
      probs, T, numCls,
      this.charDict.charTable,
      this.charDict.blankIdx,
    )
  }
}
