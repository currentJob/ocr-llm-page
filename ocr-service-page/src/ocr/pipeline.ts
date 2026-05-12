import * as ort from 'onnxruntime-web'
import {
  preprocessDocOri,
  preprocessDet,
  preprocessTextlineOri,
  preprocessRec,
  cropByBox,
} from './preprocess'
import { dbPostprocess } from './dbPostprocess'

// WASM 파일은 setup_web_models.py 가 public/ 루트에 복사해 둠
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths  = import.meta.env.BASE_URL

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

// ── 문자 사전 ─────────────────────────────────────────────────────────────────

interface CharDict {
  charTable: (string | null)[]
  blankIdx:  number
}

async function loadCharDict(): Promise<CharDict> {
  const res = await fetch(`${import.meta.env.BASE_URL}models/charDict.json`)
  if (!res.ok) throw new Error('charDict.json 로드 실패')
  return res.json()
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

// ── 메인 파이프라인 ───────────────────────────────────────────────────────────

export class KoreanOCR {
  private docOri:     ort.InferenceSession
  private det:        ort.InferenceSession
  private textlineOri:ort.InferenceSession
  private rec:        ort.InferenceSession
  private charDict:   CharDict
  private recMaxW:    number

  private constructor(
    docOri: ort.InferenceSession,
    det:    ort.InferenceSession,
    tlo:    ort.InferenceSession,
    rec:    ort.InferenceSession,
    dict:   CharDict,
  ) {
    this.docOri      = docOri
    this.det         = det
    this.textlineOri = tlo
    this.rec         = rec
    this.charDict    = dict

    this.recMaxW = 320   // rec 입력 너비 기본값
  }

  /** 모델 로딩 (진행 콜백 선택) */
  static async create(
    onProgress?: (p: LoadProgress) => void,
  ): Promise<KoreanOCR> {
    const steps = [
      'PP-LCNet_x1_0_doc_ori',
      'PP-OCRv5_mobile_det',
      'PP-LCNet_x1_0_textline_ori',
      'korean_PP-OCRv5_mobile_rec',
    ]
    const total = steps.length + 1   // +1 for charDict

    const notify = (step: string, done: number) =>
      onProgress?.({ step, done, total })

    notify('문자 사전 로드 중...', 0)
    const charDict = await loadCharDict()

    const sessions: ort.InferenceSession[] = []
    for (let i = 0; i < steps.length; i++) {
      notify(`${steps[i]} 로드 중...`, i + 1)
      const sess = await ort.InferenceSession.create(
        `${import.meta.env.BASE_URL}models/${steps[i]}.onnx`,
        { executionProviders: ['wasm'] },
      )
      sessions.push(sess)
    }

    notify('완료', total)
    return new KoreanOCR(sessions[0], sessions[1], sessions[2], sessions[3], charDict)
  }

  /** 이미지에서 OCR 수행 */
  async predict(img: HTMLImageElement): Promise<OcrItem[]> {
    // 1. 문서 방향 보정
    const rotated = await this._correctDocOrientation(img)
    const rW = rotated.naturalWidth
    const rH = rotated.naturalHeight

    // 2. 텍스트 검출
    const detectedBoxes = await this._detect(rotated, rW, rH)

    // 검출 실패 시 전체 이미지를 단일 텍스트 라인으로 처리 (이미 크롭된 한 줄 이미지 대응)
    const boxes = detectedBoxes.length > 0
      ? detectedBoxes
      : [{ box: [[0,0],[rW,0],[rW,rH],[0,rH]] as [number,number][], score: 1.0 }]

    // 3. 각 텍스트 라인 인식
    const items: OcrItem[] = []
    for (const { box, score: detScore } of boxes) {
      const crop = cropByBox(rotated, box as [number,number][])
      if (!crop) continue

      const oriented = await this._correctTextlineOrientation(crop)
      const { text, score: recScore } = await this._recognize(oriented)
      if (!text) continue

      items.push({ text, recScore, detScore, box: box as [number,number][] })
    }
    return items
  }

  // ── 내부 단계 ──────────────────────────────────────────────────────────────

  private async _correctDocOrientation(img: HTMLImageElement): Promise<HTMLImageElement> {
    const data   = preprocessDocOri(img)
    const tensor = new ort.Tensor('float32', data, [1, 3, 224, 224])
    const out    = await this.docOri.run({ [this.docOri.inputNames[0]]: tensor })
    const logits = out[this.docOri.outputNames[0]].data as Float32Array
    const cls    = argmax(logits)
    const angles = [0, 90, 180, 270] as const
    const angle  = angles[cls]

    if (angle === 0) return img
    return rotateImage(img, angle)
  }

  private async _detect(
    img: HTMLImageElement,
    origW: number,
    origH: number,
  ) {
    const { tensor: data, newH, newW, ratioH, ratioW } = preprocessDet(img)
    const tensor = new ort.Tensor('float32', data, [1, 3, newH, newW])
    const out    = await this.det.run({ [this.det.inputNames[0]]: tensor })

    const pred = out[this.det.outputNames[0]].data as Float32Array
    // shape: [1, 1, H, W] → skip first 2 dims
    return dbPostprocess(pred, newH, newW, ratioH, ratioW, origH, origW)
  }

  private async _correctTextlineOrientation(
    crop: OffscreenCanvas,
  ): Promise<OffscreenCanvas> {
    const data   = preprocessTextlineOri(crop)
    const tensor = new ort.Tensor('float32', data, [1, 3, 80, 160])
    const out    = await this.textlineOri.run({
      [this.textlineOri.inputNames[0]]: tensor,
    })
    const logits = out[this.textlineOri.outputNames[0]].data as Float32Array
    if (argmax(logits) !== 1) return crop   // 정방향

    // 180° 회전
    const w   = crop.width
    const h   = crop.height
    const rot = new OffscreenCanvas(w, h)
    const ctx = rot.getContext('2d')!
    ctx.translate(w, h)
    ctx.rotate(Math.PI)
    ctx.drawImage(crop, 0, 0)
    return rot
  }

  private async _recognize(
    crop: OffscreenCanvas,
  ): Promise<{ text: string; score: number }> {
    const data   = preprocessRec(crop, this.recMaxW)
    const tensor = new ort.Tensor('float32', data, [1, 3, 48, this.recMaxW])
    const out    = await this.rec.run({ [this.rec.inputNames[0]]: tensor })

    const probs = out[this.rec.outputNames[0]].data as Float32Array
    // shape: [1, T, numCls]
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

// ── 유틸 ─────────────────────────────────────────────────────────────────────

function argmax(arr: Float32Array): number {
  let maxIdx = 0
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[maxIdx]) maxIdx = i
  }
  return maxIdx
}

function rotateImage(img: HTMLImageElement, angle: 90 | 180 | 270): Promise<HTMLImageElement> {
  const w = angle === 180 ? img.naturalWidth  : img.naturalHeight
  const h = angle === 180 ? img.naturalHeight : img.naturalWidth

  const canvas = document.createElement('canvas')
  canvas.width  = w
  canvas.height = h
  const ctx     = canvas.getContext('2d')!

  ctx.translate(w / 2, h / 2)
  ctx.rotate((angle * Math.PI) / 180)
  ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2)

  return new Promise<HTMLImageElement>((resolve) => {
    const out = new Image()
    out.onload = () => resolve(out)
    out.src = canvas.toDataURL()
  })
}
