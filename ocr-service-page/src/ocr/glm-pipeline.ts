/**
 * GLM-OCR VLM 파이프라인 (zai-org/GLM-OCR)
 *
 * 구조:
 *   1. PP-OCRv5_mobile_det  — 텍스트 박스 검출 (기존 PP-OCR 검출기 재사용)
 *   2. encoder_model.onnx   — CogViT 이미지 인코더
 *   3. decoder_model_merged.onnx — GLM-0.5B 디코더 (KV-cache 포함)
 *
 * 필요 파일 (public/models/glm-ocr/):
 *   encoder_model.onnx
 *   decoder_model_merged.onnx
 *   tokenizer.json
 *   preprocessor_config.json
 *   config.json
 *
 * convert_glm_ocr.py 를 실행하면 위 파일이 자동 생성됩니다.
 */

import * as ort from './ort'
import { preprocessDet, cropByBox } from './preprocess'
import { dbPostprocess } from './dbPostprocess'
import type { OcrItem, LoadProgress } from './pipeline'

// ── 타입 ─────────────────────────────────────────────────────────────────────

interface PreprocessorConfig {
  image_size?:    number | { height: number; width: number }
  size?:          number | { height: number; width: number }
  crop_size?:     number | { height: number; width: number }
  image_mean?:    number[]
  image_std?:     number[]
  do_normalize?:  boolean
  do_resize?:     boolean
}

interface GenerationConfig {
  eos_token_id?:    number | number[]
  pad_token_id?:    number
  max_new_tokens?:  number
  bos_token_id?:    number
}

// ── 이미지 전처리 ────────────────────────────────────────────────────────────

/** CogViT 표준 전처리 기본값 (preprocessor_config.json 로 덮어씀) */
const DEFAULT_IMG_SIZE  = 336
const DEFAULT_IMG_MEAN  = [0.48145466, 0.4578275,  0.40821073]
const DEFAULT_IMG_STD   = [0.26862954, 0.26130258, 0.27577711]

function getImageSize(cfg: PreprocessorConfig): number {
  const s = cfg.image_size ?? cfg.size ?? cfg.crop_size ?? DEFAULT_IMG_SIZE
  if (typeof s === 'number') return s
  return s.height   // height == width (정사각형 입력)
}

function preprocessImage(
  src:    OffscreenCanvas | HTMLImageElement,
  size:   number,
  mean:   number[],
  std:    number[],
): Float32Array {
  const canvas = new OffscreenCanvas(size, size)
  const ctx    = canvas.getContext('2d')!
  ctx.drawImage(src as CanvasImageSource, 0, 0, size, size)
  const { data } = ctx.getImageData(0, 0, size, size)

  const out = new Float32Array(3 * size * size)
  for (let ch = 0; ch < 3; ch++) {
    for (let i = 0; i < size * size; i++) {
      const v = data[i * 4 + ch] / 255
      out[ch * size * size + i] = (v - mean[ch]) / std[ch]
    }
  }
  return out
}

// ── 토크나이저 (tokenizer.json BPE) ──────────────────────────────────────────

interface TokenizerJson {
  model:       { vocab: Record<string, number>; merges: string[] }
  added_tokens?: Array<{ id: number; content: string }>
  post_processor?: unknown
}

class SimpleTokenizer {
  private vocab:   Map<string, number>
  private decoder: Map<number, string>
  private merges:  Map<string, number>
  readonly eosId:  number
  readonly bosId:  number
  readonly padId:  number

  constructor(tj: TokenizerJson, genCfg: GenerationConfig) {
    this.vocab   = new Map(Object.entries(tj.model.vocab))
    this.decoder = new Map(Object.entries(tj.model.vocab).map(([k, v]) => [v, k]))

    // special tokens override from added_tokens
    tj.added_tokens?.forEach(t => {
      this.vocab.set(t.content, t.id)
      this.decoder.set(t.id, t.content)
    })

    this.merges = new Map(
      tj.model.merges.map((m, i) => [m.replace(' ', 'Ġ'), i])
    )

    const eosRaw = genCfg.eos_token_id
    this.eosId   = Array.isArray(eosRaw) ? eosRaw[0] : (eosRaw ?? 2)
    this.bosId   = genCfg.bos_token_id ?? 1
    this.padId   = genCfg.pad_token_id ?? 0
  }

  encode(text: string): number[] {
    // UTF-8 바이트 → BPE
    const bytes  = new TextEncoder().encode(text)
    let symbols: string[] = [...bytes].map(b => String.fromCharCode(b))

    // BPE merges
    let changed = true
    while (changed) {
      changed = false
      let best = Infinity, bestIdx = -1
      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i] + 'Ġ' + symbols[i + 1]
        const rank = this.merges.get(pair) ?? Infinity
        if (rank < best) { best = rank; bestIdx = i }
      }
      if (bestIdx >= 0) {
        symbols = [
          ...symbols.slice(0, bestIdx),
          symbols[bestIdx] + symbols[bestIdx + 1],
          ...symbols.slice(bestIdx + 2),
        ]
        changed = true
      }
    }

    return symbols.map(s => this.vocab.get(s) ?? this.vocab.get('<unk>') ?? 0)
  }

  decode(ids: number[]): string {
    const tokens = ids
      .filter(id => id !== this.eosId && id !== this.bosId && id !== this.padId)
      .map(id => this.decoder.get(id) ?? '')
    // G byte-level 디코딩
    const bytes = new Uint8Array(tokens.join('').split('').map(c => c.charCodeAt(0)))
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes)
  }
}

// ── 프롬프트 빌더 ────────────────────────────────────────────────────────────

function buildInputIds(
  tokenizer: SimpleTokenizer,
  prompt:    string,
): Int32Array {
  const ids = [
    tokenizer.bosId,
    ...tokenizer.encode(prompt),
  ]
  return new Int32Array(ids)
}

// ── 그리디 디코딩 ────────────────────────────────────────────────────────────

async function greedyDecode(
  decoder:        ort.InferenceSession,
  encoderOut:     ort.Tensor,
  startIds:       Int32Array,
  eosId:          number,
  maxNewTokens:   number = 128,
): Promise<{ ids: number[]; avgLogprob: number }> {
  const generated: number[] = []
  const logprobs:  number[] = []

  // 입력 이름 확인
  const inputNames  = decoder.inputNames
  const outputNames = decoder.outputNames

  // 초기 디코더 입력 (KV-cache 없음, 첫 스텝)
  let currentIds = new Int32Array(startIds)
  let pastKVFeed: Record<string, ort.Tensor> = {}

  for (let step = 0; step < maxNewTokens; step++) {
    const seqLen = currentIds.length
    const attMask = new Int32Array(seqLen).fill(1)

    // 기본 입력 구성
    const feed: Record<string, ort.Tensor> = {
      input_ids:              new ort.Tensor('int32', currentIds, [1, seqLen]),
      attention_mask:         new ort.Tensor('int32', attMask,    [1, seqLen]),
    }

    // encoder_hidden_states (있을 경우)
    if (inputNames.includes('encoder_hidden_states')) {
      feed['encoder_hidden_states'] = encoderOut
    }

    // past_key_values (KV-cache) 주입
    Object.assign(feed, pastKVFeed)

    const out = await decoder.run(feed)

    // logits: [1, seq_len, vocab_size] → 마지막 토큰
    const logits = out[outputNames[0]].data as Float32Array
    const dims   = out[outputNames[0]].dims
    const vocabSize = dims[dims.length - 1]
    const lastOffset = (dims[1] - 1) * vocabSize  // 마지막 시퀀스 위치

    let maxVal = -Infinity, nextToken = 0
    for (let v = 0; v < vocabSize; v++) {
      if (logits[lastOffset + v] > maxVal) {
        maxVal = logits[lastOffset + v]; nextToken = v
      }
    }

    generated.push(nextToken)
    logprobs.push(maxVal)

    if (nextToken === eosId) break

    // KV-cache 업데이트
    pastKVFeed = {}
    for (const name of outputNames.slice(1)) {
      if (name.includes('key') || name.includes('value') || name.includes('past')) {
        pastKVFeed[name.replace('present', 'past')] = out[name]
      }
    }

    // 다음 스텝은 새로 생성된 토큰만
    currentIds = new Int32Array([nextToken])
  }

  const avgLogprob = logprobs.length
    ? logprobs.reduce((a, b) => a + b) / logprobs.length
    : 0

  return { ids: generated, avgLogprob }
}

// ── GLM-OCR 클래스 ────────────────────────────────────────────────────────────

export class GlmOCR {
  private det:       ort.InferenceSession
  private encoder:   ort.InferenceSession
  private decoder:   ort.InferenceSession
  private tokenizer: SimpleTokenizer
  private imgSize:   number
  private imgMean:   number[]
  private imgStd:    number[]
  private maxNew:    number
  private eosId:     number
  private ocrPrompt: string

  private constructor(
    det:       ort.InferenceSession,
    encoder:   ort.InferenceSession,
    decoder:   ort.InferenceSession,
    tokenizer: SimpleTokenizer,
    imgSize:   number,
    imgMean:   number[],
    imgStd:    number[],
    maxNew:    number,
    eosId:     number,
    ocrPrompt: string,
  ) {
    this.det       = det
    this.encoder   = encoder
    this.decoder   = decoder
    this.tokenizer = tokenizer
    this.imgSize   = imgSize
    this.imgMean   = imgMean
    this.imgStd    = imgStd
    this.maxNew    = maxNew
    this.eosId     = eosId
    this.ocrPrompt = ocrPrompt
  }

  static async create(
    onProgress?: (p: LoadProgress) => void,
  ): Promise<GlmOCR> {
    const base   = import.meta.env.BASE_URL
    const modelBase = `${base}models/glm-ocr/`
    const total  = 5
    const notify = (step: string, done: number) =>
      onProgress?.({ step, done, total })

    // ── 설정 파일 로드 ─────────────────────────────────────────────────────
    notify('GLM-OCR 설정 로드 중...', 0)

    const [ppCfgRes, genCfgRes, tokRes] = await Promise.all([
      fetch(`${modelBase}preprocessor_config.json`),
      fetch(`${modelBase}generation_config.json`),
      fetch(`${modelBase}tokenizer.json`),
    ])

    if (!tokRes.ok) {
      throw new Error(
        `GLM-OCR 파일을 찾을 수 없습니다.\n` +
        `convert_glm_ocr.py 를 실행해 public/models/glm-ocr/ 를 생성하세요.`
      )
    }

    const ppCfg:  PreprocessorConfig = ppCfgRes.ok  ? await ppCfgRes.json()  : {}
    const genCfg: GenerationConfig   = genCfgRes.ok ? await genCfgRes.json() : {}
    const tokJson: TokenizerJson     = await tokRes.json()

    const imgSize = getImageSize(ppCfg)
    const imgMean = ppCfg.image_mean ?? DEFAULT_IMG_MEAN
    const imgStd  = ppCfg.image_std  ?? DEFAULT_IMG_STD
    const maxNew  = genCfg.max_new_tokens ?? 256
    const tokenizer = new SimpleTokenizer(tokJson, genCfg)

    // GLM chat template: <|user|>\n{prompt}\n<|assistant|>\n
    const ocrPrompt = '<|user|>\nOCR the text in the image.\n<|assistant|>\n'

    // ── 검출 모델 ─────────────────────────────────────────────────────────
    notify('텍스트 검출 모델 로드 중...', 1)
    const detUrl = `${base}models/PP-OCRv5_mobile_det.onnx`
    const detCheck = await fetch(detUrl, { method: 'HEAD' })
    if (!detCheck.ok) throw new Error('PP-OCRv5_mobile_det.onnx 를 찾을 수 없습니다.')
    const det = await ort.InferenceSession.create(detUrl, { executionProviders: ['wasm'] })

    // ── 인코더 모델 ───────────────────────────────────────────────────────
    notify('GLM-OCR 인코더 로드 중...', 2)
    const encUrl = `${modelBase}encoder_model.onnx`
    const encCheck = await fetch(encUrl, { method: 'HEAD' })
    if (!encCheck.ok) {
      throw new Error(
        `encoder_model.onnx 를 찾을 수 없습니다.\n` +
        `convert_glm_ocr.py --quantize 를 실행해 ONNX 파일을 생성하세요.`
      )
    }
    const encoder = await ort.InferenceSession.create(encUrl, { executionProviders: ['wasm'] })

    // ── 디코더 모델 ───────────────────────────────────────────────────────
    notify('GLM-OCR 디코더 로드 중...', 3)
    const decUrl = `${modelBase}decoder_model_merged.onnx`
    const decCheck = await fetch(decUrl, { method: 'HEAD' })
    if (!decCheck.ok) throw new Error('decoder_model_merged.onnx 를 찾을 수 없습니다.')
    const decoder = await ort.InferenceSession.create(decUrl, { executionProviders: ['wasm'] })

    notify('완료', total)
    return new GlmOCR(
      det, encoder, decoder, tokenizer,
      imgSize, imgMean, imgStd, maxNew, tokenizer.eosId, ocrPrompt,
    )
  }

  /** 이미지에서 OCR 수행 */
  async predict(img: HTMLImageElement): Promise<OcrItem[]> {
    const rW = img.naturalWidth
    const rH = img.naturalHeight

    // ── 검출 ─────────────────────────────────────────────────────────────
    const detectedBoxes = await this._detect(img, rW, rH)
    const boxes = detectedBoxes.length > 0
      ? detectedBoxes
      : [{ box: [[0, 0], [rW, 0], [rW, rH], [0, rH]] as [number, number][], score: 1.0 }]

    const items: OcrItem[] = []
    for (const { box, score: detScore } of boxes) {
      const crop = cropByBox(img, box as [number, number][])
      if (!crop) continue

      const { text, score: recScore } = await this._recognize(crop)
      if (!text.trim()) continue

      items.push({ text: text.trim(), recScore, detScore, box: box as [number, number][] })
    }
    return items
  }

  // ── 내부 단계 ─────────────────────────────────────────────────────────────

  private async _detect(img: HTMLImageElement, origW: number, origH: number) {
    const { tensor: data, newH, newW, ratioH, ratioW } = preprocessDet(img)
    const tensor = new ort.Tensor('float32', data, [1, 3, newH, newW])
    const out    = await this.det.run({ [this.det.inputNames[0]]: tensor })
    const pred   = out[this.det.outputNames[0]].data as Float32Array
    return dbPostprocess(pred, newH, newW, ratioH, ratioW, origH, origW)
  }

  private async _recognize(
    crop: OffscreenCanvas,
  ): Promise<{ text: string; score: number }> {
    // ── 이미지 인코딩 ──────────────────────────────────────────────────────
    const pixelData = preprocessImage(crop, this.imgSize, this.imgMean, this.imgStd)
    const pixelTensor = new ort.Tensor('float32', pixelData, [1, 3, this.imgSize, this.imgSize])

    const encFeed: Record<string, ort.Tensor> = {}
    // 인코더 입력 이름에 따라 동적으로 매핑
    for (const name of this.encoder.inputNames) {
      if (name === 'pixel_values') {
        encFeed[name] = pixelTensor
      }
    }
    if (Object.keys(encFeed).length === 0) {
      // 첫 번째 입력에 pixel_values 바인딩 (이름이 다를 경우)
      encFeed[this.encoder.inputNames[0]] = pixelTensor
    }

    const encOut   = await this.encoder.run(encFeed)
    const encTensor = encOut[this.encoder.outputNames[0]]  // [1, seq, hidden]

    // ── 텍스트 생성 ────────────────────────────────────────────────────────
    const startIds = buildInputIds(this.tokenizer, this.ocrPrompt)
    const { ids, avgLogprob } = await greedyDecode(
      this.decoder, encTensor, startIds, this.eosId, this.maxNew,
    )

    const text  = this.tokenizer.decode(ids)
    // softmax 기반 점수 대신 logprob → 0~1 sigmoid 근사
    const score = 1 / (1 + Math.exp(-avgLogprob / 10))

    return { text, score: Math.max(0, Math.min(1, score)) }
  }
}
