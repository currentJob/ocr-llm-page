/**
 * 브라우저 내 LLM 추론 엔진 — @huggingface/transformers (Transformers.js v3)
 *
 * 별도 서버 없이 브라우저에서 ONNX 모델을 직접 실행합니다.
 * 첫 실행 시 HuggingFace CDN에서 다운로드 후 브라우저에 영구 캐시합니다.
 *
 * 오프라인 선사용: python convert_llm.py
 *   → public/models/llm/onnx-community/Qwen2.5-0.5B-Instruct/ 에 저장
 *   → 이후 자동으로 로컬 파일 우선 사용
 */

import { pipeline, TextStreamer, env } from '@huggingface/transformers'

// ── 설정 ───────────────────────────────────────────────────────────────────────

// 항상 org 포함 HF 모델 ID 사용 (로컬/리모트 모두 동일)
const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct'

export interface LLMLoadProgress {
  file:     string
  loaded:   number   // bytes
  total:    number   // bytes
  progress: number   // 0-100
}

// ── 내부 상태 ─────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _pipe: any = null

// ── 모델 소스 감지 ─────────────────────────────────────────────────────────────

async function setupEnv(): Promise<void> {
  // 브라우저에서 모든 HTTP fetch는 allowRemoteModels = true 필요
  env.allowRemoteModels = true

  // 로컬 파일이 있으면 CDN 대신 로컬 서버 사용
  // 파일 경로는 HF org/model 구조를 그대로 반영해야 함
  //   public/models/llm/onnx-community/Qwen2.5-0.5B-Instruct/config.json
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}models/llm/${MODEL_ID}/config.json`, { method: 'HEAD' })
    if (res.ok) {
      env.localModelPath = `${location.origin}${import.meta.env.BASE_URL}models/llm/`
      console.info('[LLM] 로컬 모델 파일 사용:', `/models/llm/${MODEL_ID}/`)
      return
    }
  } catch { /* 로컬 파일 없음 */ }

  console.info('[LLM] HuggingFace CDN 사용 (브라우저 캐시 적용)')
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/** 모델이 이미 메모리에 로드되었는지 확인 */
export function isLLMLoaded(): boolean { return _pipe !== null }

/**
 * LLM 모델 로딩 (lazy — 첫 호출 시만 초기화)
 * 로컬 파일이 있으면 로컬 우선, 없으면 HuggingFace CDN
 */
export async function loadLLM(onProgress?: (p: LLMLoadProgress) => void): Promise<void> {
  if (_pipe) return

  await setupEnv()

  // WebGPU 우선, 미지원 시 CPU(WASM) 자동 폴백
  const device: 'webgpu' | 'cpu' = 'gpu' in navigator ? 'webgpu' : 'cpu'
  console.info('[LLM] 백엔드:', device)

  const load = (dev: 'webgpu' | 'cpu') =>
    pipeline('text-generation', MODEL_ID, {
      dtype:  'q4',
      device: dev,
      progress_callback: (info: Record<string, unknown>) => {
        if (info.status === 'progress' && onProgress) {
          onProgress({
            file:     String(info.file     ?? ''),
            loaded:   Number(info.loaded   ?? 0),
            total:    Number(info.total    ?? 0),
            progress: Number(info.progress ?? 0),
          })
        }
      },
    })

  try {
    _pipe = await load(device)
  } catch (e) {
    if (device === 'webgpu') {
      console.warn('[LLM] WebGPU 실패, WASM으로 재시도:', e)
      _pipe = await load('cpu')
    } else {
      throw e
    }
  }
}

/** 모델을 메모리에서 해제 */
export function unloadLLM(): void { _pipe = null }

// ── 텍스트 생성 ────────────────────────────────────────────────────────────────

function buildMessages(texts: string[]): Array<{ role: string; content: string }> {
  const list = texts.map((t, i) => `${i + 1}. ${t}`).join('\n')
  return [
    {
      role:    'system',
      content: '당신은 OCR로 추출된 한국어 텍스트를 분석하고 체계적으로 정리하는 AI 어시스턴트입니다. 간결하고 명확하게 답변하세요.',
    },
    {
      role:    'user',
      content: `이미지에서 OCR로 추출된 텍스트 목록입니다:\n\n${list}\n\n위 내용을 다음 형식으로 정리해주세요:\n\n**문서 유형**: (영수증/명함/문서/표지판/기타)\n**요약**: (1~2문장)\n**핵심 정보**:\n- (중요 항목들)\n\n**정리된 텍스트**:\n(읽기 좋은 순서로 재배열)`,
    },
  ]
}

/**
 * OCR 텍스트 분석 및 요약 (토큰 스트리밍)
 * loadLLM() 을 먼저 호출해야 합니다.
 */
export async function summarize(
  texts:   string[],
  onToken: (token: string) => void,
): Promise<void> {
  if (!_pipe) throw new Error('LLM이 로딩되지 않았습니다. loadLLM()을 먼저 호출하세요.')

  const messages = buildMessages(texts)

  const streamer = new TextStreamer(_pipe.tokenizer, {
    skip_prompt:         true,
    skip_special_tokens: true,
    callback_function:   (token: string) => onToken(token),
  })

  await _pipe(messages, {
    max_new_tokens: 512,
    temperature:    0.3,
    do_sample:      true,
    streamer,
  })
}
