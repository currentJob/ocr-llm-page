import type { OcrItem, LoadProgress } from './ocr/pipeline'
import type { LLMLoadProgress }       from './llm/engine'

export type { OcrItem, LoadProgress, LLMLoadProgress }

// export type OcrModelType = 'ppocr' | 'glm-ocr'
export type OcrModelType = 'ppocr'
export type Phase        = 'loading-model' | 'ready' | 'roi' | 'running' | 'done' | 'error'
export type LlmStatus    = 'idle' | 'loading-model' | 'running' | 'done' | 'error'

export interface Roi {
  x: number; y: number; w: number; h: number
}

export interface HistoryEntry {
  id:       string
  ts:       number
  thumb:    string
  natW?:    number
  natH?:    number
  filename: string
  items:    OcrItem[]
}

export const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
  '#8b5cf6', '#06b6d4', '#84cc16', '#f97316',
] as const
