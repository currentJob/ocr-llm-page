import { useRef, useCallback } from 'react'
import { KoreanOCR }           from '../ocr/pipeline'
import { GlmOCR }              from '../ocr/glm-pipeline'
import type { OcrModelType, LoadProgress } from '../types'

export function useOcrModel() {
  const ocrRef = useRef<KoreanOCR | GlmOCR | null>(null)

  const loadModel = useCallback(async (
    type:       OcrModelType,
    onProgress: (p: LoadProgress) => void,
  ) => {
    ocrRef.current = null
    if (type === 'ppocr') {
      ocrRef.current = await KoreanOCR.create(onProgress)
    } else {
      ocrRef.current = await GlmOCR.create(onProgress)
    }
  }, [])

  return { ocrRef, loadModel }
}
