import { useRef, useCallback } from 'react'
import { KoreanOCR }           from '../ocr/pipeline'
import type { LoadProgress }   from '../types'

export function useOcrModel() {
  const ocrRef = useRef<KoreanOCR | null>(null)

  const loadModel = useCallback(async (onProgress: (p: LoadProgress) => void) => {
    ocrRef.current = null
    ocrRef.current = await KoreanOCR.create(onProgress)
  }, [])

  return { ocrRef, loadModel }
}
