import { useState, useCallback }         from 'react'
import { loadLLM, summarize, isLLMLoaded } from '../llm/engine'
import type { LlmStatus, LLMLoadProgress } from '../types'

export function useLlm() {
  const [status,   setStatus]   = useState<LlmStatus>('idle')
  const [progress, setProgress] = useState<LLMLoadProgress | null>(null)
  const [error,    setError]    = useState('')
  const [summary,  setSummary]  = useState('')

  const runAnalysis = useCallback(async (texts: string[]) => {
    if (!texts.length) return
    setError('')
    setSummary('')

    if (!isLLMLoaded()) {
      setStatus('loading-model')
      setProgress(null)
      try {
        await loadLLM(p => setProgress(p))
      } catch (e) {
        setStatus('error')
        setError(String(e))
        return
      }
    }

    setStatus('running')
    try {
      await summarize(texts, token => setSummary(prev => prev + token))
      setStatus('done')
    } catch (e) {
      setStatus('error')
      setError(String(e))
    }
  }, [])

  const reset = useCallback(() => {
    setStatus('idle')
    setSummary('')
    setError('')
    setProgress(null)
  }, [])

  return {
    llmStatus:   status,
    llmProgress: progress,
    llmError:    error,
    summary,
    runAnalysis,
    reset,
  }
}
