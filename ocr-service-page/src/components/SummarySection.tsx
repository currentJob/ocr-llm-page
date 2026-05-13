import type { LlmStatus, LLMLoadProgress } from '../types'

interface Props {
  status:   LlmStatus
  progress: LLMLoadProgress | null
  error:    string
  summary:  string
}

export default function SummarySection({ status, progress, error, summary }: Props) {
  return (
    <div className="summary-section">
      <div className="summary-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="15" height="15">
          <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <span>AI 분석 결과</span>
        {status === 'running'       && <span className="spinner sm" />}
        {status === 'done'          && <span className="summary-badge">완료</span>}
        {status === 'loading-model' && <span className="summary-tag">모델 로딩 중...</span>}
        {status === 'error'         && <span className="summary-badge error">오류</span>}
      </div>

      {status === 'loading-model' && (
        <div className="summary-model-loading">
          <p className="model-load-msg">
            {progress
              ? `다운로드 중: ${progress.file.split('/').pop() ?? ''}`
              : 'Qwen2.5-0.5B ONNX 모델 로딩 중... (첫 실행 시 ~350MB 다운로드)'}
          </p>
          {progress && (
            <>
              <div className="model-load-bar">
                <div className="model-load-fill" style={{ width: `${progress.progress}%` }} />
              </div>
              <p className="model-load-pct">{progress.progress.toFixed(0)}%</p>
            </>
          )}
        </div>
      )}

      {status === 'error' && <div className="summary-error">{error}</div>}

      {(status === 'running' || status === 'done') && summary && (
        <pre className="summary-text">
          {summary}
          {status === 'running' && <span className="cursor-blink">▋</span>}
        </pre>
      )}
    </div>
  )
}
