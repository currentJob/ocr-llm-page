import type { OcrModelType, Phase, HistoryEntry } from '../types'

interface Props {
  modelType:       OcrModelType
  phase:           Phase
  history:         HistoryEntry[]
  onSwitchModel:   (t: OcrModelType) => void
  onToggleHistory: () => void
}

const MODEL_LABELS: Record<OcrModelType, { label: string; title: string }> = {
  ppocr:    { label: 'PP-OCR',  title: 'PP-OCRv5 한국어 특화 4단계 파이프라인' },
  // 'glm-ocr':{ label: 'GLM-OCR', title: 'GLM-OCR ONNX 인식 모델' },
}

export default function Header({ modelType, phase, history, onSwitchModel, onToggleHistory }: Props) {
  return (
    <header className="header">
      <div className="header-inner">
        <h1>Korean OCR</h1>
        <p>정적 페이지 서비스로 별도의 서버 없이 한국어 텍스트 인식</p>

        <div className="model-selector">
          {(Object.entries(MODEL_LABELS) as [OcrModelType, { label: string; title: string }][]).map(([t, { label, title }]) => (
            <button
              key={t}
              className={`model-btn${modelType === t ? ' active' : ''}`}
              onClick={() => onSwitchModel(t)}
              disabled={phase === 'loading-model'}
              title={title}
            >
              {label}
            </button>
          ))}
        </div>

        {history.length > 0 && (
          <button className="history-toggle" onClick={onToggleHistory}>
            <svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 000-1.5h-3.25V5z" clipRule="evenodd"/>
            </svg>
            히스토리 {history.length}
          </button>
        )}
      </div>
    </header>
  )
}
