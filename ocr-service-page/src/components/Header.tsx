import type { OcrModelType, Phase } from '../types'
import Icon from './Icon'

interface Props {
  modelType:       OcrModelType
  phase:           Phase
  onSwitchModel:   (t: OcrModelType) => void
}

const MODEL_LABELS: Record<OcrModelType, { label: string; title: string }> = {
  ppocr:    { label: 'PP-OCR',  title: 'PP-OCRv5 한국어 특화 4단계 파이프라인' },
  // 'glm-ocr':{ label: 'GLM-OCR', title: 'GLM-OCR ONNX 인식 모델' },
}

export default function Header({ modelType, phase, onSwitchModel }: Props) {
  return (
    <section className="header" aria-label="소개">
      <div className="header-inner">
        <p className="eyebrow"><Icon name="shield" /> On-device · OCR + LLM</p>
        <h1>사진 속 한국어를, 바로 텍스트로.</h1>
        <p>서버 없이 이 브라우저에서 글자를 읽고 요약합니다. 사진은 기기 밖으로 나가지 않습니다.</p>

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

      </div>
    </section>
  )
}
