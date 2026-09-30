import Icon from './Icon'

interface Props {
  backend: string | null
}

export default function Header({ backend }: Props) {
  return (
    <section className="header" aria-label="소개">
      <div className="header-inner">
        <p className="eyebrow"><Icon name="shield" /> On-device · OCR + LLM</p>
        <h1>사진 속 한국어를, 바로 텍스트로.</h1>
        <p>서버 없이 이 브라우저에서 글자를 읽고 요약합니다. 사진은 기기 밖으로 나가지 않습니다.</p>

        <div className="model-selector">
          <span className="model-btn active" title="PP-OCRv5 한국어 파이프라인 (문서 방향 → 검출 → 줄 방향 → 인식)">PP-OCRv5</span>
          {backend && (
            <span className="model-btn" title={backend === 'webgpu' ? 'GPU 로 추론합니다' : '이 브라우저는 WebGPU 를 지원하지 않아 CPU 로 추론합니다'}>
              {backend === 'webgpu' ? 'WebGPU' : 'CPU (WASM)'}
            </span>
          )}
        </div>
      </div>
    </section>
  )
}
