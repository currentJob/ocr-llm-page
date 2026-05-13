import type { RefObject } from 'react'

interface Props {
  videoRef:  RefObject<HTMLVideoElement | null>
  onStop:    () => void
  onCapture: () => void
}

export default function CameraOverlay({ videoRef, onStop, onCapture }: Props) {
  return (
    <div className="camera-overlay" onClick={e => { if (e.target === e.currentTarget) onStop() }}>
      <div className="camera-modal">
        <video ref={videoRef} autoPlay playsInline className="camera-video" />
        <div className="camera-actions">
          <button className="camera-btn cancel"  onClick={onStop}>취소</button>
          <button className="camera-btn capture" onClick={onCapture}>
            <span className="camera-shutter" />촬영
          </button>
        </div>
      </div>
    </div>
  )
}
