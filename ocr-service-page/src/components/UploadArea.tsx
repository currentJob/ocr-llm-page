import { useState }                  from 'react'
import type { ChangeEvent, DragEvent } from 'react'

interface Props {
  onFiles:      (files: File[]) => void
  onCameraOpen: () => void
}

export default function UploadArea({ onFiles, onCameraOpen }: Props) {
  const [dragging, setDragging] = useState(false)

  function handleInput(e: ChangeEvent<HTMLInputElement>) {
    onFiles(Array.from(e.target.files ?? []))
    e.target.value = ''
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    onFiles(Array.from(e.dataTransfer.files))
  }

  return (
    <div className="upload-area">
      <label
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input type="file" accept="image/*" multiple onChange={handleInput} hidden />
        <div className="dropzone-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <p className="dropzone-text">클릭하거나 이미지를 드래그하세요</p>
        <p className="dropzone-sub">PNG · JPG · WEBP · BMP · 여러 파일 선택 가능</p>
      </label>

      <button className="camera-open-btn" onClick={onCameraOpen}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="18" height="18">
          <path d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        카메라로 촬영
      </button>
    </div>
  )
}
