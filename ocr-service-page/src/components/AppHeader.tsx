import type { ChangeEvent } from 'react'
import Icon from './Icon'
import { ThemeControls } from './ThemeControls'

interface Props {
  historyCount:    number
  historyOpen:     boolean
  busy:            boolean
  onFiles:         (files: File[]) => void
  onCameraOpen:    () => void
  onToggleHistory: () => void
}

/** 상단 헤더: 로고·워드마크, 새 이미지·카메라·기록, 소스 링크. 모바일에서는 아이콘만 보인다. */
export default function AppHeader({ historyCount, historyOpen, busy, onFiles, onCameraOpen, onToggleHistory }: Props) {
  function pick(e: ChangeEvent<HTMLInputElement>) {
    onFiles(Array.from(e.target.files ?? []))
    e.target.value = ''
  }
  return (
    <header className="cj-header">
      <a className="cj-brand" href="./">
        <span className="cj-brand-mark"><Icon name="logo" className="" /></span><span>Korean OCR</span>
      </a>
      <nav className="cj-nav" aria-label="작업">
        <label aria-label="새 이미지" className={busy ? 'is-disabled' : undefined}>
          <input type="file" accept="image/*" multiple hidden disabled={busy} onChange={pick} />
          <Icon name="image" /><span className="cj-label">새 이미지</span>
        </label>
        <button type="button" onClick={onCameraOpen} disabled={busy} aria-label="카메라로 촬영">
          <Icon name="camera" /><span className="cj-label">카메라</span>
        </button>
        <button type="button" onClick={onToggleHistory} disabled={!historyCount} aria-pressed={historyOpen} aria-label={`기록 ${historyCount}개`}>
          <Icon name="history" /><span className="cj-label">기록{historyCount ? ` ${historyCount}` : ''}</span>
        </button>
      </nav>
      <div className="cj-header-actions">
        <ThemeControls />
        <a className="cj-pill" href="https://github.com/currentJob/ocr-llm-page" target="_blank" rel="noreferrer" aria-label="소스 코드 (GitHub)">
          <Icon name="github" /><span className="cj-label">소스</span>
        </a>
      </div>
    </header>
  )
}
