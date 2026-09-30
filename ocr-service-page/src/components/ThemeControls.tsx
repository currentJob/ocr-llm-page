import type React from 'react'

/** 강조 색상·테마 전환 — city-walk-planner 헤더와 같은 자리. 클릭은 index.html 의 공통 스크립트가 처리한다. */
const ACCENTS: [string, string, string, string][] = [
  ['green', '초록', '#245548', '#7cc4a6'], ['sky', '하늘', '#0b5f8a', '#7cc6ee'], ['yellow', '노랑', '#7a5600', '#f0c75e'],
  ['purple', '보라', '#5f3bab', '#b9a3f5'], ['orange', '주황', '#a8431b', '#f4a27c'], ['rose', '장미', '#a82d5c', '#f39bbd'],
]

export function ThemeControls() {
  return (
    <>
      <details className="cj-accent">
        <summary aria-label="강조 색상 선택" title="강조 색상"><span className="cj-dot" /></summary>
        <div className="cj-swatches" role="group" aria-label="강조 색상">
          {ACCENTS.map(([id, label, light, dark]) => (
            <button key={id} type="button" data-cj-accent={id} aria-label={label} style={{ '--l': light, '--d': dark } as React.CSSProperties} />
          ))}
        </div>
      </details>
      <button type="button" className="cj-theme-btn" data-cj-theme-toggle aria-label="밝은/어두운 테마 전환" title="테마 전환">
        <svg className="cj-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" /></svg>
        <svg className="cj-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" /></svg>
      </button>
    </>
  )
}
