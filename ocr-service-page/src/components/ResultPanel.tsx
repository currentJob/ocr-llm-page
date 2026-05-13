import { useState, useRef }    from 'react'
import type { CSSProperties }  from 'react'
import type { OcrItem, Phase, LlmStatus } from '../types'
import { COLORS }              from '../types'

export interface FilteredItem { item: OcrItem; origIdx: number }

interface Props {
  phase:           Phase
  error:           string
  items:           OcrItem[]
  selected:        Set<number>
  filteredWithIdx: FilteredItem[]
  threshold:       number
  searchQuery:     string
  llmStatus:       LlmStatus
  onToggleItem:      (i: number) => void
  onSelectAll:       () => void
  onDeselectAll:     () => void
  onEditItem:        (idx: number, text: string) => void
  onThresholdChange: (v: number) => void
  onSearchChange:    (v: string) => void
  onRunAnalysis:     () => void
}

export default function ResultPanel({
  phase, error, items, selected, filteredWithIdx,
  threshold, searchQuery, llmStatus,
  onToggleItem, onSelectAll, onDeselectAll, onEditItem,
  onThresholdChange, onSearchChange, onRunAnalysis,
}: Props) {
  const [editingIdx,   setEditingIdx]   = useState<number | null>(null)
  const [editingText,  setEditingText]  = useState('')
  const [copyFeedback, setCopyFeedback] = useState(false)
  const editInputRef = useRef<HTMLInputElement>(null)

  function startEdit(i: number, text: string) {
    setEditingIdx(i); setEditingText(text)
    setTimeout(() => editInputRef.current?.focus(), 0)
  }

  function commitEdit() {
    if (editingIdx === null) return
    onEditItem(editingIdx, editingText)
    setEditingIdx(null)
  }

  function copySelected() {
    const text = items.filter((_, i) => selected.has(i)).map(it => it.text).join('\n')
    navigator.clipboard.writeText(text)
      .then(() => { setCopyFeedback(true); setTimeout(() => setCopyFeedback(false), 1800) })
  }

  function exportAs(fmt: 'txt' | 'json' | 'csv') {
    const exp = items.filter((_, i) => selected.has(i))
    let content = '', mime = 'text/plain'
    if (fmt === 'txt') {
      content = exp.map(it => it.text).join('\n')
    } else if (fmt === 'json') {
      content = JSON.stringify(exp.map(it => ({
        text: it.text, score: Math.round(it.recScore * 100), box: it.box,
      })), null, 2)
      mime = 'application/json'
    } else {
      content = 'text,score\n' + exp.map(it =>
        `"${it.text.replace(/"/g, '""')}",${Math.round(it.recScore * 100)}`
      ).join('\n')
      mime = 'text/csv'
    }
    const a = Object.assign(document.createElement('a'), {
      href:     URL.createObjectURL(new Blob(['﻿' + content], { type: mime })),
      download: `ocr-result.${fmt}`,
    })
    a.click(); URL.revokeObjectURL(a.href)
  }

  // ── 상태별 패널 ───────────────────────────────────────────────────────────

  if (phase === 'running') return (
    <div className="results-panel">
      <div className="panel-status"><span className="spinner" /><span>인식 중...</span></div>
    </div>
  )

  if (phase === 'roi') return (
    <div className="results-panel">
      <div className="panel-status muted">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="28" height="28" style={{opacity:.35}}>
          <path d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <span>인식 대기 중</span>
      </div>
    </div>
  )

  if (phase === 'error') return (
    <div className="results-panel">
      <div className="panel-status error">
        <span className="error-icon sm">✕</span><span>{error}</span>
      </div>
    </div>
  )

  if (phase === 'done' && items.length === 0) return (
    <div className="results-panel">
      <div className="panel-status muted"><span>텍스트를 찾을 수 없습니다</span></div>
    </div>
  )

  // ── 결과 목록 ─────────────────────────────────────────────────────────────

  const isAnalyzing = llmStatus === 'loading-model' || llmStatus === 'running'

  return (
    <div className="results-panel">
      <div className="panel-header">
        <span className="panel-title">검출된 텍스트</span>
        <span className="badge">{filteredWithIdx.length}/{items.length}</span>
        <div className="select-actions">
          <button className="select-btn" onClick={onSelectAll}>전체</button>
          <button className="select-btn" onClick={onDeselectAll}>해제</button>
        </div>
      </div>

      <div className="search-row">
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" width="13" height="13" style={{flexShrink:0,color:'var(--text-3)'}}>
          <path d="M19 19l-4-4m0-7A7 7 0 111 8a7 7 0 0114 0z" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <input className="search-input" placeholder="텍스트 검색..."
          value={searchQuery} onChange={e => onSearchChange(e.target.value)} />
        {searchQuery && <button className="search-clear" onClick={() => onSearchChange('')}>✕</button>}
      </div>

      <div className="threshold-row">
        <span className="threshold-label">최소 신뢰도</span>
        <input type="range" className="rot-slider" min={0} max={95} step={5}
          value={threshold} onChange={e => onThresholdChange(Number(e.target.value))} />
        <span className="threshold-val">{threshold}%</span>
      </div>

      <ul className="result-list">
        {filteredWithIdx.map(({ item, origIdx }) => (
          <li
            key={origIdx}
            className={`result-item${selected.has(origIdx) ? ' selected' : ''}`}
            style={{ '--accent': COLORS[origIdx % COLORS.length], '--delay': `${origIdx * 20}ms` } as CSSProperties}
            onClick={() => { if (editingIdx !== origIdx) onToggleItem(origIdx) }}
          >
            <span className="item-check">{selected.has(origIdx) ? '✓' : ''}</span>
            <span className="item-index" style={{ background: COLORS[origIdx % COLORS.length] }}>
              {origIdx + 1}
            </span>
            {editingIdx === origIdx ? (
              <input
                ref={editInputRef}
                className="item-edit-input"
                value={editingText}
                onChange={e => setEditingText(e.target.value)}
                onBlur={commitEdit}
                onKeyDown={e => {
                  if (e.key === 'Enter')  commitEdit()
                  if (e.key === 'Escape') setEditingIdx(null)
                  e.stopPropagation()
                }}
                onClick={e => e.stopPropagation()}
              />
            ) : (
              <span
                className="item-text"
                onDoubleClick={e => { e.stopPropagation(); startEdit(origIdx, item.text) }}
              >
                {item.text}
              </span>
            )}
            <span className="item-score">{(item.recScore * 100).toFixed(0)}%</span>
            <button className="item-copy" title="복사"
              onClick={e => { e.stopPropagation(); navigator.clipboard.writeText(item.text) }}>
              <svg viewBox="0 0 20 20" fill="currentColor" width="11" height="11">
                <path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z"/>
                <path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.378 6H4.5z"/>
              </svg>
            </button>
          </li>
        ))}
      </ul>

      <button className={`copy-btn${copyFeedback ? ' copied' : ''}`} onClick={copySelected}>
        {copyFeedback
          ? <><svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13"><path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd"/></svg>복사됨</>
          : <><svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13"><path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z"/><path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.378 6H4.5z"/></svg>선택 텍스트 복사</>
        }
      </button>

      <div className="export-row">
        <span className="export-label">내보내기</span>
        {(['txt', 'json', 'csv'] as const).map(fmt => (
          <button key={fmt} className="export-btn" onClick={() => exportAs(fmt)}>
            {fmt.toUpperCase()}
          </button>
        ))}
      </div>

      <button className="analyze-btn" onClick={onRunAnalysis} disabled={isAnalyzing}>
        {isAnalyzing
          ? <><span className="spinner sm" />{llmStatus === 'loading-model' ? '모델 로딩 중...' : 'AI 분석 중...'}</>
          : <>
              <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                <path d="M10 1a6 6 0 00-3.815 10.631C7.237 12.5 8 13.443 8 14.456v.644a.75.75 0 00.572.729 6.016 6.016 0 002.856 0A.75.75 0 0012 15.1v-.644c0-1.013.762-1.957 1.815-2.825A6 6 0 0010 1zM8.863 17.414a.75.75 0 00-.226 1.483 9.066 9.066 0 002.726 0 .75.75 0 00-.226-1.483 7.553 7.553 0 01-2.274 0z"/>
              </svg>
              {llmStatus === 'done' ? 'AI 재분석' : 'AI 분석'}
            </>
        }
      </button>
    </div>
  )
}
