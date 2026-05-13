import type { HistoryEntry } from '../types'

interface Props {
  history:   HistoryEntry[]
  onRestore: (entry: HistoryEntry) => void
  onRemove:  (id: string) => void
  onClose:   () => void
}

export default function HistoryPanel({ history, onRestore, onRemove, onClose }: Props) {
  return (
    <div className="history-panel">
      <div className="history-header">
        <span>인식 히스토리</span>
        <button className="history-close" onClick={onClose}>✕</button>
      </div>
      <div className="history-grid">
        {history.map(entry => (
          <div key={entry.id} className="history-card" onClick={() => onRestore(entry)}>
            <img src={entry.thumb} alt="" className="history-thumb" />
            <div className="history-meta">
              <span className="history-filename">{entry.filename || '무제'}</span>
              <span className="history-row">
                <span className="history-count">{entry.items.length}건</span>
                <span className="history-time">
                  {new Date(entry.ts).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                </span>
              </span>
            </div>
            <button
              className="history-del"
              onClick={e => { e.stopPropagation(); onRemove(entry.id) }}
            >✕</button>
          </div>
        ))}
      </div>
    </div>
  )
}
