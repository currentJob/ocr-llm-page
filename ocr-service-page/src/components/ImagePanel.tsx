import { useCallback, useEffect }            from 'react'
import type { ChangeEvent, MouseEvent, RefObject } from 'react'
import type { OcrItem, Phase, Roi }               from '../types'
import type { RoiDraw }                           from '../hooks/useRoi'
import { COLORS }                                 from '../types'

interface Props {
  imageUrl:   string
  imgRef:     RefObject<HTMLImageElement | null>
  canvasRef:  RefObject<HTMLCanvasElement | null>
  brightness: number
  contrast:   number
  phase:       Phase
  items:       OcrItem[]
  selected:    Set<number>
  natSize:     { w: number; h: number } | null
  roiFixed:    Roi | null
  roiDraw:     RoiDraw | null
  threshold:   number
  searchQuery: string
  rotation:    number
  onRotationChange:   (deg: number) => void
  onBrightnessChange: (v: number) => void
  onContrastChange:   (v: number) => void
  onResetPreprocess:  () => void
  onCanvasMouseDown:  (e: MouseEvent<HTMLCanvasElement>) => void
  onCanvasMouseMove:  (e: MouseEvent<HTMLCanvasElement>) => void
  onCommitRoi:        () => void
  onFileInput:    (e: ChangeEvent<HTMLInputElement>) => void
  onCameraOpen:   () => void
  onResetRoi:     () => void
  onRunOcr:       () => void
  onNextFile:     (() => void) | null
  fileQueueCount: number
}

export default function ImagePanel({
  imageUrl, imgRef, canvasRef, brightness, contrast,
  phase, items, selected, natSize, roiFixed, roiDraw, threshold, searchQuery,
  rotation, onRotationChange,
  onBrightnessChange, onContrastChange, onResetPreprocess,
  onCanvasMouseDown, onCanvasMouseMove, onCommitRoi,
  onFileInput, onCameraOpen, onResetRoi, onRunOcr,
  onNextFile, fileQueueCount,
}: Props) {

  // ── Canvas redraw ──────────────────────────────────────────────────────────
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img) return
    if (canvas.width !== img.clientWidth || canvas.height !== img.clientHeight) {
      canvas.width  = img.clientWidth
      canvas.height = img.clientHeight
    }
    const cw  = canvas.width, ch = canvas.height
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, cw, ch)

    if (phase === 'roi' || phase === 'running') {
      let rx = 0, ry = 0, rw = 0, rh = 0
      if (roiDraw) {
        rx = Math.min(roiDraw.sx, roiDraw.cx); ry = Math.min(roiDraw.sy, roiDraw.cy)
        rw = Math.abs(roiDraw.cx - roiDraw.sx); rh = Math.abs(roiDraw.cy - roiDraw.sy)
      } else if (roiFixed && natSize) {
        rx = roiFixed.x * (cw / natSize.w); ry = roiFixed.y * (ch / natSize.h)
        rw = roiFixed.w * (cw / natSize.w); rh = roiFixed.h * (ch / natSize.h)
      }
      if (rw > 0 && rh > 0) {
        ctx.fillStyle = 'rgba(0,0,0,0.5)'
        ctx.fillRect(0, 0, cw, ch)
        ctx.clearRect(rx, ry, rw, rh)
        ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2.5; ctx.setLineDash([])
        ctx.strokeRect(rx, ry, rw, rh)
        const hs = 8; ctx.fillStyle = '#60a5fa'
        ;[[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]].forEach(([cx2, cy2]) => {
          ctx.fillRect(cx2 - hs / 2, cy2 - hs / 2, hs, hs)
        })
      }
    }

    if (phase === 'done' && natSize) {
      const sx = cw / natSize.w, sy = ch / natSize.h
      items.forEach((item, i) => {
        if (!selected.has(i)) return
        const passes = (item.recScore * 100) >= threshold &&
          (!searchQuery || item.text.toLowerCase().includes(searchQuery.toLowerCase()))
        if (!passes) return
        const color = COLORS[i % COLORS.length]
        const pts   = item.box.map(([x, y]) => [x * sx, y * sy])
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1])
        pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
        ctx.closePath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([]); ctx.stroke()
        const label = `${i + 1}`, fs = Math.max(10, Math.round(13 * Math.min(sx, sy)))
        ctx.font = `bold ${fs}px sans-serif`
        const tw = ctx.measureText(label).width
        ctx.fillStyle = color; ctx.fillRect(pts[0][0] - 1, pts[0][1] - fs - 2, tw + 8, fs + 4)
        ctx.fillStyle = '#fff'; ctx.fillText(label, pts[0][0] + 3, pts[0][1] - 2)
      })
      if (roiFixed) {
        ctx.strokeStyle = 'rgba(96,165,250,0.45)'; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4])
        ctx.strokeRect(
          roiFixed.x * (cw / natSize.w), roiFixed.y * (ch / natSize.h),
          roiFixed.w * (cw / natSize.w), roiFixed.h * (ch / natSize.h),
        )
        ctx.setLineDash([])
      }
    }
  }, [phase, items, selected, natSize, roiFixed, roiDraw, threshold, searchQuery, canvasRef, imgRef])

  useEffect(() => {
    redraw()
    window.addEventListener('resize', redraw)
    return () => window.removeEventListener('resize', redraw)
  }, [redraw])

  const hasPreprocess = brightness !== 100 || contrast !== 100

  return (
    <div className="image-col">

      {/* 이미지 + 캔버스 */}
      <div className="image-wrap">
        <img
          ref={imgRef} src={imageUrl} alt="uploaded" onLoad={redraw}
          className="preview-img"
          style={hasPreprocess ? { filter: `brightness(${brightness}%) contrast(${contrast}%)` } : undefined}
        />
        <canvas
          ref={canvasRef}
          className={`overlay-canvas${phase === 'roi' ? ' roi-mode' : ''}`}
          onMouseDown={onCanvasMouseDown}
          onMouseMove={onCanvasMouseMove}
          onMouseUp={onCommitRoi}
          onMouseLeave={onCommitRoi}
        />
        {phase === 'running' && (
          <div className="img-overlay"><span className="spinner" /><span>인식 중...</span></div>
        )}
      </div>

      {/* 회전 */}
      <div className="tool-card">
        <div className="tool-header">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="14" height="14">
            <path d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>이미지 회전</span>
        </div>
        <div className="rot-controls">
          <button className="rot-btn" onClick={() => onRotationChange(rotation - 5)}>
            <svg viewBox="0 0 20 20" fill="currentColor" width="12" height="12"><path fillRule="evenodd" d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z" clipRule="evenodd"/></svg>5
          </button>
          <button className="rot-btn sm" onClick={() => onRotationChange(rotation - 1)}>
            <svg viewBox="0 0 20 20" fill="currentColor" width="10" height="10"><path fillRule="evenodd" d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z" clipRule="evenodd"/></svg>1
          </button>
          <div className="rot-angle">
            <span className="rot-value">{rotation > 0 ? `+${rotation}` : rotation}°</span>
          </div>
          <button className="rot-btn sm" onClick={() => onRotationChange(rotation + 1)}>
            1<svg viewBox="0 0 20 20" fill="currentColor" width="10" height="10"><path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd"/></svg>
          </button>
          <button className="rot-btn" onClick={() => onRotationChange(rotation + 5)}>
            5<svg viewBox="0 0 20 20" fill="currentColor" width="12" height="12"><path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd"/></svg>
          </button>
        </div>
        <div className="slider-row">
          <span className="slider-label">-180°</span>
          <input type="range" className="rot-slider" min={-180} max={180} step={1}
            value={rotation} onChange={e => onRotationChange(Number(e.target.value))} />
          <span className="slider-label">+180°</span>
        </div>
        {rotation !== 0 && (
          <button className="rot-reset" onClick={() => onRotationChange(0)}>원래대로</button>
        )}
      </div>

      {/* 전처리 */}
      <div className="tool-card">
        <div className="tool-header">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="14" height="14">
            <path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>이미지 전처리</span>
          {hasPreprocess && (
            <button className="tool-reset" onClick={onResetPreprocess}>초기화</button>
          )}
        </div>
        <div className="slider-row">
          <span className="slider-label" style={{ width: '2.5rem' }}>밝기</span>
          <input type="range" className="rot-slider" min={50} max={200} step={5}
            value={brightness} onChange={e => onBrightnessChange(Number(e.target.value))} />
          <span className="slider-val">{brightness}%</span>
        </div>
        <div className="slider-row">
          <span className="slider-label" style={{ width: '2.5rem' }}>대비</span>
          <input type="range" className="rot-slider" min={50} max={250} step={5}
            value={contrast} onChange={e => onContrastChange(Number(e.target.value))} />
          <span className="slider-val">{contrast}%</span>
        </div>
      </div>

      {/* 액션 버튼 */}
      <div className="image-actions">
        <label className="btn-ghost">
          <input type="file" accept="image/*" multiple onChange={onFileInput} hidden />
          다른 이미지
        </label>
        <button className="btn-ghost" onClick={onCameraOpen}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="13" height="13">
            <path d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          카메라
        </button>
        {(phase === 'done' || phase === 'error') && (
          <button className="btn-ghost" onClick={onResetRoi}>ROI 재설정</button>
        )}
        <div className="actions-spacer" />
        {onNextFile && (
          <button className="btn-ghost queue-next" onClick={onNextFile}>
            다음 ({fileQueueCount}개 남음) →
          </button>
        )}
        {phase === 'roi' && (
          <button className="btn-primary" onClick={onRunOcr}>
            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
              <path fillRule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm6.39-2.908a.75.75 0 01.766.027l3.5 2.25a.75.75 0 010 1.262l-3.5 2.25A.75.75 0 018 12.25v-4.5a.75.75 0 01.39-.658z" clipRule="evenodd"/>
            </svg>
            인식 시작
          </button>
        )}
        {(phase === 'done' || phase === 'error') && (
          <button className="btn-primary" onClick={onRunOcr}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" width="14" height="14">
              <path d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            다시 인식
          </button>
        )}
      </div>

      {phase === 'roi' && (
        <p className="roi-hint">
          {roiFixed
            ? '✓ 영역 선택됨 — 다시 드래그하면 변경됩니다'
            : '드래그로 인식할 영역(ROI)을 선택하세요 · 미선택 시 전체 이미지 인식'}
        </p>
      )}
    </div>
  )
}
