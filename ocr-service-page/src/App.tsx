import { useCallback, useEffect, useRef, useState } from 'react'
import { KoreanOCR } from './ocr/pipeline'
import type { OcrItem, LoadProgress } from './ocr/pipeline'
import { loadLLM, summarize, isLLMLoaded } from './llm/engine'
import type { LLMLoadProgress } from './llm/engine'
import './App.css'

const COLORS = [
  '#3b82f6','#10b981','#f59e0b','#ef4444',
  '#8b5cf6','#06b6d4','#84cc16','#f97316',
]

type Phase = 'loading-model' | 'ready' | 'roi' | 'running' | 'done' | 'error'
interface Roi { x: number; y: number; w: number; h: number }
interface HistoryEntry { id: string; ts: number; thumb: string; filename: string; items: OcrItem[] }

const HISTORY_KEY = 'ocr-history'
const MAX_HISTORY = 8

function loadHistory(): HistoryEntry[] {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]') } catch { return [] }
}

export default function App() {
  const [phase, setPhase]       = useState<Phase>('loading-model')
  const [progress, setProgress] = useState<LoadProgress | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [natSize, setNatSize]   = useState<{ w: number; h: number } | null>(null)
  const [items, setItems]       = useState<OcrItem[]>([])
  const [error, setError]       = useState('')
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [roiFixed, setRoiFixed] = useState<Roi | null>(null)
  const [roiDraw, setRoiDraw]   = useState<{ sx:number;sy:number;cx:number;cy:number } | null>(null)
  const [rotation, setRotation] = useState(0)

  const [llmStatus, setLlmStatus]             = useState<'idle'|'loading-model'|'running'|'done'|'error'>('idle')
  const [llmLoadProgress, setLlmLoadProgress] = useState<LLMLoadProgress | null>(null)
  const [llmError, setLlmError]               = useState('')
  const [summary, setSummary]                 = useState('')

  // ── 신규 상태 ──────────────────────────────────────────────────────────────
  const [editingIdx, setEditingIdx]     = useState<number | null>(null)
  const [editingText, setEditingText]   = useState('')
  const [threshold, setThreshold]       = useState(0)
  const [searchQuery, setSearchQuery]   = useState('')
  const [brightness, setBrightness]     = useState(100)
  const [contrast, setContrast]         = useState(100)
  const [showCamera, setShowCamera]     = useState(false)
  const [history, setHistory]           = useState<HistoryEntry[]>(loadHistory)
  const [showHistory, setShowHistory]   = useState(false)
  const [copyFeedback, setCopyFeedback] = useState(false)
  const [fileQueue, setFileQueue]       = useState<File[]>([])
  const [currentFilename, setCurrentFilename] = useState('')

  const ocrRef         = useRef<KoreanOCR | null>(null)
  const imgRef         = useRef<HTMLImageElement>(null)
  const canvasRef      = useRef<HTMLCanvasElement>(null)
  const loadedImgRef   = useRef<HTMLImageElement | null>(null)
  const originalImgRef = useRef<HTMLImageElement | null>(null)
  const originalUrlRef = useRef<string | null>(null)
  const debounceRef    = useRef<ReturnType<typeof setTimeout> | null>(null)
  const rotReqRef      = useRef(0)
  const videoRef       = useRef<HTMLVideoElement>(null)
  const streamRef      = useRef<MediaStream | null>(null)
  const editInputRef   = useRef<HTMLInputElement>(null)

  useEffect(() => {
    KoreanOCR.create(p => setProgress(p))
      .then(ocr => { ocrRef.current = ocr; setPhase('ready') })
      .catch(e  => { setError(String(e));  setPhase('error') })
    return () => { streamRef.current?.getTracks().forEach(t => t.stop()) }
  }, [])

  // ── 필터된 아이템 ─────────────────────────────────────────────────────────
  const filteredWithIdx = items
    .map((item, origIdx) => ({ item, origIdx }))
    .filter(({ item }) =>
      (item.recScore * 100) >= threshold &&
      (!searchQuery || item.text.toLowerCase().includes(searchQuery.toLowerCase()))
    )

  // ── 캔버스 그리기 ─────────────────────────────────────────────────────────
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img) return
    if (canvas.width !== img.clientWidth || canvas.height !== img.clientHeight) {
      canvas.width  = img.clientWidth
      canvas.height = img.clientHeight
    }
    const cw = canvas.width, ch = canvas.height
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
        ;[[rx,ry],[rx+rw,ry],[rx+rw,ry+rh],[rx,ry+rh]].forEach(([cx2,cy2]) => {
          ctx.fillRect(cx2 - hs/2, cy2 - hs/2, hs, hs)
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
        const pts = item.box.map(([x, y]) => [x * sx, y * sy])
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
        ctx.strokeRect(roiFixed.x*(cw/natSize.w), roiFixed.y*(ch/natSize.h),
                       roiFixed.w*(cw/natSize.w), roiFixed.h*(ch/natSize.h))
        ctx.setLineDash([])
      }
    }
  }, [phase, items, selected, natSize, roiFixed, roiDraw, threshold, searchQuery])

  useEffect(() => {
    redraw()
    window.addEventListener('resize', redraw)
    return () => window.removeEventListener('resize', redraw)
  }, [redraw])

  // ── 회전 ─────────────────────────────────────────────────────────────────
  async function applyRotation(deg: number) {
    if (!originalImgRef.current) return
    const reqId = ++rotReqRef.current
    const orig  = originalImgRef.current
    if (deg === 0) {
      if (rotReqRef.current !== reqId) return
      loadedImgRef.current = orig
      setImageUrl(originalUrlRef.current!)
      setNatSize({ w: orig.naturalWidth, h: orig.naturalHeight })
      return
    }
    const rad = (deg * Math.PI) / 180
    const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad))
    const nw  = Math.round(orig.naturalWidth * cos + orig.naturalHeight * sin)
    const nh  = Math.round(orig.naturalWidth * sin + orig.naturalHeight * cos)
    const c   = document.createElement('canvas')
    c.width = nw; c.height = nh
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, nw, nh)
    ctx.translate(nw / 2, nh / 2); ctx.rotate(rad)
    ctx.drawImage(orig, -orig.naturalWidth / 2, -orig.naturalHeight / 2)
    const url = c.toDataURL('image/jpeg', 0.93)
    const img = await new Promise<HTMLImageElement>(r => { const o = new Image(); o.onload = () => r(o); o.src = url })
    if (rotReqRef.current !== reqId) return
    loadedImgRef.current = img
    setImageUrl(url); setNatSize({ w: nw, h: nh })
  }

  function handleRotationChange(deg: number) {
    const clamped = Math.max(-180, Math.min(180, deg))
    setRotation(clamped); setRoiFixed(null); setItems([]); setSelected(new Set())
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => applyRotation(clamped), 80)
  }

  // ── 전처리 이미지 ─────────────────────────────────────────────────────────
  async function getPreprocessedImg(): Promise<HTMLImageElement> {
    const src = loadedImgRef.current
    if (!src) throw new Error('No image')
    if (brightness === 100 && contrast === 100) return src
    const c = document.createElement('canvas')
    c.width = src.naturalWidth; c.height = src.naturalHeight
    const ctx = c.getContext('2d')!
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`
    ctx.drawImage(src, 0, 0)
    return new Promise(r => { const o = new Image(); o.onload = () => r(o); o.src = c.toDataURL('image/jpeg', 0.93) })
  }

  // ── 파일 처리 ─────────────────────────────────────────────────────────────
  function processFile(file: File) {
    if (!file.type.startsWith('image/') || !ocrRef.current) return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    const url = URL.createObjectURL(file)
    setItems([]); setSelected(new Set()); setRoiFixed(null); setRoiDraw(null)
    setPhase('roi'); setError(''); setRotation(0)
    setLlmStatus('idle'); setSummary(''); setLlmLoadProgress(null)
    setSearchQuery(''); setThreshold(0); setBrightness(100); setContrast(100)
    setCurrentFilename(file.name)
    const img = new Image()
    img.onload = () => {
      originalImgRef.current = img; originalUrlRef.current = url
      loadedImgRef.current = img
      setNatSize({ w: img.naturalWidth, h: img.naturalHeight }); setImageUrl(url)
    }
    img.src = url
  }

  function processFiles(files: File[]) {
    const imgs = files.filter(f => f.type.startsWith('image/'))
    if (!imgs.length) return
    setFileQueue(imgs.slice(1)); processFile(imgs[0])
  }

  // ── OCR 실행 ──────────────────────────────────────────────────────────────
  async function runOcr() {
    if (!ocrRef.current || !loadedImgRef.current || !natSize) return
    setPhase('running'); setItems([]); setSelected(new Set())
    const srcImg = await getPreprocessedImg()
    let ocrImg: HTMLImageElement = srcImg, offsetX = 0, offsetY = 0
    if (roiFixed) {
      const { x, y, w, h } = roiFixed
      const crop = document.createElement('canvas')
      crop.width = w; crop.height = h
      crop.getContext('2d')!.drawImage(srcImg, x, y, w, h, 0, 0, w, h)
      ocrImg = await new Promise<HTMLImageElement>(r => {
        const o = new Image(); o.onload = () => r(o); o.src = crop.toDataURL()
      })
      offsetX = x; offsetY = y
    }
    try {
      const result = await ocrRef.current.predict(ocrImg)
      const offsetted: OcrItem[] = result.map(item => ({
        ...item,
        box: item.box.map(([bx, by]) => [bx + offsetX, by + offsetY]) as [number, number][],
      }))
      setItems(offsetted); setSelected(new Set(offsetted.map((_, i) => i)))
      setPhase('done')
      await addToHistory(loadedImgRef.current!, offsetted)
    } catch (e) {
      setError(String(e)); setPhase('error')
    }
  }

  // ── 히스토리 ──────────────────────────────────────────────────────────────
  async function addToHistory(img: HTMLImageElement, resultItems: OcrItem[]) {
    const ratio = Math.min(120 / img.naturalWidth, 120 / img.naturalHeight, 1)
    const w = Math.round(img.naturalWidth * ratio), h = Math.round(img.naturalHeight * ratio)
    const c = document.createElement('canvas'); c.width = w; c.height = h
    c.getContext('2d')!.drawImage(img, 0, 0, w, h)
    const entry: HistoryEntry = { id: Date.now().toString(), ts: Date.now(), thumb: c.toDataURL('image/jpeg', 0.7), filename: currentFilename, items: resultItems }
    setHistory(prev => {
      const next = [entry, ...prev].slice(0, MAX_HISTORY)
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(next)) } catch {}
      return next
    })
  }

  function restoreHistory(entry: HistoryEntry) {
    setItems(entry.items); setSelected(new Set(entry.items.map((_, i) => i)))
    setPhase('done'); setSearchQuery(''); setThreshold(0)
    setLlmStatus('idle'); setSummary(''); setShowHistory(false)
  }

  function removeHistory(id: string) {
    setHistory(prev => {
      const next = prev.filter(e => e.id !== id)
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(next)) } catch {}
      return next
    })
  }

  // ── ROI 이벤트 ────────────────────────────────────────────────────────────
  function onCanvasMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (phase !== 'roi') return
    const rect = canvasRef.current!.getBoundingClientRect()
    setRoiFixed(null)
    setRoiDraw({ sx: e.clientX - rect.left, sy: e.clientY - rect.top, cx: e.clientX - rect.left, cy: e.clientY - rect.top })
  }
  function onCanvasMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!roiDraw) return
    const rect = canvasRef.current!.getBoundingClientRect()
    setRoiDraw(prev => prev ? { ...prev, cx: Math.max(0, Math.min(e.clientX - rect.left, canvasRef.current!.width)), cy: Math.max(0, Math.min(e.clientY - rect.top, canvasRef.current!.height)) } : null)
  }
  function commitRoi() {
    if (!roiDraw || !natSize || !canvasRef.current) { setRoiDraw(null); return }
    const { sx, sy, cx, cy } = roiDraw, x = Math.min(sx, cx), y = Math.min(sy, cy)
    const w = Math.abs(cx - sx), h = Math.abs(cy - sy)
    setRoiDraw(null)
    if (w > 5 && h > 5) {
      const c = canvasRef.current
      setRoiFixed({ x: Math.round(x * natSize.w / c.width), y: Math.round(y * natSize.h / c.height), w: Math.round(w * natSize.w / c.width), h: Math.round(h * natSize.h / c.height) })
    }
  }

  function onFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    processFiles(Array.from(e.target.files ?? [])); e.target.value = ''
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false); processFiles(Array.from(e.dataTransfer.files))
  }
  function toggleItem(i: number) {
    setSelected(prev => { const n = new Set(prev); n.has(i) ? n.delete(i) : n.add(i); return n })
  }
  const selectAll   = () => setSelected(new Set(items.map((_, i) => i)))
  const deselectAll = () => setSelected(new Set())
  function resetRoi() {
    setRoiFixed(null); setRoiDraw(null); setItems([]); setSelected(new Set()); setPhase('roi')
    setLlmStatus('idle'); setSummary(''); setLlmLoadProgress(null); setSearchQuery(''); setThreshold(0)
  }

  // ── 인라인 편집 ───────────────────────────────────────────────────────────
  function startEdit(i: number, text: string) {
    setEditingIdx(i); setEditingText(text)
    setTimeout(() => editInputRef.current?.focus(), 0)
  }
  function commitEdit() {
    if (editingIdx === null) return
    setItems(prev => prev.map((it, i) => i === editingIdx ? { ...it, text: editingText } : it))
    setEditingIdx(null)
  }

  // ── 내보내기 ─────────────────────────────────────────────────────────────
  function exportAs(fmt: 'txt' | 'json' | 'csv') {
    const exp = items.filter((_, i) => selected.has(i))
    let content = '', mime = 'text/plain'
    if (fmt === 'txt') {
      content = exp.map(it => it.text).join('\n')
    } else if (fmt === 'json') {
      content = JSON.stringify(exp.map(it => ({ text: it.text, score: Math.round(it.recScore * 100), box: it.box })), null, 2)
      mime = 'application/json'
    } else {
      content = 'text,score\n' + exp.map(it => `"${it.text.replace(/"/g, '""')}",${Math.round(it.recScore * 100)}`).join('\n')
      mime = 'text/csv'
    }
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob(['﻿' + content], { type: mime })),
      download: `ocr-result.${fmt}`,
    })
    a.click(); URL.revokeObjectURL(a.href)
  }

  function copySelected() {
    navigator.clipboard.writeText(items.filter((_, i) => selected.has(i)).map(it => it.text).join('\n'))
      .then(() => { setCopyFeedback(true); setTimeout(() => setCopyFeedback(false), 1800) })
  }

  // ── 카메라 ───────────────────────────────────────────────────────────────
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
      streamRef.current = stream; setShowCamera(true)
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = stream }, 50)
    } catch { alert('카메라를 사용할 수 없습니다.') }
  }
  function stopCamera() {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setShowCamera(false)
  }
  function captureCamera() {
    const v = videoRef.current; if (!v) return
    const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    c.toBlob(blob => {
      if (!blob) return
      stopCamera(); processFile(new File([blob], 'camera.jpg', { type: 'image/jpeg' }))
    }, 'image/jpeg', 0.92)
  }

  // ── AI 분석 ──────────────────────────────────────────────────────────────
  async function runAnalysis() {
    if (!items.length) return
    setLlmError(''); setSummary('')
    if (!isLLMLoaded()) {
      setLlmStatus('loading-model'); setLlmLoadProgress(null)
      try { await loadLLM(p => setLlmLoadProgress(p)) }
      catch (e) { setLlmStatus('error'); setLlmError(String(e)); return }
    }
    setLlmStatus('running')
    try {
      await summarize(items.map(it => it.text), token => setSummary(prev => prev + token))
      setLlmStatus('done')
    } catch (e) { setLlmStatus('error'); setLlmError(String(e)) }
  }

  const isLoading  = phase === 'loading-model'
  const showUpload = phase === 'ready'
  const showResult = imageUrl !== null && !isLoading
  const hasQueue   = fileQueue.length > 0

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <h1>Korean OCR</h1>
          <p>브라우저에서 직접 실행 — 서버 없이 한국어 텍스트 인식</p>
          {history.length > 0 && (
            <button className="history-toggle" onClick={() => setShowHistory(s => !s)}>
              <svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 000-1.5h-3.25V5z" clipRule="evenodd"/>
              </svg>
              히스토리 {history.length}
            </button>
          )}
        </div>
      </header>

      {/* ── 카메라 오버레이 ── */}
      {showCamera && (
        <div className="camera-overlay" onClick={e => { if (e.target === e.currentTarget) stopCamera() }}>
          <div className="camera-modal">
            <video ref={videoRef} autoPlay playsInline className="camera-video" />
            <div className="camera-actions">
              <button className="camera-btn cancel" onClick={stopCamera}>취소</button>
              <button className="camera-btn capture" onClick={captureCamera}>
                <span className="camera-shutter" />촬영
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── 히스토리 패널 ── */}
      {showHistory && history.length > 0 && (
        <div className="history-panel">
          <div className="history-header">
            <span>인식 히스토리</span>
            <button className="history-close" onClick={() => setShowHistory(false)}>✕</button>
          </div>
          <div className="history-grid">
            {history.map(entry => (
              <div key={entry.id} className="history-card" onClick={() => restoreHistory(entry)}>
                <img src={entry.thumb} alt="" className="history-thumb" />
                <div className="history-meta">
                  <span className="history-filename">{entry.filename || '무제'}</span>
                  <span className="history-row">
                    <span className="history-count">{entry.items.length}건</span>
                    <span className="history-time">{new Date(entry.ts).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}</span>
                  </span>
                </div>
                <button className="history-del" onClick={e => { e.stopPropagation(); removeHistory(entry.id) }}>✕</button>
              </div>
            ))}
          </div>
        </div>
      )}

      <main className="main">

        {isLoading && (
          <div className="center-card">
            <span className="spinner lg" />
            <p className="loading-title">모델 로딩 중</p>
            {progress && (<>
              <p className="loading-step">{progress.step}</p>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${(progress.done / progress.total) * 100}%` }} />
              </div>
              <p className="loading-count">{progress.done} / {progress.total}</p>
            </>)}
          </div>
        )}

        {phase === 'error' && !imageUrl && (
          <div className="center-card">
            <div className="error-icon">✕</div>
            <p className="loading-title">모델 로딩 실패</p>
            <p className="loading-step">{error}</p>
          </div>
        )}

        {showUpload && (
          <div className="upload-area">
            <label className={`dropzone${dragging ? ' dragging' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <input type="file" accept="image/*" multiple onChange={onFileInput} hidden />
              <div className="dropzone-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <p className="dropzone-text">클릭하거나 이미지를 드래그하세요</p>
              <p className="dropzone-sub">PNG · JPG · WEBP · BMP · 여러 파일 선택 가능</p>
            </label>
            <button className="camera-open-btn" onClick={startCamera}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="18" height="18">
                <path d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              카메라로 촬영
            </button>
          </div>
        )}

        {showResult && (<>
          <div className="result-layout">

            <div className="image-col">
              <div className="image-wrap">
                <img ref={imgRef} src={imageUrl!} alt="uploaded" onLoad={redraw}
                  className="preview-img"
                  style={(brightness !== 100 || contrast !== 100)
                    ? { filter: `brightness(${brightness}%) contrast(${contrast}%)` }
                    : undefined}
                />
                <canvas ref={canvasRef}
                  className={`overlay-canvas${phase === 'roi' ? ' roi-mode' : ''}`}
                  onMouseDown={onCanvasMouseDown} onMouseMove={onCanvasMouseMove}
                  onMouseUp={commitRoi} onMouseLeave={commitRoi} />
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
                  <button className="rot-btn" onClick={() => handleRotationChange(rotation - 5)}>
                    <svg viewBox="0 0 20 20" fill="currentColor" width="12" height="12"><path fillRule="evenodd" d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z" clipRule="evenodd"/></svg>5
                  </button>
                  <button className="rot-btn sm" onClick={() => handleRotationChange(rotation - 1)}>
                    <svg viewBox="0 0 20 20" fill="currentColor" width="10" height="10"><path fillRule="evenodd" d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z" clipRule="evenodd"/></svg>1
                  </button>
                  <div className="rot-angle"><span className="rot-value">{rotation > 0 ? `+${rotation}` : rotation}°</span></div>
                  <button className="rot-btn sm" onClick={() => handleRotationChange(rotation + 1)}>
                    1<svg viewBox="0 0 20 20" fill="currentColor" width="10" height="10"><path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd"/></svg>
                  </button>
                  <button className="rot-btn" onClick={() => handleRotationChange(rotation + 5)}>
                    5<svg viewBox="0 0 20 20" fill="currentColor" width="12" height="12"><path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd"/></svg>
                  </button>
                </div>
                <div className="slider-row">
                  <span className="slider-label">-180°</span>
                  <input type="range" className="rot-slider" min={-180} max={180} step={1}
                    value={rotation} onChange={e => handleRotationChange(Number(e.target.value))} />
                  <span className="slider-label">+180°</span>
                </div>
                {rotation !== 0 && (
                  <button className="rot-reset" onClick={() => handleRotationChange(0)}>원래대로</button>
                )}
              </div>

              {/* 전처리 */}
              <div className="tool-card">
                <div className="tool-header">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="14" height="14">
                    <path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>이미지 전처리</span>
                  {(brightness !== 100 || contrast !== 100) && (
                    <button className="tool-reset" onClick={() => { setBrightness(100); setContrast(100) }}>초기화</button>
                  )}
                </div>
                <div className="slider-row">
                  <span className="slider-label" style={{ width: '2.5rem' }}>밝기</span>
                  <input type="range" className="rot-slider" min={50} max={200} step={5}
                    value={brightness} onChange={e => setBrightness(Number(e.target.value))} />
                  <span className="slider-val">{brightness}%</span>
                </div>
                <div className="slider-row">
                  <span className="slider-label" style={{ width: '2.5rem' }}>대비</span>
                  <input type="range" className="rot-slider" min={50} max={250} step={5}
                    value={contrast} onChange={e => setContrast(Number(e.target.value))} />
                  <span className="slider-val">{contrast}%</span>
                </div>
              </div>

              {/* 액션 */}
              <div className="image-actions">
                <label className="btn-ghost">
                  <input type="file" accept="image/*" multiple onChange={onFileInput} hidden />다른 이미지
                </label>
                <button className="btn-ghost" onClick={startCamera}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="13" height="13">
                    <path d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  카메라
                </button>
                {(phase === 'done' || phase === 'error') && (
                  <button className="btn-ghost" onClick={resetRoi}>ROI 재설정</button>
                )}
                <div className="actions-spacer" />
                {hasQueue && phase === 'done' && (
                  <button className="btn-ghost queue-next" onClick={() => {
                    processFile(fileQueue[0]); setFileQueue(q => q.slice(1))
                  }}>
                    다음 ({fileQueue.length}개 남음) →
                  </button>
                )}
                {phase === 'roi' && (
                  <button className="btn-primary" onClick={runOcr}>
                    <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                      <path fillRule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm6.39-2.908a.75.75 0 01.766.027l3.5 2.25a.75.75 0 010 1.262l-3.5 2.25A.75.75 0 018 12.25v-4.5a.75.75 0 01.39-.658z" clipRule="evenodd"/>
                    </svg>
                    인식 시작
                  </button>
                )}
                {(phase === 'done' || phase === 'error') && (
                  <button className="btn-primary" onClick={runOcr}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" width="14" height="14">
                      <path d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    다시 인식
                  </button>
                )}
              </div>

              {phase === 'roi' && (
                <p className="roi-hint">
                  {roiFixed ? '✓ 영역 선택됨 — 다시 드래그하면 변경됩니다' : '드래그로 인식할 영역(ROI)을 선택하세요 · 미선택 시 전체 이미지 인식'}
                </p>
              )}
            </div>

            {/* 결과 패널 */}
            <div className="results-panel">
              {phase === 'running' && (
                <div className="panel-status"><span className="spinner" /><span>인식 중...</span></div>
              )}
              {phase === 'roi' && (
                <div className="panel-status muted">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="28" height="28" style={{opacity:.35}}>
                    <path d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>인식 대기 중</span>
                </div>
              )}
              {phase === 'error' && imageUrl && (
                <div className="panel-status error">
                  <span className="error-icon sm">✕</span><span>{error}</span>
                </div>
              )}
              {phase === 'done' && items.length === 0 && (
                <div className="panel-status muted"><span>텍스트를 찾을 수 없습니다</span></div>
              )}
              {phase === 'done' && items.length > 0 && (<>
                <div className="panel-header">
                  <span className="panel-title">검출된 텍스트</span>
                  <span className="badge">{filteredWithIdx.length}/{items.length}</span>
                  <div className="select-actions">
                    <button className="select-btn" onClick={selectAll}>전체</button>
                    <button className="select-btn" onClick={deselectAll}>해제</button>
                  </div>
                </div>

                {/* 검색 */}
                <div className="search-row">
                  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" width="13" height="13" style={{flexShrink:0,color:'var(--text-3)'}}>
                    <path d="M19 19l-4-4m0-7A7 7 0 111 8a7 7 0 0114 0z" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <input className="search-input" placeholder="텍스트 검색..."
                    value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
                  {searchQuery && <button className="search-clear" onClick={() => setSearchQuery('')}>✕</button>}
                </div>

                {/* 신뢰도 임계값 */}
                <div className="threshold-row">
                  <span className="threshold-label">최소 신뢰도</span>
                  <input type="range" className="rot-slider" min={0} max={95} step={5}
                    value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
                  <span className="threshold-val">{threshold}%</span>
                </div>

                {/* 결과 목록 */}
                <ul className="result-list">
                  {filteredWithIdx.map(({ item, origIdx }) => (
                    <li key={origIdx}
                      className={`result-item${selected.has(origIdx) ? ' selected' : ''}`}
                      style={{ '--accent': COLORS[origIdx % COLORS.length], '--delay': `${origIdx * 20}ms` } as React.CSSProperties}
                      onClick={() => { if (editingIdx !== origIdx) toggleItem(origIdx) }}>
                      <span className="item-check">{selected.has(origIdx) ? '✓' : ''}</span>
                      <span className="item-index" style={{ background: COLORS[origIdx % COLORS.length] }}>{origIdx + 1}</span>
                      {editingIdx === origIdx ? (
                        <input ref={editInputRef} className="item-edit-input" value={editingText}
                          onChange={e => setEditingText(e.target.value)}
                          onBlur={commitEdit}
                          onKeyDown={e => { if (e.key === 'Enter') commitEdit(); if (e.key === 'Escape') setEditingIdx(null); e.stopPropagation() }}
                          onClick={e => e.stopPropagation()} />
                      ) : (
                        <span className="item-text" onDoubleClick={e => { e.stopPropagation(); startEdit(origIdx, item.text) }}>
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

                {/* 복사 + 내보내기 */}
                <button className={`copy-btn${copyFeedback ? ' copied' : ''}`} onClick={copySelected}>
                  {copyFeedback
                    ? <><svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13"><path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd"/></svg>복사됨</>
                    : <><svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13"><path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z"/><path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.378 6H4.5z"/></svg>선택 텍스트 복사</>
                  }
                </button>

                <div className="export-row">
                  <span className="export-label">내보내기</span>
                  <button className="export-btn" onClick={() => exportAs('txt')}>TXT</button>
                  <button className="export-btn" onClick={() => exportAs('json')}>JSON</button>
                  <button className="export-btn" onClick={() => exportAs('csv')}>CSV</button>
                </div>

                <button className="analyze-btn" onClick={runAnalysis}
                  disabled={llmStatus === 'loading-model' || llmStatus === 'running'}>
                  {(llmStatus === 'loading-model' || llmStatus === 'running')
                    ? <><span className="spinner sm" />{llmStatus === 'loading-model' ? '모델 로딩 중...' : 'AI 분석 중...'}</>
                    : <><svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path d="M10 1a6 6 0 00-3.815 10.631C7.237 12.5 8 13.443 8 14.456v.644a.75.75 0 00.572.729 6.016 6.016 0 002.856 0A.75.75 0 0012 15.1v-.644c0-1.013.762-1.957 1.815-2.825A6 6 0 0010 1zM8.863 17.414a.75.75 0 00-.226 1.483 9.066 9.066 0 002.726 0 .75.75 0 00-.226-1.483 7.553 7.553 0 01-2.274 0z"/></svg>{llmStatus === 'done' ? 'AI 재분석' : 'AI 분석'}</>
                  }
                </button>
              </>)}
            </div>
          </div>

          {/* AI 분석 결과 */}
          {llmStatus !== 'idle' && (
            <div className="summary-section">
              <div className="summary-header">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="15" height="15">
                  <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span>AI 분석 결과</span>
                {llmStatus === 'running'      && <span className="spinner sm" />}
                {llmStatus === 'done'         && <span className="summary-badge">완료</span>}
                {llmStatus === 'loading-model'&& <span className="summary-tag">모델 로딩 중...</span>}
                {llmStatus === 'error'        && <span className="summary-badge error">오류</span>}
              </div>
              {llmStatus === 'loading-model' && (
                <div className="summary-model-loading">
                  <p className="model-load-msg">
                    {llmLoadProgress ? `다운로드 중: ${llmLoadProgress.file.split('/').pop() ?? ''}` : 'Qwen2.5-0.5B ONNX 모델 로딩 중... (첫 실행 시 ~350MB 다운로드)'}
                  </p>
                  {llmLoadProgress && (<>
                    <div className="model-load-bar"><div className="model-load-fill" style={{ width: `${llmLoadProgress.progress}%` }} /></div>
                    <p className="model-load-pct">{llmLoadProgress.progress.toFixed(0)}%</p>
                  </>)}
                </div>
              )}
              {llmStatus === 'error' && <div className="summary-error">{llmError}</div>}
              {(llmStatus === 'running' || llmStatus === 'done') && summary && (
                <pre className="summary-text">{summary}{llmStatus === 'running' ? <span className="cursor-blink">▋</span> : null}</pre>
              )}
            </div>
          )}
        </>)}
      </main>
    </div>
  )
}
