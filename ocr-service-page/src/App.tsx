import { useEffect, useRef, useState } from 'react'
import type { OcrItem, Phase, OcrModelType, LoadProgress, HistoryEntry } from './types'
import { useOcrModel }    from './hooks/useOcrModel'
import { useImage }       from './hooks/useImage'
import { useRoi }         from './hooks/useRoi'
import { useHistory }     from './hooks/useHistory'
import { useCamera }      from './hooks/useCamera'
import { useLlm }         from './hooks/useLlm'
import Header             from './components/Header'
import CameraOverlay      from './components/CameraOverlay'
import HistoryPanel       from './components/HistoryPanel'
import UploadArea         from './components/UploadArea'
import ImagePanel         from './components/ImagePanel'
import ResultPanel        from './components/ResultPanel'
import type { FilteredItem } from './components/ResultPanel'
import SummarySection     from './components/SummarySection'
import './App.css'

export default function App() {
  // ── 핵심 상태 ──────────────────────────────────────────────────────────────
  const [phase,    setPhase]    = useState<Phase>('loading-model')
  const [error,    setError]    = useState('')
  const [items,    setItems]    = useState<OcrItem[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [modelType, setModelType] = useState<OcrModelType>('ppocr')
  const [progress,  setProgress]  = useState<LoadProgress | null>(null)
  const [currentFilename, setCurrentFilename] = useState('')
  const [threshold,   setThreshold]   = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [fileQueue,   setFileQueue]   = useState<File[]>([])

  // ── DOM refs (캔버스/이미지 — useRoi·ImagePanel 공유) ─────────────────────
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef<HTMLImageElement>(null)

  // ── 커스텀 훅 ─────────────────────────────────────────────────────────────
  const { ocrRef, loadModel }        = useOcrModel()
  const image                        = useImage()
  const roi                          = useRoi(image.natSize, canvasRef, phase)
  const hist                         = useHistory()
  const llm                          = useLlm()
  // processFile 은 정의 후 useCamera 에 주입 (onCaptureRef 패턴으로 stale-closure 방지)
  const camera                       = useCamera(processFile)

  // ── 모델 로드 ─────────────────────────────────────────────────────────────

  useEffect(() => {
    loadOcrModel('ppocr')
    return camera.stopOnUnmount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function loadOcrModel(type: OcrModelType, hadImage = false) {
    setPhase('loading-model')
    setProgress(null)
    try {
      await loadModel(type, p => setProgress(p))
      setPhase(hadImage ? 'roi' : 'ready')
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }

  async function switchModel(type: OcrModelType) {
    if (type === modelType || phase === 'loading-model') return
    setModelType(type)
    resetResults()
    await loadOcrModel(type, image.imageUrl !== null)
  }

  // ── OCR 생명주기 ──────────────────────────────────────────────────────────

  function resetResults() {
    setItems([])
    setSelected(new Set())
    roi.clearRoi()
  }

  function resetRoi() {
    resetResults()
    setPhase('roi')
    setThreshold(0)
    setSearchQuery('')
    llm.reset()
  }

  function processFile(file: File) {
    if (!file.type.startsWith('image/') || !ocrRef.current) return
    const url = URL.createObjectURL(file)
    resetResults()
    setPhase('roi')
    setError('')
    setThreshold(0)
    setSearchQuery('')
    setCurrentFilename(file.name)
    image.resetTransforms()
    llm.reset()
    const img = new Image()
    img.onload = () => image.setFile(url, img)
    img.src = url
  }

  function processFiles(files: File[]) {
    const imgs = files.filter(f => f.type.startsWith('image/'))
    if (!imgs.length) return
    setFileQueue(imgs.slice(1))
    processFile(imgs[0])
  }

  async function runOcr() {
    if (!ocrRef.current || !image.loadedImgRef.current || !image.natSize) return
    setPhase('running')
    setItems([])
    setSelected(new Set())

    const srcImg = await image.getPreprocessedImg()
    let ocrImg = srcImg, offsetX = 0, offsetY = 0

    if (roi.roiFixed) {
      const { x, y, w, h } = roi.roiFixed
      const c = document.createElement('canvas')
      c.width = w; c.height = h
      c.getContext('2d')!.drawImage(srcImg, x, y, w, h, 0, 0, w, h)
      ocrImg = await new Promise<HTMLImageElement>(r => {
        const o = new Image(); o.onload = () => r(o); o.src = c.toDataURL()
      })
      offsetX = x; offsetY = y
    }

    try {
      const result    = await ocrRef.current.predict(ocrImg)
      const offsetted = result.map(it => ({
        ...it,
        box: it.box.map(([bx, by]) => [bx + offsetX, by + offsetY]) as [number, number][],
      }))
      setItems(offsetted)
      setSelected(new Set(offsetted.map((_, i) => i)))
      setPhase('done')
      await hist.addToHistory(image.loadedImgRef.current!, offsetted, currentFilename)
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }

  // ── 히스토리 복원 ─────────────────────────────────────────────────────────

  async function restoreHistory(entry: HistoryEntry) {
    setItems(entry.items)
    setSelected(new Set(entry.items.map((_, i) => i)))
    setPhase('done')
    setThreshold(0)
    setSearchQuery('')
    setCurrentFilename(entry.filename)
    image.resetTransforms()
    llm.reset()
    hist.setShowHistory(false)
    // IndexedDB의 고화질 프리뷰 우선, 없으면 썸네일 폴백
    const url = (await hist.getPreview(entry.id)) ?? entry.thumb
    const img = new Image()
    img.onload = () => image.setFile(
      url, img,
      entry.natW && entry.natH ? { w: entry.natW, h: entry.natH } : undefined,
    )
    img.src = url
  }

  // ── 파생 값 ───────────────────────────────────────────────────────────────

  const filteredWithIdx: FilteredItem[] = items
    .map((item, origIdx) => ({ item, origIdx }))
    .filter(({ item }) =>
      (item.recScore * 100) >= threshold &&
      (!searchQuery || item.text.toLowerCase().includes(searchQuery.toLowerCase()))
    )

  const showResult = image.imageUrl !== null && phase !== 'loading-model'
  const hasQueue   = fileQueue.length > 0

  // ── 렌더 ──────────────────────────────────────────────────────────────────

  return (
    <div className="app">
      <Header
        modelType={modelType}
        phase={phase}
        history={hist.history}
        onSwitchModel={switchModel}
        onToggleHistory={() => hist.setShowHistory(s => !s)}
      />

      {camera.showCamera && (
        <CameraOverlay
          videoRef={camera.videoRef}
          onStop={camera.stopCamera}
          onCapture={camera.captureCamera}
        />
      )}

      {hist.showHistory && hist.history.length > 0 && (
        <HistoryPanel
          history={hist.history}
          onRestore={restoreHistory}
          onRemove={hist.removeHistory}
          onClose={() => hist.setShowHistory(false)}
        />
      )}

      <main className="main">

        {/* 모델 로딩 */}
        {phase === 'loading-model' && (
          <div className="center-card">
            <span className="spinner lg" />
            <p className="loading-title">'PP-OCR' 모델 로딩 중</p>
            {progress && (
              <>
                <p className="loading-step">{progress.step}</p>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${(progress.done / progress.total) * 100}%` }} />
                </div>
                <p className="loading-count">{progress.done} / {progress.total}</p>
              </>
            )}
          </div>
        )}

        {/* 모델 로딩 실패 */}
        {phase === 'error' && !image.imageUrl && (
          <div className="center-card">
            <div className="error-icon">✕</div>
            <p className="loading-title">모델 로딩 실패</p>
            <p className="loading-step" style={{ whiteSpace: 'pre-line' }}>{error}</p>
          </div>
        )}

        {/* 초기 업로드 */}
        {phase === 'ready' && (
          <UploadArea
            onFiles={processFiles}
            onCameraOpen={camera.startCamera}
          />
        )}

        {/* 인식 작업 뷰 */}
        {showResult && (
          <>
            <div className="result-layout">
              <ImagePanel
                imageUrl={image.imageUrl!}
                imgRef={imgRef}
                canvasRef={canvasRef}
                brightness={image.brightness}
                contrast={image.contrast}
                phase={phase}
                items={items}
                selected={selected}
                natSize={image.natSize}
                roiFixed={roi.roiFixed}
                roiDraw={roi.roiDraw}
                threshold={threshold}
                searchQuery={searchQuery}
                rotation={image.rotation}
                onRotationChange={image.handleRotationChange}
                onBrightnessChange={image.setBrightness}
                onContrastChange={image.setContrast}
                onResetPreprocess={() => { image.setBrightness(100); image.setContrast(100) }}
                onCanvasMouseDown={roi.onCanvasMouseDown}
                onCanvasMouseMove={roi.onCanvasMouseMove}
                onCommitRoi={roi.commitRoi}
                onFileInput={e => { processFiles(Array.from(e.target.files ?? [])); e.target.value = '' }}
                onCameraOpen={camera.startCamera}
                onResetRoi={resetRoi}
                onRunOcr={runOcr}
                onNextFile={hasQueue && phase === 'done'
                  ? () => { processFile(fileQueue[0]); setFileQueue(q => q.slice(1)) }
                  : null}
                fileQueueCount={fileQueue.length}
              />

              <ResultPanel
                phase={phase}
                error={error}
                items={items}
                selected={selected}
                filteredWithIdx={filteredWithIdx}
                threshold={threshold}
                searchQuery={searchQuery}
                llmStatus={llm.llmStatus}
                onToggleItem={i => setSelected(prev => {
                  const n = new Set(prev); n.has(i) ? n.delete(i) : n.add(i); return n
                })}
                onSelectAll={() => setSelected(new Set(items.map((_, i) => i)))}
                onDeselectAll={() => setSelected(new Set())}
                onEditItem={(idx, text) =>
                  setItems(prev => prev.map((it, i) => i === idx ? { ...it, text } : it))
                }
                onThresholdChange={setThreshold}
                onSearchChange={setSearchQuery}
                onRunAnalysis={() =>
                  llm.runAnalysis(items.filter((_, i) => selected.has(i)).map(it => it.text))
                }
              />
            </div>

            {llm.llmStatus !== 'idle' && (
              <SummarySection
                status={llm.llmStatus}
                progress={llm.llmProgress}
                error={llm.llmError}
                summary={llm.summary}
              />
            )}
          </>
        )}

      </main>
    </div>
  )
}
