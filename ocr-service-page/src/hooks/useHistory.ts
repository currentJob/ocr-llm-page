import { useState, useCallback } from 'react'
import type { HistoryEntry, OcrItem } from '../types'

const LS_KEY   = 'ocr-history'
const MAX_SIZE = 8
const DB_NAME  = 'ocr-history-db'
const DB_VER   = 1
const STORE    = 'previews'

function lsLoad(): HistoryEntry[] {
  try { return JSON.parse(localStorage.getItem(LS_KEY) ?? '[]') } catch { return [] }
}
function lsSave(entries: HistoryEntry[]): void {
  try { localStorage.setItem(LS_KEY, JSON.stringify(entries)) } catch {}
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VER)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror   = () => reject(req.error)
  })
}

async function idbPut(id: string, dataUrl: string) {
  const db = await openDb()
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite')
    tx.objectStore(STORE).put(dataUrl, id)
    tx.oncomplete = () => resolve()
    tx.onerror    = () => reject(tx.error)
  })
}

async function idbGet(id: string): Promise<string | undefined> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).get(id)
    req.onsuccess = () => resolve(req.result as string | undefined)
    req.onerror   = () => reject(req.error)
  })
}

async function idbDelete(id: string) {
  const db = await openDb()
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite')
    tx.objectStore(STORE).delete(id)
    tx.oncomplete = () => resolve()
    tx.onerror    = () => reject(tx.error)
  })
}

function makeCanvas(img: HTMLImageElement, maxPx: number, quality: number): string {
  const ratio = Math.min(maxPx / img.naturalWidth, maxPx / img.naturalHeight, 1)
  const w = Math.round(img.naturalWidth  * ratio)
  const h = Math.round(img.naturalHeight * ratio)
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  c.getContext('2d')!.drawImage(img, 0, 0, w, h)
  return c.toDataURL('image/jpeg', quality)
}

export function useHistory() {
  const [history,     setHistory]     = useState<HistoryEntry[]>(lsLoad)
  const [showHistory, setShowHistory] = useState(false)

  const addToHistory = useCallback(async (
    img:      HTMLImageElement,
    items:    OcrItem[],
    filename: string,
  ) => {
    const entry: HistoryEntry = {
      id:    Date.now().toString(),
      ts:    Date.now(),
      thumb: makeCanvas(img, 120, 0.7),
      natW:  img.naturalWidth,
      natH:  img.naturalHeight,
      filename,
      items,
    }
    setHistory(prev => {
      const next = [entry, ...prev].slice(0, MAX_SIZE)
      lsSave(next)
      return next
    })
    // 고화질 프리뷰는 IndexedDB에 별도 저장 (최대 2400px)
    idbPut(entry.id, makeCanvas(img, 2400, 0.92)).catch(() => {})
  }, [])

  const removeHistory = useCallback((id: string) => {
    setHistory(prev => {
      const next = prev.filter(e => e.id !== id)
      lsSave(next)
      return next
    })
    idbDelete(id).catch(() => {})
  }, [])

  const getPreview = useCallback(async (id: string): Promise<string | undefined> => {
    try { return await idbGet(id) } catch { return undefined }
  }, [])

  return { history, showHistory, setShowHistory, addToHistory, removeHistory, getPreview }
}
