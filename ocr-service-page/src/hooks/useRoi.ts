import { useState, useCallback }  from 'react'
import type { MouseEvent, RefObject } from 'react'
import type { Roi, Phase } from '../types'

export type RoiDraw = { sx: number; sy: number; cx: number; cy: number }

export function useRoi(
  natSize:   { w: number; h: number } | null,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  phase:     Phase,
) {
  const [roiFixed, setRoiFixed] = useState<Roi | null>(null)
  const [roiDraw,  setRoiDraw]  = useState<RoiDraw | null>(null)

  const onCanvasMouseDown = useCallback((e: MouseEvent<HTMLCanvasElement>) => {
    if (phase !== 'roi') return
    const rect = canvasRef.current!.getBoundingClientRect()
    setRoiFixed(null)
    setRoiDraw({
      sx: e.clientX - rect.left, sy: e.clientY - rect.top,
      cx: e.clientX - rect.left, cy: e.clientY - rect.top,
    })
  }, [phase, canvasRef])

  const onCanvasMouseMove = useCallback((e: MouseEvent<HTMLCanvasElement>) => {
    if (!roiDraw) return
    const c    = canvasRef.current!
    const rect = c.getBoundingClientRect()
    setRoiDraw(prev => prev ? {
      ...prev,
      cx: Math.max(0, Math.min(e.clientX - rect.left, c.width)),
      cy: Math.max(0, Math.min(e.clientY - rect.top,  c.height)),
    } : null)
  }, [roiDraw, canvasRef])

  const commitRoi = useCallback(() => {
    if (!roiDraw || !natSize || !canvasRef.current) { setRoiDraw(null); return }
    const { sx, sy, cx, cy } = roiDraw
    const x = Math.min(sx, cx), y = Math.min(sy, cy)
    const w = Math.abs(cx - sx), h = Math.abs(cy - sy)
    setRoiDraw(null)
    if (w > 5 && h > 5) {
      const c = canvasRef.current
      setRoiFixed({
        x: Math.round(x * natSize.w / c.width),
        y: Math.round(y * natSize.h / c.height),
        w: Math.round(w * natSize.w / c.width),
        h: Math.round(h * natSize.h / c.height),
      })
    }
  }, [roiDraw, natSize, canvasRef])

  const clearRoi = useCallback(() => {
    setRoiFixed(null)
    setRoiDraw(null)
  }, [])

  return { roiFixed, roiDraw, onCanvasMouseDown, onCanvasMouseMove, commitRoi, clearRoi }
}
