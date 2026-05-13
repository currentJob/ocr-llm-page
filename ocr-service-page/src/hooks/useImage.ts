import { useRef, useState, useCallback } from 'react'

function loadImg(url: string): Promise<HTMLImageElement> {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.src    = url
  })
}

export function useImage() {
  const [imageUrl,   setImageUrl]   = useState<string | null>(null)
  const [natSize,    setNatSize]    = useState<{ w: number; h: number } | null>(null)
  const [rotation,   setRotation]   = useState(0)
  const [brightness, setBrightness] = useState(100)
  const [contrast,   setContrast]   = useState(100)

  const loadedImgRef   = useRef<HTMLImageElement | null>(null)
  const originalImgRef = useRef<HTMLImageElement | null>(null)
  const originalUrlRef = useRef<string | null>(null)
  const debounceRef    = useRef<ReturnType<typeof setTimeout> | null>(null)
  const rotReqRef      = useRef(0)

  const applyRotation = useCallback(async (deg: number) => {
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
    const nw  = Math.round(orig.naturalWidth  * cos + orig.naturalHeight * sin)
    const nh  = Math.round(orig.naturalWidth  * sin + orig.naturalHeight * cos)
    const c   = document.createElement('canvas')
    c.width = nw; c.height = nh
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, nw, nh)
    ctx.translate(nw / 2, nh / 2)
    ctx.rotate(rad)
    ctx.drawImage(orig, -orig.naturalWidth / 2, -orig.naturalHeight / 2)

    const url = c.toDataURL('image/jpeg', 0.93)
    const img = await loadImg(url)
    if (rotReqRef.current !== reqId) return
    loadedImgRef.current = img
    setImageUrl(url)
    setNatSize({ w: nw, h: nh })
  }, [])

  const handleRotationChange = useCallback((deg: number) => {
    const clamped = Math.max(-180, Math.min(180, deg))
    setRotation(clamped)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => applyRotation(clamped), 80)
  }, [applyRotation])

  const setFile = useCallback((url: string, img: HTMLImageElement, overrideNatSize?: { w: number; h: number }) => {
    originalImgRef.current = img
    originalUrlRef.current = url
    loadedImgRef.current   = img
    setImageUrl(url)
    setNatSize(overrideNatSize ?? { w: img.naturalWidth, h: img.naturalHeight })
  }, [])

  const resetTransforms = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    setRotation(0)
    setBrightness(100)
    setContrast(100)
  }, [])

  const getPreprocessedImg = useCallback(async (): Promise<HTMLImageElement> => {
    const src = loadedImgRef.current
    if (!src) throw new Error('No image')
    if (brightness === 100 && contrast === 100) return src
    const c = document.createElement('canvas')
    c.width = src.naturalWidth; c.height = src.naturalHeight
    const ctx = c.getContext('2d')!
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`
    ctx.drawImage(src, 0, 0)
    return loadImg(c.toDataURL('image/jpeg', 0.93))
  }, [brightness, contrast])

  return {
    imageUrl, natSize,
    rotation, brightness, contrast,
    loadedImgRef,
    setFile, resetTransforms,
    handleRotationChange,
    setBrightness, setContrast,
    getPreprocessedImg,
  }
}
