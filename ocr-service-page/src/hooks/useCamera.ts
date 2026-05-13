import { useRef, useState, useCallback } from 'react'

export function useCamera(onCapture: (file: File) => void) {
  const [showCamera, setShowCamera] = useState(false)
  const videoRef     = useRef<HTMLVideoElement>(null)
  const streamRef    = useRef<MediaStream | null>(null)
  const onCaptureRef = useRef(onCapture)
  onCaptureRef.current = onCapture   // keep ref current without re-rendering

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setShowCamera(false)
  }, [])

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
      streamRef.current = stream
      setShowCamera(true)
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = stream }, 50)
    } catch {
      alert('카메라를 사용할 수 없습니다.')
    }
  }, [])

  const captureCamera = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    c.toBlob(blob => {
      if (!blob) return
      stopCamera()
      onCaptureRef.current(new File([blob], 'camera.jpg', { type: 'image/jpeg' }))
    }, 'image/jpeg', 0.92)
  }, [stopCamera])

  const stopOnUnmount = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
  }, [])

  return { showCamera, videoRef, startCamera, stopCamera, captureCamera, stopOnUnmount }
}
