/**
 * ONNX Runtime 로더. WebGPU 어댑터가 있으면 WebGPU 빌드(연산 대부분 GPU, 나머지 CPU),
 * 없으면 WASM 빌드만 받는다. wasm 은 public/ 의 파일을 쓴다 — Vite 사전 번들 안에서는
 * ORT 의 상대 경로 계산이 어긋나기 때문이다.
 */
export type Ort = typeof import('onnxruntime-web/wasm')
export type Backend = 'webgpu' | 'wasm'

async function hasWebGpu(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
  if (!gpu) return false
  try { return !!(await gpu.requestAdapter()) } catch { return false }
}

export async function loadOrt(prefer?: Backend): Promise<{ ort: Ort; backend: Backend }> {
  const base = import.meta.env.BASE_URL
  if (prefer !== 'wasm' && await hasWebGpu()) {
    const ort = await import('onnxruntime-web/webgpu') as unknown as Ort
    ort.env.wasm.numThreads = 1
    // 모양 계산용 작은 연산을 CPU 에 두는 건 ORT 의 의도된 동작인데, 세션마다 경고를 찍어 끈다
    ort.env.logLevel = 'error'
    ort.env.wasm.wasmPaths = { wasm: `${base}ort-wasm-simd-threaded.asyncify.wasm` }
    return { ort, backend: 'webgpu' }
  }
  const ort = await import('onnxruntime-web/wasm')
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = { wasm: `${base}ort-wasm-simd-threaded.wasm` }
  return { ort, backend: 'wasm' }
}
