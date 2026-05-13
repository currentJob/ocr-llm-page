import * as ort from 'onnxruntime-web/wasm'

// Keep ORT on the plain WASM backend. Supplying an mjs override points ORT at
// public/*.mjs, which Vite refuses to import during dev.
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  wasm: `${import.meta.env.BASE_URL}ort-wasm-simd-threaded.wasm`,
}

export * from 'onnxruntime-web/wasm'
