import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const base = process.env.VITE_BASE_URL ?? '/'

export default defineConfig({
  base,
  plugins: [react()],
  optimizeDeps: {
    // onnxruntime-web 는 Vite pre-bundle 대상에서 제외
    // (동적 import 체인이 복잡해 pre-bundle 시 오류 발생)
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },
  resolve: {
    alias: {
      // WASM 전용 번들로 교체 → .jsep.mjs (JSEP/WebGPU) 동적 import 제거
      // onnxruntime-web 기본 진입점은 WebGPU 감지 시 /public 의 .jsep.mjs 를
      // 동적 import() 하는데, Vite 5+ 는 public/ 파일의 모듈 import 를 차단함.
      // WASM 전용 번들은 JSEP 로딩 코드 자체가 없으므로 이 문제가 발생하지 않음.
      'onnxruntime-web': 'onnxruntime-web/dist/ort.wasm.min.mjs',
    },
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
