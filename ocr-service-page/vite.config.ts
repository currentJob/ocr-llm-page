import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const base = process.env.VITE_BASE_URL ?? '/'

/**
 * onnxruntime-web 1.20+ 는 WebGPU(JSEP) / JSPI / asyncify 백엔드 파일을
 * 동적 import() 로 로드한다. 이 파일들은 /public 에 있는데, Vite 5+ 는
 * public/ 파일을 ES 모듈로 import() 하는 것을 차단한다.
 *
 * 해결책: `enforce: 'pre'` 플러그인으로 Vite 내장 검사보다 먼저 실행해
 * 해당 import 를 빈 가상 모듈(stub)로 대체한다.
 * executionProviders: ['wasm'] 만 사용하므로 JSEP/JSPI/asyncify 가 없어도
 * ORT 는 try-catch 로 처리 후 WASM 백엔드로 정상 폴백한다.
 *
 * ※ resolve.alias 방식은 @huggingface/transformers 내부의 중첩된
 *   onnxruntime-web 까지 영향을 줘 `onnxruntime-web/webgpu` 서브패스
 *   임포트가 깨지는 부작용이 있어 플러그인 방식을 사용한다.
 */
const ortPublicMjsStub: Plugin = {
  name:    'ort-public-mjs-stub',
  enforce: 'pre',
  resolveId(id: string) {
    if (/ort-wasm-simd-threaded\.(jsep|jspi|asyncify)\.m?js$/.test(id)) {
      return '\0ort-stub'
    }
  },
  load(id: string) {
    if (id === '\0ort-stub') {
      return 'export default undefined'
    }
  },
}

export default defineConfig({
  base,
  plugins: [react(), ortPublicMjsStub],
  optimizeDeps: {
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
