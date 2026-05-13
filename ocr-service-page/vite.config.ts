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
/**
 * ORT 1.26.0 은 Um() 내부에서 항상 ort-wasm-simd-threaded.jsep.mjs 를 로드한다.
 * pipeline.ts 에서 wasmPaths.mjs 를 base(.mjs)로 명시해 jsep 를 우회하므로
 * 이 stub 은 직접 트리거되지 않는다. 단, @huggingface/transformers 내부에서
 * 의도치 않게 jsep/jspi/asyncify 변형을 import() 하려 할 때의 방어 목적으로 유지.
 *
 * ※ no-op async 함수: Bu=undefined → we() 실패를 유발하므로
 *    throw 방식으로 변경해 ORT 내부 try-catch 가 WASM 초기화 실패로 처리하게 함.
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
      // throw 로 import() 자체를 reject → ORT try-catch 가 잡아 초기화 실패 처리
      return 'throw new Error("ort-stub: jsep/jspi/asyncify backend not supported")'
    }
  },
}

export default defineConfig({
  base,
  plugins: [react(), ortPublicMjsStub],
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-dom/client',
      'react/jsx-dev-runtime',
      'react/jsx-runtime',
    ],
    // onnxruntime-web 는 제외하지 않음 — exclude 시 raw ESM 로드로 전환되어
    // ort-wasm-simd-threaded.mjs (WASM bootstrap) 까지 public/ 에서
    // 동적 import() 하려다 Vite 에 차단되면서 WASM 초기화가 실패함.
    exclude: ['@huggingface/transformers'],
  },
  server: {
    hmr: {
      host: '127.0.0.1',
    },
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
