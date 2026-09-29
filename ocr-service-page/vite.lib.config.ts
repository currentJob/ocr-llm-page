import { defineConfig } from 'vite'
import { ortPublicMjsStub } from './vite.config'

/**
 * OCR 파이프라인만 다른 사이트에서 가져다 쓸 수 있게 ES 모듈 하나로 묶는다.
 * 결과: dist/lib/korean-ocr.mjs → https://<owner>.github.io/<repo>/lib/korean-ocr.mjs
 *
 * 모델(.onnx)·문자 사전·ORT wasm 은 이 사이트에 이미 올라가 있는 것을 그대로 쓴다.
 * 그래서 base 를 **절대 주소**로 굽는다 — 다른 출처의 페이지가 import 해도 여기서 받아 온다
 * (GitHub Pages 는 Access-Control-Allow-Origin: * 를 준다).
 */
const base = process.env.VITE_LIB_BASE ?? 'https://currentjob.github.io/ocr-llm-page/'

export default defineConfig({
  base,
  plugins: [ortPublicMjsStub],
  publicDir: false,
  build: {
    outDir: 'dist/lib',
    emptyOutDir: true,
    // lib 모드는 wasm 을 base64 로 통째로 박아 34MB 가 된다. 일반 빌드로 진입점만 ES 모듈로 내보내면
    // wasm 은 파일로 남고, 실제로는 위 base 의 ort-wasm-simd-threaded.wasm 을 받는다.
    rollupOptions: {
      input: 'src/ocr/lib.ts',
      preserveEntrySignatures: 'strict',
      output: { format: 'es', entryFileNames: 'korean-ocr.mjs' },
    },
  },
})
