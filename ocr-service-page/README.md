# Korean OCR · LLM 요약

## 다른 사이트에서 OCR 쓰기

빌드하면 앱과 함께 OCR 파이프라인만 담은 ES 모듈 `lib/korean-ocr.mjs` 가 배포된다(약 80KB).
모델·문자 사전·ORT wasm 은 이 사이트의 파일을 절대 주소로 받는다(GitHub Pages 는 CORS `*`).

```js
const { KoreanOCR } = await import('https://currentjob.github.io/ocr-llm-page/lib/korean-ocr.mjs')
const ocr = await KoreanOCR.create((p) => console.log(p.step))   // 첫 사용 때 모델 약 30MB
const items = await ocr.predict(imageElement)                     // [{ text, recScore, detScore, box }]
```

- 이미지는 브라우저 밖으로 나가지 않는다.
- 인식 모델은 한국어·영문·숫자용이다(한자는 읽지 못한다).
- 쓰는 곳: [City Walk Planner](https://currentjob.github.io/city-walk-planner/) 동행 경비의 "영수증 사진으로 채우기".
- 빌드 설정은 `vite.lib.config.ts`, 모델 주소는 환경 변수 `VITE_LIB_BASE`(배포 워크플로가 넣는다).

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
