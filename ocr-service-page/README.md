# Korean OCR · LLM 요약

이미지 속 한국어 문장을 브라우저에서 읽고, 추출한 내용을 LLM 으로 요약합니다. 서버 없이 동작하며 이미지는 기기 밖으로 나가지 않습니다.

- 사이트: https://currentjob.github.io/ocr-llm-page/

## 구성

| 부분 | 내용 |
|---|---|
| 글자 인식 | PP-OCRv5 한국어 파이프라인(문서 방향 → 텍스트 검출 → 줄 방향 → 인식), ONNX Runtime Web. WebGPU 가 되면 GPU, 아니면 WASM(CPU) |
| 요약 | Qwen2.5-0.5B-Instruct, Transformers.js (WebGPU 우선, 없으면 CPU) |
| 화면 | React + TypeScript + Vite. 이미지 업로드·카메라 촬영, 영역 선택, 결과 검색·복사·내보내기, 기록 |

## 실행

```sh
npm ci
npm run dev
npm run build     # dist/ (앱) + dist/lib/korean-ocr.mjs (다른 사이트용 OCR 모듈)
```

GitHub Pages 배포는 `.github/workflows/deploy.yml` 이 `main` push 때 합니다(`VITE_BASE_URL`, `VITE_LIB_BASE` 를 넣어 빌드).

## 다른 사이트에서 OCR 쓰기

```js
const { KoreanOCR } = await import('https://currentjob.github.io/ocr-llm-page/lib/korean-ocr.mjs')
const ocr = await KoreanOCR.create((p) => console.log(p.step))   // 첫 사용 때 모델 약 30MB. 두 번째 인자로 { backend: 'wasm' } 등 지정 가능
const items = await ocr.predict(imageElement)                     // [{ text, recScore, detScore, box }]
```

모델·문자 사전·wasm 은 이 사이트의 파일을 절대 주소로 받습니다. 인식 모델은 한국어·영문·숫자용이라 한자는 읽지 못합니다.
