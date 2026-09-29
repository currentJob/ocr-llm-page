/**
 * 외부 사이트용 진입점 (vite.lib.config.ts 로 빌드).
 *
 *   const { KoreanOCR } = await import('https://currentjob.github.io/ocr-llm-page/lib/korean-ocr.mjs')
 *   const ocr = await KoreanOCR.create(p => console.log(p.step))
 *   const items = await ocr.predict(imageElement)   // [{ text, recScore, detScore, box }]
 *
 * 앱과 같은 파이프라인·모델을 쓴다. 이미지는 브라우저 밖으로 나가지 않는다.
 */
export { KoreanOCR } from './pipeline'
export type { OcrItem, LoadProgress } from './pipeline'
