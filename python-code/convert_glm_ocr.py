"""
zai-org/GLM-OCR 다운로드 및 ONNX 변환 스크립트

모델 구조:
  CogViT 비전 인코더 + GLM-0.5B 언어 디코더 (encoder-decoder VLM)
  원본 크기: 약 2.65 GB (safetensors)

ONNX 변환 결과 (optimum 사용):
  encoder_model.onnx          - CogViT 이미지 인코더
  decoder_model_merged.onnx   - GLM 디코더 (KV-cache 포함)
  tokenizer.json 등 설정 파일

사용법:
  # 1단계: 다운로드 + ONNX 변환 (처음 한 번)
  python convert_glm_ocr.py

  # INT8 양자화 (변환 후 크기 축소, 약 650 MB)
  python convert_glm_ocr.py --quantize

  # 다운로드만 (변환 없이)
  python convert_glm_ocr.py --download-only

  # ONNX만 (이미 다운로드됨)
  python convert_glm_ocr.py --skip-download

필요 패키지:
  pip install torch transformers optimum[exporters] onnx==1.17.0 onnxruntime==1.25.1
  pip install huggingface_hub sentencepiece protobuf
  (양자화) pip install optimum[onnxruntime]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────────────

HF_MODEL_ID   = "zai-org/GLM-OCR"
OPSET         = 14      # VLM 은 opset 14 이상 필요

SCRIPT_DIR    = Path(__file__).parent
HF_LOCAL      = SCRIPT_DIR / "hf_models" / "zai-org__GLM-OCR"
ONNX_DIR      = SCRIPT_DIR / "onnx_models" / "glm_ocr"
ONNX_INT8_DIR = SCRIPT_DIR / "onnx_models" / "glm_ocr_int8"
PUBLIC_MODELS = SCRIPT_DIR.parent / "ocr-service-page" / "public" / "models" / "glm-ocr"

# glm-pipeline.ts 에서 참조하는 파일 이름 (변경 금지)
ENCODER_NAME         = "encoder_model.onnx"
DECODER_MERGED_NAME  = "decoder_model_merged.onnx"


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _hr(title: str = "") -> None:
    if title:
        print(f"\n── {title} {'─' * max(0, 48 - len(title))}")
    else:
        print("─" * 56)


def _ok(msg: str)   -> None: print(f"  [✓] {msg}")
def _skip(msg: str) -> None: print(f"  [-] {msg}")
def _warn(msg: str) -> None: print(f"  [!] {msg}")
def _fail(msg: str) -> None: print(f"  [✗] {msg}", file=sys.stderr)


def _require(*packages: str) -> None:
    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SystemExit(
            f"패키지 미설치: {', '.join(missing)}\n"
            f"  pip install {' '.join(missing)}"
        )


# ── 1. HuggingFace 다운로드 ──────────────────────────────────────────────────

def download(local_dir: Path) -> Path:
    _require("huggingface_hub")
    from huggingface_hub import snapshot_download

    marker = local_dir / ".download_complete"
    if marker.exists():
        _skip(f"이미 다운로드됨: {local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  모델 ID : {HF_MODEL_ID}")
    print(f"  저장 위치: {local_dir}")
    print(f"  예상 크기: ~2.7 GB (safetensors 포함)")

    snapshot_download(
        repo_id         = HF_MODEL_ID,
        local_dir       = str(local_dir),
        ignore_patterns = [
            "*.msgpack", "*.h5",
            "flax_model*", "rust_model*", "tf_model*",
        ],
    )
    marker.touch()
    _ok("다운로드 완료")
    return local_dir


# ── 2. ONNX 변환 (optimum) ───────────────────────────────────────────────────

def _check_optimum() -> None:
    try:
        import optimum  # noqa
        from optimum.exporters.onnx import main_export  # noqa
    except ImportError:
        raise SystemExit(
            "optimum 미설치:\n"
            "  pip install 'optimum[exporters]'"
        )


def convert_to_onnx(model_dir: Path, onnx_out: Path) -> None:
    _check_optimum()

    encoder_path = onnx_out / ENCODER_NAME
    decoder_path = onnx_out / DECODER_MERGED_NAME

    if encoder_path.exists() and decoder_path.exists():
        _skip(f"ONNX 파일 이미 존재: {onnx_out}")
        return

    onnx_out.mkdir(parents=True, exist_ok=True)

    # ── optimum Python API 로 내보내기 ────────────────────────────────────────
    # trust_remote_code 필수 (GLM-OCR 는 커스텀 아키텍처)
    try:
        from optimum.exporters.onnx import main_export

        print(f"  입력  : {model_dir}")
        print(f"  출력  : {onnx_out}")
        print(f"  opset : {OPSET}")
        print(f"  task  : image-text-to-text")
        print("  (GLM-OCR 는 2.65 GB — 변환에 수 분이 소요됩니다)")

        main_export(
            model_name_or_path = str(model_dir),
            output             = str(onnx_out),
            task               = "image-text-to-text",
            opset              = OPSET,
            trust_remote_code  = True,
            monolith           = False,   # encoder / decoder 분리
            no_post_process    = False,
        )
        _ok("optimum ONNX 변환 완료")

    except Exception as e:
        _warn(f"optimum Python API 실패: {e}")
        _warn("optimum-cli 로 재시도합니다...")
        _convert_via_cli(model_dir, onnx_out)


def _convert_via_cli(model_dir: Path, onnx_out: Path) -> None:
    """optimum-cli 를 subprocess 로 실행 (Python API 실패 시 폴백)."""
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model",             str(model_dir),
        "--task",              "image-text-to-text",
        "--opset",             str(OPSET),
        "--trust-remote-code",
        str(onnx_out),
    ]
    print(f"  실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            "optimum-cli 변환 실패.\n"
            f"  직접 실행:\n"
            f"  optimum-cli export onnx \\\n"
            f"    --model {model_dir} \\\n"
            f"    --task image-text-to-text \\\n"
            f"    --opset {OPSET} \\\n"
            f"    --trust-remote-code \\\n"
            f"    {onnx_out}"
        )
    _ok("optimum-cli 변환 완료")


# ── 3. INT8 양자화 ───────────────────────────────────────────────────────────

def quantize(onnx_dir: Path, out_dir: Path) -> None:
    """ONNX 모델을 INT8 동적 양자화합니다 (~650 MB)."""
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        raise SystemExit(
            "optimum[onnxruntime] 미설치:\n"
            "  pip install 'optimum[onnxruntime]'"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static    = False,
        per_channel  = False,
        reduce_range = False,
    )

    for onnx_file in onnx_dir.glob("*.onnx"):
        q_path = out_dir / onnx_file.name
        if q_path.exists():
            _skip(f"이미 양자화됨: {q_path.name}")
            continue

        print(f"  양자화: {onnx_file.name}")
        quantizer = ORTQuantizer.from_pretrained(str(onnx_dir), file_name=onnx_file.name)
        quantizer.quantize(
            save_dir              = str(out_dir),
            quantization_config   = qconfig,
        )
        mb = q_path.stat().st_size / 1024 / 1024
        _ok(f"{q_path.name}  ({mb:.1f} MB)")

    # 설정 파일 복사 (tokenizer 등)
    for f in onnx_dir.iterdir():
        if f.suffix not in (".onnx",) and not (out_dir / f.name).exists():
            shutil.copy2(f, out_dir / f.name)


# ── 4. 설정 파일 복사 ────────────────────────────────────────────────────────

def copy_config_files(hf_dir: Path, onnx_dir: Path) -> None:
    """tokenizer, preprocessor_config 등 설정 파일을 ONNX 디렉토리로 복사합니다."""
    config_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "config.json",
    ]
    for name in config_files:
        src = hf_dir / name
        dst = onnx_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            _ok(f"설정 복사: {name}")


# ── 5. glm_vocab.json 생성 ───────────────────────────────────────────────────

def build_glm_vocab(onnx_dir: Path, public_dir: Path) -> None:
    """tokenizer.json 에서 vocab 을 추출해 public 에 복사합니다.

    glm-pipeline.ts 는 tokenizer.json 을 직접 사용하므로
    별도 glm_vocab.json 변환 없이 파일만 복사합니다.
    """
    src = onnx_dir / "tokenizer.json"
    if not src.exists():
        _warn("tokenizer.json 없음 — vocab 복사 생략")
        return
    dst = public_dir / "tokenizer.json"
    if not dst.exists():
        shutil.copy2(src, dst)
        kb = dst.stat().st_size / 1024
        _ok(f"tokenizer.json 복사  ({kb:.0f} KB)")
    else:
        _skip("tokenizer.json 이미 존재")


# ── 6. ONNX 검증 ─────────────────────────────────────────────────────────────

def verify(onnx_dir: Path) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        _warn("onnxruntime 미설치 — 검증 생략")
        return

    for name in (ENCODER_NAME, DECODER_MERGED_NAME):
        path = onnx_dir / name
        if not path.exists():
            _warn(f"파일 없음: {name}")
            continue
        try:
            sess   = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inputs = [f"{i.name}: {i.shape}" for i in sess.get_inputs()]
            outputs= [f"{o.name}: {o.shape}" for o in sess.get_outputs()]
            mb     = path.stat().st_size / 1024 / 1024
            _ok(f"{name}  ({mb:.1f} MB)")
            for s in inputs:  print(f"      입력  {s}")
            for s in outputs: print(f"      출력  {s}")
        except Exception as e:
            _fail(f"{name} 검증 실패: {e}")


# ── 7. public/models/glm-ocr 복사 ───────────────────────────────────────────

def copy_to_public(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    copy_targets = list(src_dir.glob("*.onnx")) + [
        src_dir / "tokenizer.json",
        src_dir / "tokenizer_config.json",
        src_dir / "preprocessor_config.json",
        src_dir / "generation_config.json",
        src_dir / "config.json",
    ]

    for src in copy_targets:
        if not src.exists():
            continue
        dst = dst_dir / src.name
        if dst.exists():
            _skip(f"이미 존재: {dst.name}")
            continue
        shutil.copy2(src, dst)
        mb = dst.stat().st_size / 1024 / 1024
        label = f"{mb:.1f} MB" if mb >= 0.1 else f"{dst.stat().st_size / 1024:.0f} KB"
        _ok(f"{dst.name}  ({label})")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="zai-org/GLM-OCR 다운로드 + ONNX 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--download-only", action="store_true",
        help="다운로드만 수행 (ONNX 변환 없음)",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="다운로드 생략 (이미 hf_models/ 에 존재)",
    )
    p.add_argument(
        "--quantize", action="store_true",
        help="INT8 양자화 수행 (크기: ~2.65 GB → ~650 MB)",
    )
    p.add_argument(
        "--use-quantized", action="store_true",
        help="양자화된 모델을 public/ 에 복사",
    )
    p.add_argument(
        "--no-copy", action="store_true",
        help="public/models/glm-ocr 복사 생략",
    )
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    _hr()
    print("zai-org/GLM-OCR  ONNX 변환 파이프라인")
    print(f"  HF 모델 ID : {HF_MODEL_ID}")
    print(f"  로컬 저장  : {HF_LOCAL}")
    print(f"  ONNX 출력  : {ONNX_DIR}")
    print(f"  퍼블릭     : {PUBLIC_MODELS}")
    _hr()

    # ── 다운로드 ─────────────────────────────────────────────────────────────
    if not args.skip_download:
        _hr("1. HuggingFace 다운로드")
        download(HF_LOCAL)

    if args.download_only:
        _hr("완료 (다운로드만)")
        return

    # ── ONNX 변환 ─────────────────────────────────────────────────────────────
    _hr("2. ONNX 변환 (optimum)")
    convert_to_onnx(HF_LOCAL, ONNX_DIR)

    _hr("3. 설정 파일 복사")
    copy_config_files(HF_LOCAL, ONNX_DIR)

    # ── INT8 양자화 ───────────────────────────────────────────────────────────
    if args.quantize:
        _hr("4. INT8 양자화")
        quantize(ONNX_DIR, ONNX_INT8_DIR)
        copy_config_files(HF_LOCAL, ONNX_INT8_DIR)

    # ── ONNX 검증 ─────────────────────────────────────────────────────────────
    _hr("5. ONNX 검증")
    verify_dir = ONNX_INT8_DIR if (args.quantize and ONNX_INT8_DIR.exists()) else ONNX_DIR
    verify(verify_dir)

    # ── public 복사 ───────────────────────────────────────────────────────────
    if not args.no_copy:
        _hr("6. public/models/glm-ocr 복사")
        src_dir = ONNX_INT8_DIR if args.use_quantized and ONNX_INT8_DIR.exists() else ONNX_DIR
        copy_to_public(src_dir, PUBLIC_MODELS)
        build_glm_vocab(src_dir, PUBLIC_MODELS)

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    _hr()
    print("완료")
    print()
    files = {
        ENCODER_NAME:        PUBLIC_MODELS / ENCODER_NAME,
        DECODER_MERGED_NAME: PUBLIC_MODELS / DECODER_MERGED_NAME,
        "tokenizer.json":    PUBLIC_MODELS / "tokenizer.json",
    }
    for name, path in files.items():
        status = "✓" if path.exists() else "✗ 없음"
        mb     = f"  ({path.stat().st_size/1024/1024:.1f} MB)" if path.exists() else ""
        print(f"  {status}  {name}{mb}")
    _hr()


if __name__ == "__main__":
    main()
