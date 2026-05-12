#!/usr/bin/env python3
"""
Transformers.js 호환 ONNX 모델 다운로드 스크립트

onnx-community 의 사전 변환된 ONNX 모델을 다운로드하여
public/models/llm/ 에 저장합니다.
torch / numpy / scipy 불필요 — huggingface_hub 만 사용.

사용법:
  python convert_llm.py              # Qwen2.5-0.5B (기본, ~350 MB)
  python convert_llm.py --large      # Qwen2.5-1.5B (~900 MB)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── 모델 구성 ──────────────────────────────────────────────────────────────────

MODELS = {
    "small": {
        "hf_id":  "onnx-community/Qwen2.5-0.5B-Instruct",
        "desc":   "Qwen2.5-0.5B-Instruct ONNX (Transformers.js, ~350 MB)",
    },
    "large": {
        "hf_id":  "onnx-community/Qwen2.5-1.5B-Instruct",
        "desc":   "Qwen2.5-1.5B-Instruct ONNX (Transformers.js, ~900 MB)",
    },
}

DEST_ROOT = (
    Path(__file__).parent.parent
    / "ocr-service-page" / "public" / "models" / "llm"
)

# ── 패키지 설치 ────────────────────────────────────────────────────────────────

def pip(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# ── 다운로드 ──────────────────────────────────────────────────────────────────

def download_model(model_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    # 이미 다운로드된 경우 건너뜀
    if (dest / "config.json").exists():
        size_mb = sum(
            f.stat().st_size for f in dest.rglob("*") if f.is_file()
        ) / 1024 ** 2
        print(f"  ✓ 이미 존재: {dest}  ({size_mb:.0f} MB)")
        return

    print(f"  HuggingFace 다운로드: {model_id}")
    print(f"  저장 위치: {dest}")
    print("  (첫 실행 시 수백 MB — 완료까지 기다려주세요)\n")

    # q4 ONNX + 설정/토크나이저 파일만 다운로드
    ALLOW = [
        "*.json",
        "*.txt",
        "onnx/*q4*.onnx",
        "onnx/*q4*.onnx_data",
    ]

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        allow_patterns=ALLOW,
        local_dir=str(dest),
    )

    total_mb = sum(
        f.stat().st_size for f in dest.rglob("*") if f.is_file()
    ) / 1024 ** 2
    print(f"\n  ✓ 완료: {total_mb:.0f} MB")

# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Transformers.js ONNX 모델 다운로드")
    parser.add_argument(
        "--large", action="store_true",
        help="Qwen2.5-1.5B 다운로드 (기본: 0.5B)",
    )
    args = parser.parse_args()

    cfg      = MODELS["large" if args.large else "small"]
    model_id = cfg["hf_id"]
    dest     = DEST_ROOT / model_id

    print(f"\n=== {cfg['desc']} ===\n")

    print("[1/2] huggingface_hub 설치 확인")
    pip("huggingface_hub")

    print("\n[2/2] 모델 다운로드")
    download_model(model_id, dest)

    print(f"\n{'='*50}")
    print(f"완료! 브라우저에서 자동으로 로컬 모델을 사용합니다.")
    print(f"위치: {dest}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
