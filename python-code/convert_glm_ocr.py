"""
GLM-OCR 모델 다운로드 및 ONNX 변환 스크립트

지원 변환 경로:
  1. HuggingFace PyTorch 모델 → ONNX  (기본값)
  2. PaddleOCR 추론 모델     → ONNX  (--paddle 옵션)
  3. 로컬 .pt / .pth 체크포인트 → ONNX  (--checkpoint 옵션)

사용법:
  python convert_glm_ocr.py                                # HuggingFace 다운로드 + 변환
  python convert_glm_ocr.py --model-id <hf-repo-id>       # HF 모델 ID 직접 지정
  python convert_glm_ocr.py --paddle <paddle_model_dir>   # PaddleOCR 모델 변환
  python convert_glm_ocr.py --checkpoint model.pt \\
      --vocab vocab.txt                                    # 로컬 체크포인트 변환

필요 패키지 (PyTorch 경로):
  pip install torch onnx==1.17.0 onnxruntime==1.25.1 huggingface_hub

필요 패키지 (PaddleOCR 경로):
  pip install paddlepaddle paddle2onnx==2.1.0 onnxruntime==1.25.1
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────────────
# 실제 HuggingFace 모델 ID로 교체하세요.
# 예시: "THUDM/glm-ocr", "your-org/glm-ocr-korean", ...
DEFAULT_HF_MODEL_ID = "THUDM/glm-ocr"

# glm-pipeline.ts 와 반드시 일치해야 합니다.
REC_H     = 64    # 인식 모델 입력 높이
REC_MAX_W = 256   # 인식 모델 입력 최대 너비
OPSET     = 11    # ONNX opset 버전

# 경로
SCRIPT_DIR    = Path(__file__).parent
ONNX_DIR      = SCRIPT_DIR / "onnx_models" / "glm_ocr_rec"
PUBLIC_MODELS = SCRIPT_DIR.parent / "ocr-service-page" / "public" / "models"


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _hr() -> None:
    print("─" * 56)


def _ok(msg: str) -> None:
    print(f"  [✓] {msg}")


def _skip(msg: str) -> None:
    print(f"  [-] {msg}")


def _fail(msg: str) -> None:
    print(f"  [✗] {msg}", file=sys.stderr)


# ── 1. HuggingFace 다운로드 ──────────────────────────────────────────────────

def download_hf(model_id: str, local_dir: Path) -> Path:
    """HuggingFace Hub에서 모델을 다운로드합니다."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub 미설치: pip install huggingface_hub")

    if local_dir.exists() and any(local_dir.iterdir()):
        _skip(f"이미 다운로드됨: {local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  HuggingFace 다운로드: {model_id}")
    print(f"  저장 위치: {local_dir}")

    snapshot_download(
        repo_id   = model_id,
        local_dir = str(local_dir),
        ignore_patterns = ["*.msgpack", "*.h5", "flax_model*", "rust_model*"],
    )
    _ok("다운로드 완료")
    return local_dir


# ── 2. PyTorch → ONNX ────────────────────────────────────────────────────────

def _find_pt_file(model_dir: Path) -> Path:
    """디렉토리에서 PyTorch 가중치 파일을 찾습니다."""
    for pattern in ("*.pt", "*.pth", "pytorch_model.bin", "model.safetensors"):
        files = sorted(model_dir.rglob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"PyTorch 가중치 파일을 찾을 수 없습니다: {model_dir}\n"
        f"  지원 형식: *.pt, *.pth, pytorch_model.bin, model.safetensors\n"
        f"  --model-id 또는 --checkpoint 옵션을 확인하세요."
    )


def convert_torch(model_dir: Path | None, checkpoint: Path | None, onnx_path: Path) -> None:
    """PyTorch 모델을 ONNX로 변환합니다."""
    try:
        import torch
    except ImportError:
        raise SystemExit("torch 미설치: pip install torch")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    # transformers AutoModel 로드 시도 (HuggingFace 표준 형식)
    model = None
    if model_dir and (model_dir / "config.json").exists():
        try:
            from transformers import AutoModel
            print(f"  transformers.AutoModel 로드 중: {model_dir}")
            model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
            model.eval()
            _ok("AutoModel 로드 완료")
        except Exception as e:
            print(f"  AutoModel 로드 실패 ({e}), 직접 로드 시도...")

    # 직접 .pt 파일 로드 (fallback)
    if model is None:
        pt_file = checkpoint or _find_pt_file(model_dir)
        print(f"  state_dict 로드 중: {pt_file}")

        # ── 여기서 모델 아키텍처를 직접 정의해야 합니다 ──────────────────────
        # 보유한 GLM-OCR 모델의 클래스를 아래에서 import 하거나 정의하세요.
        # 예:
        #   from glm_ocr_model import GlmOcrRecognizer
        #   model = GlmOcrRecognizer(num_classes=NUM_CLASSES)
        #   model.load_state_dict(torch.load(pt_file, map_location="cpu"))
        #
        # 현재는 state_dict 직접 로드를 지원하지 않습니다.
        # HuggingFace 형식(config.json + pytorch_model.bin)으로 배포된 모델을
        # 사용하거나, --paddle 옵션으로 PaddleOCR 모델을 변환하세요.
        raise RuntimeError(
            "PyTorch 직접 로드는 모델 클래스가 필요합니다.\n"
            "  1. HuggingFace 형식 모델(config.json 포함)을 사용하세요.\n"
            "  2. PaddleOCR 모델이라면 --paddle 옵션을 사용하세요."
        )

    # ── ONNX 내보내기 ─────────────────────────────────────────────────────────
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, REC_H, REC_MAX_W)

    print(f"  ONNX 내보내기: opset={OPSET}, 입력 {list(dummy.shape)}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version     = OPSET,
        input_names       = ["input"],
        output_names      = ["output"],
        dynamic_axes      = {
            "input":  {3: "width"},   # 너비만 동적 (높이는 고정)
            "output": {1: "seq_len"}, # 시퀀스 길이 동적
        },
        do_constant_folding = True,
    )

    mb = onnx_path.stat().st_size / 1024 / 1024
    _ok(f"ONNX 저장: {onnx_path.name}  ({mb:.2f} MB)")


# ── 3. PaddleOCR → ONNX ──────────────────────────────────────────────────────

def convert_paddle(paddle_dir: Path, onnx_path: Path) -> None:
    """PaddleOCR 추론 모델을 ONNX로 변환합니다."""
    try:
        import paddle2onnx
    except ImportError:
        raise SystemExit("paddle2onnx 미설치: pip install paddlepaddle paddle2onnx==2.1.0")

    # pdmodel 파일 탐색
    candidates = list(paddle_dir.rglob("inference.pdmodel")) + \
                 list(paddle_dir.rglob("model.pdmodel"))
    if not candidates:
        raise FileNotFoundError(
            f"inference.pdmodel 파일을 찾을 수 없습니다: {paddle_dir}"
        )
    model_file  = candidates[0]
    param_file  = model_file.with_name(
        "inference.pdiparams" if model_file.name == "inference.pdmodel" else "model.pdiparams"
    )
    if not param_file.exists():
        raise FileNotFoundError(f"pdiparams 파일 없음: {param_file}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  paddle2onnx 변환: {model_file.name} → {onnx_path.name}")

    paddle2onnx.export(
        model_filename  = str(model_file),
        params_filename = str(param_file),
        save_file       = str(onnx_path),
        opset_version   = OPSET,
    )

    if not onnx_path.exists():
        raise RuntimeError(f"ONNX 파일이 생성되지 않았습니다: {onnx_path}")

    mb = onnx_path.stat().st_size / 1024 / 1024
    _ok(f"ONNX 저장: {onnx_path.name}  ({mb:.2f} MB)")


# ── 4. 어휘 사전 생성 ────────────────────────────────────────────────────────

def _read_vocab_txt(vocab_file: Path) -> list[str]:
    """한 줄에 한 문자씩 적힌 텍스트 vocab 파일을 읽습니다."""
    lines = vocab_file.read_text(encoding="utf-8").splitlines()
    return [ln.rstrip("\n") for ln in lines]


def _read_vocab_json(vocab_file: Path) -> list[str]:
    """JSON 형식 vocab 파일을 읽습니다. {id: char} 또는 [char, ...] 형식 지원."""
    data = json.loads(vocab_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    raise ValueError(f"지원되지 않는 vocab JSON 형식: {vocab_file}")


def _find_vocab_in_yaml(model_dir: Path) -> list[str] | None:
    """PaddleOCR inference.yml 에서 character_dict 를 읽습니다."""
    try:
        import yaml
    except ImportError:
        return None

    yml_files = list(model_dir.rglob("inference.yml"))
    if not yml_files:
        return None

    cfg = yaml.safe_load(yml_files[0].read_text(encoding="utf-8"))
    chars = (
        cfg.get("PostProcess", {}).get("character_dict") or
        cfg.get("Global", {}).get("character_dict_path")
    )
    if isinstance(chars, list):
        return chars
    if isinstance(chars, str):
        dict_path = model_dir / chars
        if dict_path.exists():
            return _read_vocab_txt(dict_path)
    return None


def build_vocab_json(
    vocab_source: Path | None,
    model_dir:    Path | None,
    out_path:     Path,
) -> None:
    """glm_vocab.json 을 생성합니다.

    glm-pipeline.ts 가 기대하는 형식:
      { "charTable": ["a", "b", ...], "blankIdx": 0 }

    blank(CTC 공백) 은 index 0 으로 고정합니다.
    실제 문자는 index 1 부터 시작합니다.
    """
    if out_path.exists():
        _skip(f"어휘 사전 이미 존재: {out_path.name}")
        return

    char_list: list[str] | None = None

    # 1순위: 명시적으로 지정한 vocab 파일
    if vocab_source and vocab_source.exists():
        ext = vocab_source.suffix.lower()
        if ext == ".json":
            char_list = _read_vocab_json(vocab_source)
        else:
            char_list = _read_vocab_txt(vocab_source)
        _ok(f"vocab 로드: {vocab_source.name}  ({len(char_list)}개)")

    # 2순위: 모델 디렉토리의 inference.yml (PaddleOCR)
    if char_list is None and model_dir:
        char_list = _find_vocab_in_yaml(model_dir)
        if char_list:
            _ok(f"vocab (inference.yml에서 추출)  ({len(char_list)}개)")

    # 3순위: 모델 디렉토리의 vocab 파일 자동 탐색
    if char_list is None and model_dir:
        for pattern in ("vocab.txt", "char_dict.txt", "dict.txt", "characters.txt",
                        "vocab.json", "tokenizer.json"):
            files = list(model_dir.rglob(pattern))
            if files:
                ext = files[0].suffix.lower()
                char_list = _read_vocab_json(files[0]) if ext == ".json" else _read_vocab_txt(files[0])
                _ok(f"vocab 로드: {files[0].name}  ({len(char_list)}개)")
                break

    if char_list is None:
        print()
        print("  ⚠ 어휘 사전 파일을 찾지 못했습니다.")
        print("    --vocab 옵션으로 직접 파일 경로를 지정하세요.")
        print("    예: python convert_glm_ocr.py --vocab path/to/vocab.txt")
        print("    (glm_vocab.json 없이도 앱 실행은 되나, charDict.json 으로 폴백됩니다.)")
        return

    # blank 토큰이 이미 0번이면 그대로, 없으면 맨 앞에 삽입
    blank_idx = 0
    if char_list[0] not in ("", "<blank>", "blank"):
        char_list = [""] + char_list   # index 0 = blank

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"charTable": char_list, "blankIdx": blank_idx},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _ok(f"glm_vocab.json 저장: {out_path}  ({len(char_list)}개 항목)")


# ── 5. ONNX 검증 ─────────────────────────────────────────────────────────────

def verify_onnx(onnx_path: Path) -> None:
    """ONNX 모델의 입출력 shape 를 확인합니다."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  onnxruntime 미설치 — 검증 생략")
        return

    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
    except Exception as e:
        _fail(f"ONNX 그래프 검증 실패: {e}")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]
    print(f"  입력  : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    print(f"  출력  : {out.name}  shape={out.shape}  dtype={out.type}")

    # 실제 추론 테스트
    import numpy as np
    dummy = np.zeros((1, 3, REC_H, REC_MAX_W), dtype=np.float32)
    result = sess.run([out.name], {inp.name: dummy})
    o_shape = result[0].shape
    print(f"  추론 결과 shape: {o_shape}")

    if len(o_shape) != 3 or o_shape[0] != 1:
        _fail(f"출력 shape 이 예상([1, T, num_classes])과 다릅니다: {o_shape}")
    else:
        _ok(f"검증 통과  (num_classes={o_shape[2]}, time_steps={o_shape[1]})")


# ── 6. public/models 복사 ────────────────────────────────────────────────────

def copy_to_public(onnx_path: Path, vocab_path: Path) -> None:
    """변환된 파일을 웹 앱의 public/models/ 디렉토리로 복사합니다."""
    if not PUBLIC_MODELS.exists():
        print(f"  public/models 경로 없음 — 수동 복사 필요: {PUBLIC_MODELS}")
        return

    for src in (onnx_path, vocab_path):
        if not src.exists():
            continue
        dst = PUBLIC_MODELS / src.name
        shutil.copy2(src, dst)
        mb = dst.stat().st_size / 1024 / 1024
        _ok(f"복사 완료: {dst}  ({mb:.2f} MB)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLM-OCR 모델 다운로드 및 ONNX 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--model-id", default=DEFAULT_HF_MODEL_ID, metavar="HF_REPO_ID",
        help=f"HuggingFace 모델 ID (기본값: {DEFAULT_HF_MODEL_ID})",
    )
    src.add_argument(
        "--paddle", metavar="DIR",
        help="PaddleOCR 추론 모델 디렉토리 (inference.pdmodel 포함)",
    )
    src.add_argument(
        "--checkpoint", metavar="FILE",
        help="로컬 PyTorch 체크포인트 (.pt / .pth)",
    )
    p.add_argument(
        "--vocab", metavar="FILE",
        help="어휘 사전 파일 (.txt 또는 .json)",
    )
    p.add_argument(
        "--no-copy", action="store_true",
        help="public/models 로 복사하지 않음",
    )
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    onnx_path  = ONNX_DIR / "glm_ocr_rec.onnx"
    vocab_path = ONNX_DIR / "glm_vocab.json"

    _hr()
    print("GLM-OCR ONNX 변환")
    print(f"  입력 스펙: [1, 3, {REC_H}, {REC_MAX_W}]  opset={OPSET}")
    print(f"  출력 경로: {ONNX_DIR}")
    _hr()

    model_dir: Path | None = None

    # ── ONNX 변환 ─────────────────────────────────────────────────────────────
    if onnx_path.exists():
        mb = onnx_path.stat().st_size / 1024 / 1024
        _skip(f"ONNX 파일 이미 존재: {onnx_path.name}  ({mb:.2f} MB)")
    else:
        if args.paddle:
            # PaddleOCR 경로
            print("── PaddleOCR 모델 변환")
            paddle_dir = Path(args.paddle)
            model_dir  = paddle_dir
            convert_paddle(paddle_dir, onnx_path)

        elif args.checkpoint:
            # 로컬 체크포인트 경로
            print("── 로컬 체크포인트 변환")
            convert_torch(None, Path(args.checkpoint), onnx_path)

        else:
            # HuggingFace 다운로드 + PyTorch 변환
            print(f"── HuggingFace 다운로드: {args.model_id}")
            hf_local = SCRIPT_DIR / "hf_models" / args.model_id.replace("/", "__")
            model_dir = download_hf(args.model_id, hf_local)
            print()
            print("── PyTorch → ONNX 변환")
            convert_torch(model_dir, None, onnx_path)

    print()

    # ── 어휘 사전 생성 ─────────────────────────────────────────────────────────
    print("── 어휘 사전(glm_vocab.json) 생성")
    vocab_src = Path(args.vocab) if args.vocab else None
    build_vocab_json(vocab_src, model_dir, vocab_path)
    print()

    # ── ONNX 검증 ─────────────────────────────────────────────────────────────
    if onnx_path.exists():
        print("── ONNX 모델 검증")
        verify_onnx(onnx_path)
        print()

    # ── public/models 복사 ────────────────────────────────────────────────────
    if not args.no_copy:
        print("── public/models 복사")
        copy_to_public(onnx_path, vocab_path)
        print()

    _hr()
    print("완료")
    print(f"  ONNX  : {onnx_path}")
    print(f"  vocab : {vocab_path}")
    _hr()


if __name__ == "__main__":
    main()
