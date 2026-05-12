"""
PP-OCRv5 전체 파이프라인 모델 ONNX 변환 스크립트

변환 대상:
  1. PP-LCNet_x1_0_doc_ori       - 문서 방향 분류  (4-class: 0/90/180/270°)
  2. PP-OCRv5_mobile_det         - 텍스트 검출     (DB algorithm)
  3. PP-LCNet_x1_0_textline_ori  - 텍스트 라인 방향 분류 (2-class: 0/180°)
  4. korean_PP-OCRv5_mobile_rec  - 한국어 텍스트 인식

※ UVDoc(문서 왜곡 보정)은 변환 복잡도로 인해 제외

필요 패키지:
    pip install paddlepaddle paddle2onnx==2.1.0
"""

import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_DIR    = Path(__file__).parent
MODELS_ROOT = BASE_DIR / "paddle_models"
ONNX_ROOT   = BASE_DIR / "onnx_models"
OPSET       = 11

MODELS = [
    "PP-LCNet_x1_0_doc_ori",
    "PP-OCRv5_mobile_det",
    "PP-LCNet_x1_0_textline_ori",
    "korean_PP-OCRv5_mobile_rec",
]

BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com"
    "/paddlex/official_inference_model/paddle3.0.0"
)

def _progress(bc: int, bs: int, total: int) -> None:
    if total > 0:
        pct = min(100, bc * bs * 100 // total)
        bar = "#" * (pct // 5)
        print(f"\r    [{bar:<20}] {pct:3d}%", end="", flush=True)

def download(name: str) -> Path:
    infer_name = f"{name}_infer"
    tar_path   = BASE_DIR / f"{infer_name}.tar"
    if tar_path.exists():
        print(f"  [SKIP] 이미 다운로드됨: {tar_path.name}")
        return tar_path
    url = f"{BASE_URL}/{infer_name}.tar"
    print(f"  [다운로드] {url}")
    urllib.request.urlretrieve(url, tar_path, reporthook=_progress)
    print()
    return tar_path

def extract(name: str, tar_path: Path) -> Path:
    model_dir = MODELS_ROOT / name
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"  [SKIP] 이미 압축 해제됨: {model_dir.name}")
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if str(tar_path).endswith(".gz") else "r:"
    with tarfile.open(tar_path, mode) as tar:
        members  = tar.getmembers()
        top_dirs = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
        strip    = top_dirs.pop() if len(top_dirs) == 1 else None

        for m in members:
            parts = Path(m.name).parts
            if not parts:
                continue
            if strip and parts[0] == strip:
                if len(parts) == 1:
                    continue
                rel = Path(*parts[1:])
            else:
                rel = Path(*parts)
            target = model_dir / rel
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                fobj = tar.extractfile(m)
                if fobj:
                    target.write_bytes(fobj.read())

    files = [f"{f.name} ({f.stat().st_size//1024}KB)" for f in model_dir.rglob("*") if f.is_file()]
    print(f"  [압축해제] {model_dir.name} → {files}")
    return model_dir

def _check_paddle2onnx() -> None:
    r = subprocess.run(
        [sys.executable, "-c", "import paddle2onnx; print(paddle2onnx.__version__)"],
        capture_output=True,
        errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(
            "paddle2onnx 를 불러올 수 없습니다.\n"
            "  pip uninstall paddle2onnx -y\n"
            "  pip install paddle2onnx==2.1.0"
        )
    print(f"  paddle2onnx {r.stdout.strip()} 확인")

def convert(name: str, model_dir: Path) -> Path:
    onnx_dir  = ONNX_ROOT / name
    onnx_path = onnx_dir / f"{name}.onnx"
    if onnx_path.exists():
        mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"  [SKIP] 이미 변환됨: {onnx_path.relative_to(BASE_DIR)}  ({mb:.2f} MB)")
        return onnx_path

    for filename in ("inference.pdmodel", "inference.json"):
        candidates = list(model_dir.rglob(filename))
        if candidates:
            model_file = candidates[0]
            break
    else:
        files = [f.name for f in model_dir.rglob("*") if f.is_file()]
        raise FileNotFoundError(f"모델 파일 없음 in {model_dir}: {files}")

    pdiparams = model_file.with_name("inference.pdiparams")
    if not pdiparams.exists():
        raise FileNotFoundError(f"inference.pdiparams 없음: {pdiparams}")

    onnx_dir.mkdir(parents=True, exist_ok=True)
    import paddle2onnx
    paddle2onnx.export(
        model_filename  = str(model_file),
        params_filename = str(pdiparams),
        save_file       = str(onnx_path),
        opset_version   = OPSET,
    )
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX 파일이 생성되지 않았습니다: {onnx_path}")

    mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  [완료] {onnx_path.relative_to(BASE_DIR)}  ({mb:.2f} MB)")
    return onnx_path

def main() -> None:
    print("── paddle2onnx 확인")
    _check_paddle2onnx()
    print()

    for name in MODELS:
        print(f"── {name}")
        try:
            tar_path  = download(name)
            model_dir = extract(name, tar_path)
            convert(name, model_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    print("─" * 50)
    print("변환 결과:")
    for name in MODELS:
        p      = ONNX_ROOT / name / f"{name}.onnx"
        status = "✓" if p.exists() else "✗ 없음"
        label  = str(p.relative_to(BASE_DIR)) if p.exists() else name
        print(f"  {status}  {label}")

if __name__ == "__main__":
    main()