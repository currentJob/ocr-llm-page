"""
PaddleOCR(
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,        # ※ UVDoc는 미구현 (복잡도로 인해 생략)
    use_textline_orientation=True,
    device="cpu",
)
위 파이프라인과 동일한 결과를 ONNX 런타임으로 구현합니다.

필요 패키지:
    pip install onnxruntime opencv-python numpy pyyaml

사전 준비:
    python convert_all_models.py
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
ONNX_ROOT = BASE_DIR / "onnx_models"
MDL_ROOT  = BASE_DIR / "paddle_models"

DOC_ORI_ONNX     = ONNX_ROOT / "PP-LCNet_x1_0_doc_ori"       / "PP-LCNet_x1_0_doc_ori.onnx"
DET_ONNX         = ONNX_ROOT / "PP-OCRv5_mobile_det"          / "PP-OCRv5_mobile_det.onnx"
TEXTLINE_ORI_ONNX = ONNX_ROOT / "PP-LCNet_x1_0_textline_ori" / "PP-LCNet_x1_0_textline_ori.onnx"
REC_ONNX         = ONNX_ROOT / "korean_PP-OCRv5_mobile_rec"   / "korean_PP-OCRv5_mobile_rec.onnx"
REC_YML          = MDL_ROOT  / "korean_PP-OCRv5_mobile_rec"   / "inference.yml"
# ─────────────────────────────────────────────────────────────────────────────

# ── 문자 사전 ─────────────────────────────────────────────────────────────────

def _load_char_list() -> tuple[list[str], int]:
    """inference.yml 의 PostProcess.character_dict 에서 문자 리스트 로드.

    Returns:
        (chars, blank_idx)
        PP-OCRv5 CTC 규칙: blank = len(chars) (마지막 클래스)
    """
    if not REC_YML.exists():
        raise FileNotFoundError(
            f"문자 사전 파일이 없습니다: {REC_YML}\n"
            "  python convert_all_models.py 를 먼저 실행하세요."
        )
    with open(REC_YML, encoding="utf-8") as f:
        yml = yaml.safe_load(f)
    yml_chars = yml["PostProcess"]["character_dict"]

    # PaddleOCR v3 CTCLabelDecode.add_special_char():
    #   char_table = ['blank'] + yml_chars + [' ']
    #   index 0       = blank  ← get_ignored_tokens() 반환값
    #   index 1..N    = yml 문자 (11945개)
    #   index N+1     = ' ' (space)
    #   total         = 11947 ✓
    char_table = [None] + list(yml_chars) + [" "]   # index 0은 blank(None)
    blank_idx  = 0

    print(f"[사전] {len(char_table)}개 (blank=0, space={len(char_table)-1})")
    return char_table, blank_idx

# ── 전처리 ────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_chw(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def _imagenet_norm(img_rgb: np.ndarray) -> np.ndarray:
    return (img_rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD

def preprocess_doc_ori(img: np.ndarray) -> np.ndarray:
    """PP-LCNet_x1_0_doc_ori 전처리.
    ResizeImage(resize_short=256) → CenterCrop(224) → ImageNet norm
    """
    h, w  = img.shape[:2]
    short = min(h, w)
    scale = 256 / short
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))

    # center crop 224×224
    y0 = (nh - 224) // 2
    x0 = (nw - 224) // 2
    crop = resized[y0:y0 + 224, x0:x0 + 224]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return _to_chw(_imagenet_norm(rgb))

def preprocess_det(img: np.ndarray, resize_long: int = 960) -> tuple[np.ndarray, float, float]:
    """PP-OCRv5_mobile_det 전처리.
    DetResizeForTest(resize_long=960) → ImageNet norm
    Returns: (tensor [1,3,H,W], ratio_h, ratio_w)  ratio = new/orig
    """
    h, w = img.shape[:2]
    if max(h, w) > resize_long:
        ratio = resize_long / max(h, w)
    else:
        ratio = 1.0
    new_h = max(int(round(h * ratio / 32) * 32), 32)
    new_w = max(int(round(w * ratio / 32) * 32), 32)

    resized = cv2.resize(img, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = _to_chw(_imagenet_norm(rgb))
    return tensor, new_h / h, new_w / w   # ratio = resized / orig

def preprocess_textline_ori(img: np.ndarray) -> np.ndarray:
    """PP-LCNet_x1_0_textline_ori 전처리.
    ResizeImage(size=[160, 80]) → ImageNet norm  (W=160, H=80)
    """
    resized = cv2.resize(img, (160, 80))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return _to_chw(_imagenet_norm(rgb))

def preprocess_rec(img: np.ndarray, max_w: int = 320) -> np.ndarray:
    """korean_PP-OCRv5_mobile_rec 전처리.
    RecResizeImg(image_shape=[3, 48, 320]): 비율 유지 resize(H=48) + 오른쪽 패딩 + norm
    """
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ratio   = w / h
    new_w   = min(int(math.ceil(48 * ratio)), max_w)
    new_w   = max(new_w, 1)
    resized = cv2.resize(img, (new_w, 48))

    # normalize [-1, 1]
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - 0.5) / 0.5
    norm = norm.transpose(2, 0, 1)   # HWC → CHW

    # 오른쪽 패딩
    canvas = np.zeros((3, 48, max_w), dtype=np.float32)
    canvas[:, :, :new_w] = norm
    return canvas[np.newaxis]

# ── DB 후처리 ─────────────────────────────────────────────────────────────────

def _box_score(prob: np.ndarray, contour: np.ndarray) -> float:
    ph, pw = prob.shape
    pts = np.int32(contour.reshape(-1, 2))
    pts[:, 0] = np.clip(pts[:, 0], 0, pw - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, ph - 1)
    mask = np.zeros_like(prob, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    denom = mask.sum()
    return float((prob * mask).sum() / (denom + 1e-6)) if denom else 0.0

def _unclip(box: np.ndarray, ratio: float = 1.5) -> np.ndarray:
    cx = box[:, 0].mean()
    cy = box[:, 1].mean()
    pts = box.astype(np.float32)
    pts[:, 0] = (pts[:, 0] - cx) * ratio + cx
    pts[:, 1] = (pts[:, 1] - cy) * ratio + cy
    return pts.astype(np.int32)

def db_postprocess(
    pred: np.ndarray,
    ratio_h: float,
    ratio_w: float,
    orig_h: int,
    orig_w: int,
    thresh: float       = 0.3,
    box_thresh: float   = 0.5,
    unclip_ratio: float = 1.5,
) -> tuple[list[np.ndarray], list[float]]:
    """DB 검출 후처리: 확률맵 → 원본 좌표계 박스 리스트."""
    raw  = pred[0, 1] if pred.shape[1] == 2 else pred[0, 0]
    prob = raw if raw.max() <= 1.0 else 1.0 / (1.0 + np.exp(-raw))

    binary   = (prob > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []
    for cnt in contours:
        if len(cnt) < 4:
            continue
        score = _box_score(prob, cnt)
        if score < box_thresh:
            continue
        rect = cv2.minAreaRect(cnt)
        if min(rect[1]) < 3:
            continue
        box = _unclip(cv2.boxPoints(rect).astype(np.int32), unclip_ratio)

        # 원본 좌표계로 변환  (ratio = new/orig → orig = new/ratio)
        box[:, 0] = np.clip((box[:, 0] / ratio_w).astype(np.int32), 0, orig_w - 1)
        box[:, 1] = np.clip((box[:, 1] / ratio_h).astype(np.int32), 0, orig_h - 1)
        boxes.append(box)
        scores.append(score)
    return boxes, scores

# ── 원근 변환 크롭 ────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_by_box(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    pts = _order_points(box.astype(np.float32))
    tl, tr, br, bl = pts
    w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)
    h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ── CTC 디코딩 ────────────────────────────────────────────────────────────────

def ctc_decode(
    probs: np.ndarray,
    chars: list[str],
    blank_idx: int,
) -> tuple[str, float]:
    """그리디 CTC 디코딩.

    Args:
        probs     : [T, num_classes]
        chars     : 문자 리스트 (blank 미포함)
        blank_idx : blank 클래스 인덱스 (= len(chars))
    """
    indices = probs.argmax(axis=-1)
    scores  = probs.max(axis=-1)

    result, confs, prev = [], [], -1
    for idx, sc in zip(indices, scores):
        if idx == prev or idx == blank_idx:
            prev = idx
            continue
        if idx < len(chars) and chars[idx] is not None:
            result.append(chars[idx])
            confs.append(float(sc))
        prev = idx

    return "".join(result), float(np.mean(confs)) if confs else 0.0

# ── 결과 ─────────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    input_path: str
    dt_polys:  list = field(default_factory=list)
    dt_scores: list = field(default_factory=list)
    rec_text:  list = field(default_factory=list)
    rec_score: list = field(default_factory=list)

    def print(self) -> None:
        print(f"input_path: {self.input_path}")
        for i, (poly, dt, text, rec) in enumerate(
            zip(self.dt_polys, self.dt_scores, self.rec_text, self.rec_score)
        ):
            print(f"  [{i:3d}] det={dt:.2f}  rec={rec:.2f}  '{text}'")
            print(f"         box={poly.tolist()}")

# ── 메인 클래스 ───────────────────────────────────────────────────────────────

class PaddleOCR:
    """
    ONNX 기반 한국어 OCR — PP-OCRv5 풀 파이프라인.

    Usage:
        model = PaddleOCR(
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            device="cpu",
        )
        for res in model.predict("image.jpg"):
            res.print()
    """

    _DOC_ANGLES = [0, 90, 180, 270]

    def __init__(
        self,
        text_recognition_model_name: str   = "korean_PP-OCRv5_mobile_rec",  # noqa
        use_doc_orientation_classify: bool  = True,
        use_doc_unwarping: bool             = True,    # noqa (미구현)
        use_textline_orientation: bool      = True,
        device: str                         = "cpu",   # noqa (ONNX는 CPU)
    ):
        # PaddleOCR API 호환 파라미터 (ONNX에서는 사용하지 않음)
        _ = text_recognition_model_name, use_doc_unwarping, device

        def _load(path: Path, tag: str) -> ort.InferenceSession | None:
            if not path.exists():
                print(f"[경고] {tag} 모델 없음 → 해당 단계 건너뜀")
                return None
            sess = ort.InferenceSession(str(path))
            inp  = sess.get_inputs()[0]
            print(f"[로드] {tag:<15}  입력: {inp.name} {inp.shape}")
            return sess

        self._rec_sess         = _load(REC_ONNX, "rec")
        self._det_sess         = _load(DET_ONNX, "det")
        self._doc_ori_sess     = _load(DOC_ORI_ONNX, "doc_ori")     if use_doc_orientation_classify else None
        self._textline_sess    = _load(TEXTLINE_ORI_ONNX, "textline_ori") if use_textline_orientation    else None

        if self._rec_sess is None:
            raise FileNotFoundError(
                f"rec 모델 없음: {REC_ONNX}\n"
                "  python convert_all_models.py 를 먼저 실행하세요."
            )

        self._chars, self._blank = _load_char_list()

        # rec 모델 출력 클래스 수 검증
        out_shape   = self._rec_sess.get_outputs()[0].shape
        num_classes = out_shape[-1] if len(out_shape) == 3 else None
        expected    = len(self._chars)    # blank(1) + yml(11945) + space(1) = 11947
        if num_classes and num_classes != expected:
            print(
                f"[경고] rec 클래스 불일치 — 모델: {num_classes}, 기대: {expected}\n"
                "       inference.yml 의 character_dict 를 확인하세요."
            )
        else:
            print(f"[검증] rec 클래스 수 일치: {num_classes}")

        # rec 입력 최대 너비
        inp_shape       = self._rec_sess.get_inputs()[0].shape
        self._rec_max_w = inp_shape[3] if isinstance(inp_shape[3], int) and inp_shape[3] > 0 else 320
        self._rec_name  = self._rec_sess.get_inputs()[0].name

    # ── 각 단계 ──────────────────────────────────────────────────────────────

    def _correct_doc_orientation(self, img: np.ndarray) -> np.ndarray:
        if self._doc_ori_sess is None:
            return img
        tensor  = preprocess_doc_ori(img)
        out     = self._doc_ori_sess.run(None, {self._doc_ori_sess.get_inputs()[0].name: tensor})
        cls     = int(np.argmax(out[0]))
        angle   = self._DOC_ANGLES[cls]
        if angle == 0:
            return img
        rot = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
        return cv2.rotate(img, rot[angle])

    def _detect(self, img: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        if self._det_sess is None:
            h, w = img.shape[:2]
            return [np.array([[0, 0], [w, 0], [w, h], [0, h]], np.int32)], [1.0]

        h, w          = img.shape[:2]
        tensor, rh, rw = preprocess_det(img)
        out            = self._det_sess.run(None, {self._det_sess.get_inputs()[0].name: tensor})
        boxes, scores  = db_postprocess(out[0], rh, rw, h, w)
        print(f"  [det] shape={out[0].shape}  max={out[0].max():.3f}  boxes={len(boxes)}")
        return boxes, scores

    def _correct_textline_orientation(self, crop: np.ndarray) -> np.ndarray:
        if self._textline_sess is None:
            return crop
        tensor = preprocess_textline_ori(crop)
        out    = self._textline_sess.run(None, {self._textline_sess.get_inputs()[0].name: tensor})
        if int(np.argmax(out[0])) == 1:   # 180° 뒤집힘
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        return crop

    def _recognize(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0
        tensor = preprocess_rec(crop, self._rec_max_w)
        out    = self._rec_sess.run(None, {self._rec_name: tensor})
        return ctc_decode(out[0][0], self._chars, self._blank)

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def predict(self, image: "str | Path | np.ndarray") -> list[OCRResult]:
        if isinstance(image, (str, Path)):
            img_path = str(image)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image}")
        else:
            img_path, img = "<ndarray>", image.copy()

        img = self._correct_doc_orientation(img)
        boxes, dt_scores = self._detect(img)

        valid_boxes, valid_dt, rec_texts, rec_scores = [], [], [], []
        for box, dt_sc in zip(boxes, dt_scores):
            crop = crop_by_box(img, box)
            if crop.size == 0:
                continue
            crop         = self._correct_textline_orientation(crop)
            text, rec_sc = self._recognize(crop)
            if not text:
                continue
            valid_boxes.append(box)
            valid_dt.append(dt_sc)
            rec_texts.append(text)
            rec_scores.append(rec_sc)

        return [OCRResult(img_path, valid_boxes, valid_dt, rec_texts, rec_scores)]

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="한국어 OCR — ONNX 풀 파이프라인")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--no-doc-ori",      action="store_true")
    parser.add_argument("--no-textline-ori", action="store_true")
    args = parser.parse_args()

    model = PaddleOCR(
        use_doc_orientation_classify=not args.no_doc_ori,
        use_textline_orientation=not args.no_textline_ori,
    )
    print()
    for p in args.images:
        try:
            for res in model.predict(p):
                res.print()
        except FileNotFoundError as e:
            print(f"[오류] {e}")
        print()

if __name__ == "__main__":
    main()
