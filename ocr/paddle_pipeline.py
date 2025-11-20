"""基于 PaddleOCR 的版式识别流程。

相比纯启发式版本，本模块先调用 PaddleOCR 得到文本行的检测框，
再根据这些检测框推断列数、行数、每行字数等布局信息。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

try:  # 可选依赖，只有在实际使用 PaddleOCR 时才需要安装
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - 仅在未安装 PaddleOCR 时执行
    PaddleOCR = None  # type: ignore


def read_image(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path)
    return img


def is_background_light(gray: np.ndarray, margin_ratio: float = 0.03) -> bool:
    h, w = gray.shape
    m = int(min(h, w) * margin_ratio)
    samples = [
        gray[0:m, 0:m],
        gray[0:m, w - m:w],
        gray[h - m:h, 0:m],
        gray[h - m:h, w - m:w],
    ]
    mean = np.mean([np.mean(s) for s in samples if s.size > 0])
    return mean > 127


def binarize_image(gray: np.ndarray) -> np.ndarray:
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        15,
    )
    if np.mean(gray) > 127:
        th = 255 - th
    return th


def detect_border_lines(bin_img: np.ndarray, min_length_ratio: float = 0.5) -> Dict[str, int]:
    h, w = bin_img.shape
    results = {"left": 0, "right": 0, "top": 0, "bottom": 0}

    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(3, h // 100)))
    hor_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, w // 100), 1))

    vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel)
    hor = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, hor_kernel)

    cnts_v, _ = cv2.findContours(
        vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_lines = []
    for c in cnts_v:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh >= h * min_length_ratio:
            v_lines.append((x, y, ww, hh))

    cnts_h, _ = cv2.findContours(
        hor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_lines = []
    for c in cnts_h:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww >= w * min_length_ratio:
            h_lines.append((x, y, ww, hh))

    edge_thresh_x = int(w * 0.06)
    edge_thresh_y = int(h * 0.06)

    left_lines = [r for r in v_lines if r[0] <= edge_thresh_x]
    right_lines = [r for r in v_lines if (r[0] + r[2]) >= (w - edge_thresh_x)]
    top_lines = [r for r in h_lines if r[1] <= edge_thresh_y]
    bottom_lines = [r for r in h_lines if (r[1] + r[3]) >= (h - edge_thresh_y)]

    results["left"] = min(2, len(left_lines))
    results["right"] = min(2, len(right_lines))
    results["top"] = min(2, len(top_lines))
    results["bottom"] = min(2, len(bottom_lines))

    return results


@dataclass
class LineItem:
    text: str
    score: float
    box: np.ndarray  # (4,2) 四边形
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: float
    height: float
    x_center: float
    y_center: float

    @property
    def char_count(self) -> int:
        return max(0, len(self.text.replace(" ", "")))


def _entry_to_line_item(entry) -> Optional[LineItem]:
    """兼容 PaddleOCR 旧版 list 输出与新版 dict 输出。"""
    if isinstance(entry, dict):
        box = np.array(entry.get("box") or entry.get("points"))
        if box.size == 0:
            return None
        text = entry.get("text", "")
        score = float(entry.get("score", 1.0))
    else:
        box = np.array(entry[0])
        rec_part = entry[1]
        if isinstance(rec_part, (list, tuple)):
            text = rec_part[0] if rec_part else ""
            score = float(rec_part[1]) if len(rec_part) > 1 else 1.0
        else:
            text = str(rec_part)
            score = 1.0
    xs = box[:, 0]
    ys = box[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    width = x_max - x_min
    height = y_max - y_min
    return LineItem(
        text=text,
        score=score,
        box=box,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        width=width,
        height=height,
        x_center=float(xs.mean()),
        y_center=float(ys.mean()),
    )


def _cluster_columns(items: List[LineItem], gap_ratio: float = 1.8) -> List[List[LineItem]]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda it: it.x_min)
    median_width = np.median([it.width for it in sorted_items]) or 1.0
    columns: List[List[LineItem]] = []
    current = [sorted_items[0]]
    prev = sorted_items[0]
    for item in sorted_items[1:]:
        gap = item.x_min - prev.x_min
        if gap > median_width * gap_ratio:
            columns.append(sorted(current, key=lambda it: it.y_center))
            current = [item]
        else:
            current.append(item)
        prev = item
    columns.append(sorted(current, key=lambda it: it.y_center))
    return columns


class PaddleLayoutPipeline:
    """使用 PaddleOCR 进行布局分析。"""

    def __init__(self, ocr=None, ocr_kwargs: Optional[Dict] = None):
        if ocr is not None:
            self.ocr = ocr
        else:
            if PaddleOCR is None:  # pragma: no cover - 在未安装依赖时抛错
                raise ImportError(
                    "未检测到 PaddleOCR，请先安装 paddleocr 与 paddlepaddle(-gpu)。")
            default_kwargs = {
                "lang": "chinese_cht",
                "use_angle_cls": False,
            }
            if ocr_kwargs:
                default_kwargs.update(ocr_kwargs)
            self.ocr = PaddleOCR(**default_kwargs)

    def analyze_image(self, path: str) -> Dict:
        raw_result = self.ocr.ocr(path)
        line_items: List[LineItem] = []
        for page in raw_result:
            for entry in page:
                item = _entry_to_line_item(entry)
                if item:
                    line_items.append(item)

        img = read_image(path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_img = binarize_image(gray)
        borders = detect_border_lines(bin_img)
        bg_light = is_background_light(gray)

        columns = _cluster_columns(line_items)
        lines_per_column = [len(col) for col in columns]
        chars_counts = [it.char_count for it in line_items if it.char_count]

        heights = [it.height for it in line_items if it.height > 0]
        mean_line_height = float(np.mean(heights)) if heights else None
        img_h, img_w = gray.shape

        small_font = False
        double_small_lines = False
        if mean_line_height is not None:
            if mean_line_height < img_h / 80.0:
                small_font = True
            pair_count = 0
            for col in columns:
                for first, second in zip(col, col[1:]):
                    gap = second.y_min - first.y_max
                    if gap < mean_line_height * 0.6 and max(first.height, second.height) < mean_line_height * 0.9:
                        pair_count += 1
            if pair_count > max(2, len(columns)):
                double_small_lines = True

        columns_detail = [
            {
                "line_texts": [it.text for it in col],
                "boxes": [it.box.tolist() for it in col],
            }
            for col in columns
        ]

        return {
            "engine": "paddleocr",
            "border_color": "白口" if bg_light else "黑口",
            "borders": borders,
            "num_columns": len(columns),
            "lines_per_column": lines_per_column,
            "chars_per_line_median": int(np.median(chars_counts)) if chars_counts else None,
            "small_font": small_font,
            "double_small_lines": double_small_lines,
            "columns_detail": columns_detail,
            "raw_line_count": len(line_items),
        }
