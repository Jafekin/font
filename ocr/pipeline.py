"""基于 PaddleOCR 的默认版式分析入口。"""
from __future__ import annotations

from functools import lru_cache

from ocr.paddle_pipeline import PaddleLayoutPipeline


@lru_cache(maxsize=1)
def _default_pipeline() -> PaddleLayoutPipeline:
    return PaddleLayoutPipeline()


def analyze_image(path: str):
    """兼容旧接口，返回 PaddleOCR 管线的分析结果。"""
    return _default_pipeline().analyze_image(path)
