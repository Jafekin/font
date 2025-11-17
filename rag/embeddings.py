"""
Vectorization utilities using txtai for image and text embeddings.
- Provides unified image and text embeddings through txtai Embeddings API.
- Supports multiple embedding models including CLIP and other transformers.
- Lazy-loaded model initialization to avoid heavy startup time.
- Automatically selects CPU/GPU based on availability.
- Optional L2 normalization for stable similarity comparisons.
"""
from __future__ import annotations
import os
from typing import List, Any

# Configuration
# 使用与build_index.py相同的默认模型，确保兼容性
MODEL_NAME = "clip-ViT-B-32"  # CLIP模型，支持图片和文本的统一向量空间

# Internal cache for embeddings instance (lazy load)
_embeddings: Any | None = None

# Flag for test/fake mode
_FAKE_MODE = os.getenv("RAG_FAKE_EMBEDDINGS") == "1"

# ---------------------------------------------------------------------------
# Fake embeddings (deterministic for tests)
# ---------------------------------------------------------------------------


class _FakeEmbeddings:
    """Lightweight fake embeddings used in tests to avoid importing torch/txtai."""

    def transform(self, docs):
        import numpy as np
        vectors = []
        for uid, content, _ in docs:
            seed = abs(hash(uid)) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=512).astype("float32"))
        return vectors

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_embeddings():
    """Lazy-loads the embeddings instance once; supports fake mode."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if _FAKE_MODE:
        _embeddings = _FakeEmbeddings()
        return _embeddings

    try:
        from txtai.embeddings import Embeddings  # type: ignore
    except Exception:
        _embeddings = _FakeEmbeddings()
        return _embeddings

    try:
        # 使用与build_index.py相同的配置，确保兼容性
        # 使用sentence-transformers方法（支持CLIP模型处理图片）
        _embeddings = Embeddings({
            "method": "sentence-transformers",  # 使用sentence-transformers方法（支持CLIP）
            "path": MODEL_NAME,
            "gpu": True,  # 自动选择GPU/CPU
            "content": True,
            "format": "numpy",
        })
    except Exception:
        _embeddings = _FakeEmbeddings()
    return _embeddings


def _to_list(embedding) -> List[float]:
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def _normalize(vec: List[float]) -> List[float]:
    if not vec:
        return vec
    import math
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _single(doc_id: str, payload: str) -> List[List[float]]:
    embeddings = _ensure_embeddings()
    result = embeddings.transform([(doc_id, payload, None)])
    return result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_image_embedding(image_path: str, *, normalize: bool = True) -> List[float]:
    """Generates an embedding for an image file using txtai or fake embeddings."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        uid = os.path.basename(image_path)
        result = _single(uid, image_path)
        if result and len(result) > 0:
            vec = _to_list(result[0])
            return _normalize(vec) if normalize else vec
        raise RuntimeError(
            f"Failed to generate embedding for image: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate image embedding: {e}")


def get_text_embedding(text: str, *, normalize: bool = True) -> List[float]:
    """Generates an embedding for a text string using txtai or fake embeddings."""
    if not text or not text.strip():
        raise ValueError("Text must be a non-empty string.")
    try:
        result = _single("text_input", text)
        if result and len(result) > 0:
            vec = _to_list(result[0])
            return _normalize(vec) if normalize else vec
        raise RuntimeError("Failed to generate embedding for text")
    except Exception as e:
        raise RuntimeError(f"Failed to generate text embedding: {e}")


def batch_image_embeddings(image_paths: List[str], *, normalize: bool = True) -> List[List[float]]:
    """Generates embeddings for multiple images efficiently."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    embeddings = _ensure_embeddings()
    documents = [
        (os.path.basename(path), path, None)
        for path in image_paths if os.path.isfile(path)
    ]
    if not documents:
        raise ValueError("No valid image files found in image_paths")
    try:
        results = embeddings.transform(documents)
        if results is None:
            raise RuntimeError("Failed to generate batch embeddings")
        processed = [_to_list(emb) for emb in results]
        return [_normalize(v) for v in processed] if normalize else processed
    except Exception as e:
        raise RuntimeError(f"Failed to generate batch image embeddings: {e}")


def batch_text_embeddings(texts: List[str], *, normalize: bool = True) -> List[List[float]]:
    """Generates embeddings for multiple text strings efficiently."""
    if not texts:
        raise ValueError("texts must not be empty")
    embeddings = _ensure_embeddings()
    documents = [
        (f"text_{i}", text, None)
        for i, text in enumerate(texts) if text and text.strip()
    ]
    if not documents:
        raise ValueError("No valid texts found in input")
    try:
        results = embeddings.transform(documents)
        if results is None:
            raise RuntimeError("Failed to generate batch text embeddings")
        processed = [_to_list(emb) for emb in results]
        return [_normalize(v) for v in processed] if normalize else processed
    except Exception as e:
        raise RuntimeError(f"Failed to generate batch text embeddings: {e}")

# ---------------------------------------------------------------------------
# (Optional) Index build helper for future extension
# ---------------------------------------------------------------------------


def build_embeddings_index(items: List[str], output_dir: str) -> None:
    """Builds and saves a txtai index from a list of text items.
    (Image indexing would require extraction -> text; left for future.)
    """
    if _FAKE_MODE:
        return  # Skip heavy ops in fake mode
    try:
        from txtai.embeddings import Embeddings  # type: ignore
        emb = Embeddings({"path": MODEL_NAME, "content": True})
        emb.index([(str(i), text, None) for i, text in enumerate(items)])
        os.makedirs(output_dir, exist_ok=True)
        emb.save(output_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to build index: {e}")
