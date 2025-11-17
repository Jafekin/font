"""
基于txtai的检索器，支持图片和文字信息的联合检索

主要功能：
- 支持图片相似度检索
- 返回图片对应的文字信息和元数据
- 支持文本检索
- 支持元数据过滤
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional

# 允许txtai加载pickle数据（用于加载索引配置）
# 这是安全的，因为我们处理的是本地数据
if 'ALLOW_PICKLE' not in os.environ:
    os.environ['ALLOW_PICKLE'] = 'True'

# 配置日志
logger = logging.getLogger(__name__)

try:
    from txtai.embeddings import Embeddings
    TXTAI_AVAILABLE = True
except ImportError:
    TXTAI_AVAILABLE = False
    Embeddings = None  # type: ignore

# 假模式标志
_FAKE_MODE = os.getenv("RAG_FAKE_EMBEDDINGS") == "1"

# 模型名称
try:
    from rag.embeddings import MODEL_NAME
except Exception:
    MODEL_NAME = "clip-ViT-B-32"  # 默认使用CLIP模型


class TxtaiRetriever:
    """
    基于txtai的检索器，支持图片+文字信息的联合检索

    特性：
    - 支持图片相似度检索
    - 返回图片对应的文字信息和元数据
    - 支持文本检索
    - 支持元数据过滤
    """

    def __init__(self, index_path: str | None = None, *, model: str | None = None):
        """
        初始化检索器

        Args:
            index_path: txtai索引保存路径（目录）
            model: 嵌入模型路径/名称（默认使用共享的MODEL_NAME）
        """
        self.index_path = index_path or ""
        self.model = model or MODEL_NAME
        self.embeddings: Optional[Embeddings] = None
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.metadata_path = os.path.join(
            self.index_path, "metadata.json") if self.index_path else ""

        if _FAKE_MODE or not TXTAI_AVAILABLE or Embeddings is None:
            return  # 假模式

        try:
            # 初始化txtai Embeddings
            # 使用与build_index.py相同的配置，确保兼容性
            # 注意：如果索引已存在，加载时会自动使用索引中的配置
            config = {
                "method": "sentence-transformers",  # 使用sentence-transformers方法（支持CLIP）
                "path": self.model,
                "content": True,  # 保存原始内容
                "gpu": True,  # 自动选择GPU/CPU
                "format": "numpy"
            }

            # 如果提供了索引路径且存在，先尝试加载索引（会覆盖配置）
            if self.index_path and os.path.exists(self.index_path):
                try:
                    # 加载索引时会自动使用索引中保存的配置
                    self.embeddings = Embeddings()
                    self.embeddings.load(self.index_path)
                    print(f"已加载索引: {self.index_path}")
                except Exception as e:
                    print(f"警告: 加载索引失败，使用新配置: {e}")
                    # 如果加载失败，使用新配置创建
                    self.embeddings = Embeddings(config)
            else:
                # 没有索引路径或索引不存在，使用配置创建新实例
                self.embeddings = Embeddings(config)
        except Exception as e:
            print(f"警告: 初始化检索器失败: {e}")
            self.embeddings = None

        # 尝试加载元数据文件
        if self.metadata_path and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as meta_file:
                    data = json.load(meta_file)
                    if isinstance(data, dict):
                        self.metadata_store = data
                        print(f"已加载索引元数据: {self.metadata_path}")
            except Exception as meta_error:
                print(f"警告: 无法加载索引元数据: {meta_error}")

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _fake_results(k: int, with_content: bool = True, seed: int = 42) -> List[Dict[str, Any]]:
        """生成假结果（用于测试）"""
        import numpy as np
        rng = np.random.default_rng(seed)
        results = []
        for i in range(k):
            results.append({
                "id": f"fake-{i}",
                "score": float(1.0 - i * 0.05),
                "content": f"假结果内容{i}" if with_content and i % 2 == 0 else "",
                "text_info": f"假文字信息{i}",
                "image_path": f"fake_image_{i}.jpg",
                "metadata": {
                    "filename": f"fake_image_{i}.jpg",
                    "type": "image"
                }
            })
        return results

    def _wrap_search_result(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        包装txtai搜索结果，提取图片对应的文字信息

        txtai返回的格式（当content=True时）：
        {
            "id": "...",
            "score": 0.95,
            "text": "...",  # 文字内容（如果是文字文档）或图片路径（如果是图片文档）
            ...  # 其他元数据字段（直接从data字典中提取的字段）
        }

        根据build_index.py的索引格式：
        - 图片文档：data = {"object": PIL.Image, "text": text_info, ...metadata...}
        - 文字文档：data = {"text": text_info, ...metadata...}

        txtai会将data字典中的字段（除了object）作为metadata返回
        """
        # 获取基础信息
        doc_id = rec.get("id", "")

        result = {
            "id": doc_id,
            "score": rec.get("score", 0.0),
        }

        # txtai返回的text字段可能是：
        # - 对于图片文档：可能是图片路径或None（因为object字段是二进制数据）
        # - 对于文字文档：文字内容
        text_content = rec.get("text", rec.get("content", ""))
        result["content"] = text_content

        # 获取元数据
        # txtai会将data字典中的字段（除了object）直接放在结果字典中
        # 我们需要提取这些字段作为metadata
        metadata = {}
        if doc_id and doc_id in self.metadata_store:
            stored_meta = self.metadata_store.get(doc_id)
            if isinstance(stored_meta, dict):
                metadata.update(stored_meta)

        if isinstance(rec, dict):
            raw_meta = rec.get("metadata")
            if isinstance(raw_meta, dict):
                metadata.update(raw_meta)

            # 排除txtai的标准字段以及metadata本身
            excluded_keys = {"id", "score", "text", "content", "metadata"}
            for k, v in rec.items():
                if k not in excluded_keys:
                    metadata[k] = v

        result["metadata"] = metadata

        # 根据文档类型提取信息
        doc_type = metadata.get("type", "")

        if doc_type == "image":
            # 图片文档：优先使用metadata中的image_path与text_info
            image_path_val = metadata.get(
                "image_path", "") or metadata.get("filepath", "")
            text_info_val = metadata.get("text_info", "")

            image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

            # 如果text_info缺失，尝试从text_content回退（排除图片路径）
            if (not text_info_val or not text_info_val.strip()) and text_content and not text_content.lower().endswith(image_exts):
                text_info_val = text_content

            # 最后再用文件名兜底
            if (not text_info_val or not text_info_val.strip()) and metadata.get("filename"):
                text_info_val = f"图片: {metadata.get('filename')}"

            # 如果image_path看起来是占位文本（以"图片:"开头），并且id包含路径，则用id兜底
            if image_path_val.startswith("图片:") and rec.get("id"):
                image_path_val = rec.get("id")

            result["image_path"] = image_path_val
            result["text_info"] = text_info_val
        elif doc_type == "text":
            # 文字文档
            # text_info是文字内容（从data字典的text字段提取）
            result["text_info"] = text_content or metadata.get("text_info", "")
            # image_path存储在metadata中
            result["image_path"] = metadata.get("image_path", "")
            result["related_image_id"] = metadata.get("related_image_id", "")
        else:
            # 兼容旧格式或未知格式
            # 尝试从metadata中获取
            result["text_info"] = metadata.get("text_info", text_content if text_content and not text_content.endswith(
                ('.png', '.jpg', '.jpeg', '.webp')) else "")
            result["image_path"] = metadata.get("image_path", text_content if text_content and text_content.endswith(
                ('.png', '.jpg', '.jpeg', '.webp')) else "")

        return result

    # ------------------------------------------------------------------
    # 公共API
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        语义搜索（支持文本或图片路径）

        Args:
            query: 查询文本或图片路径
            k: 返回top-k结果
            filters: 元数据过滤条件（txtai支持SQL-like过滤）

        Returns:
            搜索结果列表，每个结果包含：
            - id: 文档ID
            - score: 相似度分数
            - content: 原始内容（图片路径或文字）
            - text_info: 图片对应的文字信息
            - image_path: 图片路径
            - metadata: 元数据
        """
        if not query:
            raise ValueError("query必须非空")

        if _FAKE_MODE or self.embeddings is None:
            return self._fake_results(k, seed=abs(hash(query)) % (2**32))

        try:
            # txtai的search方法支持文本和图片路径
            # 如果query是图片路径，txtai会自动处理
            logger.info(
                f"调用txtai.embeddings.search: query={query[:50] if len(query) > 50 else query}, limit={k}")
            raw_results = self.embeddings.search(query, limit=k)
            logger.info(
                f"txtai返回 {len(raw_results) if raw_results else 0} 个原始结果")

            # 包装结果
            results = [self._wrap_search_result(r) for r in raw_results]
            logger.info(f"包装后得到 {len(results)} 个结果")

            # 应用元数据过滤（如果提供）
            if filters:
                results = self._apply_filters(results, filters)
                logger.info(f"过滤后得到 {len(results)} 个结果")

            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            # 降级到假结果
            return self._fake_results(k, seed=abs(hash(query)) % (2**32))

    def search_by_vector(
        self,
        query_vector: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query_vector: 预计算的查询向量
            k: 返回top-k结果

        Returns:
            搜索结果列表
        """
        if not query_vector:
            raise ValueError("query_vector必须非空")

        if _FAKE_MODE or self.embeddings is None:
            return self._fake_results(k, seed=len(query_vector))

        try:
            raw_results = self.embeddings.search(query_vector, limit=k)
            return [self._wrap_search_result(r) for r in raw_results]
        except Exception as e:
            print(f"警告: 向量搜索失败: {e}")
            return self._fake_results(k, seed=len(query_vector))

    def search_by_image(
        self,
        image_path: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        通过图片路径搜索相似图片

        Args:
            image_path: 查询图片路径
            k: 返回top-k结果
            filters: 元数据过滤条件

        Returns:
            搜索结果列表，每个结果包含对应的文字信息
        """
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"图片不存在: {image_path}")

        logger.info(f"search_by_image: 图片路径={image_path}, k={k}")
        # 直接使用图片路径作为查询
        results = self.search(image_path, k=k, filters=filters)
        logger.info(f"search_by_image: 返回 {len(results)} 个结果")
        return results

    def batchsearch(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        批量搜索

        Args:
            queries: 查询列表（文本或图片路径）
            k: 每个查询返回top-k结果

        Returns:
            每个查询的结果列表
        """
        if not queries:
            raise ValueError("queries必须非空列表")

        results: List[List[Dict[str, Any]]] = []
        for q in queries:
            results.append(self.search(q, k=k))
        return results

    def cite(
        self,
        answer_text: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        为答案生成引用候选（将答案文本搜索回索引）

        Args:
            answer_text: 答案文本
            k: 返回top-k引用

        Returns:
            引用候选列表
        """
        if not answer_text:
            return []
        return self.search(answer_text, k=k)

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        应用元数据过滤

        Args:
            results: 搜索结果
            filters: 过滤条件

        Returns:
            过滤后的结果
        """
        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            match = True
            for key, value in filters.items():
                if metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(result)
        return filtered

    def is_ready(self) -> bool:
        """返回True如果真实的索引已加载（非假模式）"""
        return not _FAKE_MODE and TXTAI_AVAILABLE and self.embeddings is not None


# 向后兼容别名
class FaissRetriever(TxtaiRetriever):
    """已弃用的名称，保留用于向后兼容"""
    pass
