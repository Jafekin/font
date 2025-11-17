"""
RAG (Retrieval-Augmented Generation) 管道

主要功能：
- 用户上传图片后，从知识库中检索相似图片及其对应的文字信息
- 将检索到的文字信息输入给LLM
- LLM基于检索到的信息回答问题
"""
from typing import Optional, Dict, Any, List
import os
import logging
import traceback

from rag.embeddings import get_image_embedding, get_text_embedding
from rag.retriever import TxtaiRetriever
from rag.prompt import get_prompt

# 配置日志
logger = logging.getLogger(__name__)


_ANALYZE_FUNC = None


def _get_analyze_func():
    """懒加载分析函数（LLM后端），支持存根降级"""
    global _ANALYZE_FUNC
    if _ANALYZE_FUNC is not None:
        return _ANALYZE_FUNC

    if os.getenv("RAG_FAKE_EMBEDDINGS") == "1":
        def _stub(image_path, script_type, hint, prompt):
            return {
                "document_metadata": {"document_type": {"value": script_type, "confidence": 0.5}},
                "prompt_length": len(prompt),
                "used_references": []
            }
        _ANALYZE_FUNC = _stub
        return _ANALYZE_FUNC

    try:
        from app.analysis import analyze_ancient_script as real_analyze  # type: ignore
        _ANALYZE_FUNC = real_analyze
    except Exception:
        def _fallback(image_path, script_type, hint, prompt):
            return {
                "error": "analysis模块不可用",
                "document_metadata": {"document_type": {"value": script_type, "confidence": 0.0}},
                "used_references": []
            }
        _ANALYZE_FUNC = _fallback
    return _ANALYZE_FUNC


class RAGPipeline:
    """
    RAG管道，协调检索 + 提示词增强 + 下游分析

    工作流程：
    1. 用户上传图片
    2. 从知识库中检索相似图片及其对应的文字信息
    3. 将检索到的文字信息构建增强提示词
    4. 调用LLM生成分析结果
    """

    def __init__(self, index_path: str | None = None, db_path: Optional[str] = None):
        """
        初始化RAG管道

        Args:
            index_path: txtai索引路径
            db_path: 数据库路径（可选，用于未来扩展）
        """
        self.retriever = TxtaiRetriever(index_path)
        self.db_path = db_path

    # ------------------------------------------------------------------
    # 核心执行
    # ------------------------------------------------------------------
    def run(
        self,
        image_path: str,
        script_type: str,
        hint: Optional[str] = None,
        k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行完整的RAG流程

        Args:
            image_path: 用户上传的图片路径
            script_type: 文字类型（如"汉文古籍"）
            hint: 用户提示（可选）
            k: 检索top-k个相似图片
            metadata_filters: 元数据过滤条件（可选）

        Returns:
            包含分析结果和检索信息的字典：
            {
                "success": bool,
                "analysis": dict,  # LLM分析结果
                "retrieved_references": list,  # 检索到的图片ID列表
                "retrieved_text_info": list,  # 检索到的文字信息列表
                "num_references": int,
                "retrieval_scores": list,
                "citations": list,
                "pipeline_mode": str
            }
        """
        try:
            # 步骤1: 通过图片路径直接搜索相似图片
            # txtai支持直接使用图片路径进行搜索
            logger.info(f"开始检索相似图片: {image_path}, k={k}")
            retrieved_results = self.retriever.search_by_image(
                image_path,
                k=k,
                filters=metadata_filters
            )
            logger.info(f"检索完成，找到 {len(retrieved_results)} 个结果")

            # 步骤2: 提取检索到的文字信息
            retrieved_context: List[str] = []
            retrieved_text_info: List[str] = []
            used_references: List[str] = []

            for result in retrieved_results:
                # 获取图片对应的文字信息
                text_info = result.get("text_info", "")
                content = result.get("content", "")

                # 优先使用text_info，如果没有则使用content
                if text_info and text_info.strip():
                    retrieved_text_info.append(text_info)
                    retrieved_context.append(text_info)
                elif content and not content.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # content不是图片路径，可能是文字
                    retrieved_context.append(content)

                used_references.append(result.get("id", "unknown"))

            logger.info(f"提取到 {len(retrieved_context)} 条文字信息")

            # 步骤3: 构建增强提示词
            # 将检索到的文字信息作为上下文传递给LLM
            prompt = get_prompt(script_type, hint or "", retrieved_context)
            logger.info(f"提示词长度: {len(prompt)} 字符")

            # 步骤4: 调用LLM进行分析
            logger.info("开始调用LLM进行分析...")
            analyze_func = _get_analyze_func()
            
            # analyze_ancient_script需要PIL Image对象，需要从路径加载
            from PIL import Image
            try:
                image_obj = Image.open(image_path)
                logger.info(f"成功加载图片: {image_path}")
            except Exception as img_error:
                logger.error(f"加载图片失败: {img_error}")
                raise

            analysis_result = analyze_func(
                image_obj, script_type, hint or "", prompt)
            logger.info("LLM分析完成")

            # 步骤5: 生成引用（可选）
            citations = []
            if isinstance(analysis_result, dict):
                # 尝试从分析结果中提取文本用于引用
                answer_text = (
                    analysis_result.get("page_content", {}).get("page_summary")
                    or analysis_result.get("title", {}).get("title_text")
                    or str(analysis_result)
                )
                citations = self.retriever.cite(answer_text, k=min(3, k))

            return {
                "success": True,
                "analysis": analysis_result,
                "retrieved_references": used_references,
                "retrieved_text_info": retrieved_text_info,  # 新增：检索到的文字信息
                "num_references": len(retrieved_results),
                "retrieval_scores": [r.get("score", 0) for r in retrieved_results],
                "citations": citations,
                "pipeline_mode": "fake" if not self.retriever.is_ready() else "real"
            }
        except Exception as e:
            logger.error(f"RAG pipeline执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "analysis": None,
                "retrieved_references": [],
                "retrieved_text_info": []
            }

    def search_similar(
        self,
        query_image_path: Optional[str] = None,
        query_text: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        相似度搜索（仅检索，不调用LLM）

        Args:
            query_image_path: 查询图片路径
            query_text: 查询文本
            k: 返回top-k结果

        Returns:
            搜索结果列表，每个结果包含：
            - id: 文档ID
            - score: 相似度分数
            - image_path: 图片路径
            - text_info: 对应的文字信息
            - metadata: 元数据
        """
        if query_image_path:
            return self.retriever.search_by_image(query_image_path, k=k)
        if query_text:
            return self.retriever.search(query_text, k=k)
        raise ValueError("必须提供query_image_path或query_text之一")

    def batch_analyze(
        self,
        image_paths: List[str],
        script_type: str,
        hints: Optional[List[str]] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        批量RAG分析多个图片

        Args:
            image_paths: 图片路径列表
            script_type: 文字类型
            hints: 提示列表（可选，长度应与image_paths相同）
            k: 每个图片检索top-k个相似图片

        Returns:
            每个图片的分析结果列表
        """
        if not image_paths:
            raise ValueError("image_paths必须非空")
        hints = hints or [None] * len(image_paths)
        results: List[Dict[str, Any]] = []
        for image_path, hint in zip(image_paths, hints):
            results.append(self.run(image_path, script_type, hint, k))
        return results

    def cite(self, answer_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        公共引用辅助方法（供外部重用）

        Args:
            answer_text: 答案文本
            k: 返回top-k引用

        Returns:
            引用候选列表
        """
        return self.retriever.cite(answer_text, k=k)

    def get_text_info_for_image(
        self,
        image_path: str,
        k: int = 1
    ) -> List[str]:
        """
        获取指定图片对应的文字信息

        Args:
            image_path: 图片路径
            k: 返回top-k结果（通常只需要最相似的1个）

        Returns:
            文字信息列表
        """
        results = self.retriever.search_by_image(image_path, k=k)
        text_info_list = []
        for result in results:
            text_info = result.get("text_info", "")
            if text_info and text_info.strip():
                text_info_list.append(text_info)
        return text_info_list
