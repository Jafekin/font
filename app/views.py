"""Views for ancient script recognition app with end-to-end RAG integration."""
import base64
import json
import logging
import os
import tempfile
from datetime import datetime

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image

from .models import ScriptAnalysis
from rag.pipeline import RAGPipeline, analyze_with_llm

logger = logging.getLogger(__name__)

_RAG_PIPELINE: RAGPipeline | None = None


def _get_rag_pipeline() -> RAGPipeline:
    """Lazy-load a shared RAG pipeline instance."""
    global _RAG_PIPELINE
    if _RAG_PIPELINE is None:
        index_path = getattr(settings, 'RAG_INDEX_PATH', None)
        if not index_path:
            index_path = os.path.join(settings.BASE_DIR, 'rag', 'index')
        logger.info("Initializing RAG pipeline with index %s", index_path)
        _RAG_PIPELINE = RAGPipeline(index_path=index_path)
    return _RAG_PIPELINE


def _persist_uploaded_file(uploaded_file) -> str:
    """Write an uploaded file to a temporary location and rewind the stream."""
    suffix = os.path.splitext(getattr(uploaded_file, 'name', 'upload.png'))[
        1] or '.png'
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in uploaded_file.chunks():
        temp_file.write(chunk)
    temp_file.flush()
    temp_file.close()
    uploaded_file.seek(0)
    logger.debug("Persisted uploaded file %s to temp path %s",
                 getattr(uploaded_file, 'name', 'upload'), temp_file.name)
    return temp_file.name


def _persist_bytes(image_bytes: bytes, suffix: str = '.png') -> str:
    """Write raw image bytes to a temporary file and return the path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(image_bytes)
    temp_file.flush()
    temp_file.close()
    logger.debug("Persisted raw bytes to temp path %s", temp_file.name)
    return temp_file.name


def _cleanup_temp_file(path: str) -> None:
    """Best-effort removal of temporary files."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Failed to remove temp file %s",
                           path, exc_info=True)


def _run_rag_pipeline(image_path: str, script_type: str, hint: str):
    pipeline = _get_rag_pipeline()
    logger.info("Running RAG pipeline for image=%s script_type=%s", image_path,
                script_type)
    return pipeline.run(image_path=image_path, script_type=script_type, hint=hint)


def _ensure_result_text(text: str | None) -> str:
    return text or "无法识别古文字内容"


def _prepare_rag_response(rag_payload: dict, image_path: str, script_type: str, hint: str):
    """Normalize pipeline response and optionally fall back to direct LLM."""
    rag_success = rag_payload.get('success', False)
    result_text = rag_payload.get('analysis') if rag_success else None
    fallback_used = False

    if not rag_success:
        logger.warning(
            "RAG pipeline failed, falling back to direct LLM: %s", rag_payload.get('error'))
        try:
            with Image.open(image_path) as fallback_image:
                result_text = analyze_with_llm(
                    fallback_image, script_type, hint)
            fallback_used = True
        except Exception:
            logger.exception("Fallback LLM call failed")

    rag_meta = {
        'success': rag_success,
        'error': rag_payload.get('error'),
        'references': rag_payload.get('retrieved_references') or [],
        'scores': rag_payload.get('retrieval_scores') or [],
        'text_info': rag_payload.get('retrieved_text_info') or [],
        'citations': rag_payload.get('citations') or [],
        'num_references': rag_payload.get('num_references', 0) if rag_success else 0,
        'pipeline_mode': rag_payload.get('pipeline_mode'),
        'fallback_used': fallback_used,
    }

    return _ensure_result_text(result_text), rag_meta


def _serialize_for_storage(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def index(request):
    """Render the main page."""
    return render(request, 'index.html')


@require_http_methods(["POST"])
@csrf_exempt
def analyze(request):
    """API endpoint for analyzing ancient script."""
    try:
        # Get form data
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': '请上传图片'
            }, status=400)

        image_file = request.FILES['image']
        script_type = request.POST.get('script_type', '甲骨文')
        hint = request.POST.get('hint', '')

        logger.info("Received form analysis request script_type=%s filename=%s",
                    script_type, getattr(image_file, 'name', 'upload'))

        temp_path = _persist_uploaded_file(image_file)
        try:
            rag_payload = _run_rag_pipeline(temp_path, script_type, hint)
            result_text, rag_meta = _prepare_rag_response(
                rag_payload, temp_path, script_type, hint)
        finally:
            _cleanup_temp_file(temp_path)

        analysis = ScriptAnalysis.objects.create(
            image=image_file,
            script_type=script_type,
            hint=hint,
            result=result_text,
            num_references=rag_meta['num_references'],
            rag_references=_serialize_for_storage(rag_meta['references']),
            rag_scores=_serialize_for_storage(rag_meta['scores']),
            rag_text_info=_serialize_for_storage(rag_meta['text_info'])
        )

        return JsonResponse({
            'success': True,
            'result': result_text,
            'analysis_id': analysis.id,
            'rag': rag_meta
        })
        logger.info("Analysis %s completed via form request (fallback=%s)",
                    analysis.id, rag_meta['fallback_used'])

    except Exception as e:
        logger.exception("Form analysis failed")
        return JsonResponse({
            'success': False,
            'error': f'分析失败: {str(e)}'
        }, status=500)


@require_http_methods(["POST"])
@csrf_exempt
def analyze_base64(request):
    """API endpoint for analyzing with base64 image."""
    try:
        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse({
                'success': False,
                'error': '请提供图片'
            }, status=400)

        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        script_type = data.get('script_type', '甲骨文')
        hint = data.get('hint', '')

        logger.info(
            "Received base64 analysis request script_type=%s", script_type)

        temp_path = _persist_bytes(image_bytes)
        try:
            rag_payload = _run_rag_pipeline(temp_path, script_type, hint)
            result_text, rag_meta = _prepare_rag_response(
                rag_payload, temp_path, script_type, hint)
        finally:
            _cleanup_temp_file(temp_path)

        filename = data.get(
            'filename') or f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
        image_file = ContentFile(image_bytes, name=filename)

        analysis = ScriptAnalysis.objects.create(
            image=image_file,
            script_type=script_type,
            hint=hint,
            result=result_text,
            num_references=rag_meta['num_references'],
            rag_references=_serialize_for_storage(rag_meta['references']),
            rag_scores=_serialize_for_storage(rag_meta['scores']),
            rag_text_info=_serialize_for_storage(rag_meta['text_info'])
        )

        return JsonResponse({
            'success': True,
            'result': result_text,
            'analysis_id': analysis.id,
            'rag': rag_meta
        })
        logger.info("Analysis %s completed via base64 request (fallback=%s)",
                    analysis.id, rag_meta['fallback_used'])

    except Exception as e:
        logger.exception("Base64 analysis failed")
        return JsonResponse({
            'success': False,
            'error': f'分析失败: {str(e)}'
        }, status=500)


def history(request):
    """Get analysis history."""
    analyses = ScriptAnalysis.objects.all()[:20]

    data = []
    for analysis in analyses:
        data.append({
            'id': analysis.id,
            'script_type': analysis.script_type,
            'hint': analysis.hint,
            'result': analysis.result[:200] + '...' if len(analysis.result) > 200 else analysis.result,
            'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    return JsonResponse({'success': True, 'data': data})
