"""Views for ancient script recognition app."""
import base64
import io
import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PIL import Image

from .models import ScriptAnalysis

# Lazy import and initialization of OpenAI client
_client = None


def get_openai_client():
    """Get or create OpenAI client - lazy initialization."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI

            # Initialize client with only required parameters
            _client = OpenAI(
                api_key=settings.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY'),
                base_url=settings.OPENAI_BASE_URL or os.getenv(
                    'OPENAI_BASE_URL'),
            )
        except ImportError:
            raise ImportError(
                "OpenAI library is not installed. Please install it with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    return _client


def analyze_ancient_script(image, script_type: str = "甲骨文", hint: str = ""):
    """
    Analyze ancient script from image.
    Returns markdown formatted analysis result.
    """
    try:
        # Convert PIL image to base64
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            image_base64 = image

        prompt_text = (
            "你是一位古文字识别与释读助手。仅基于上传图片进行判断，不要引入图片之外的信息。"
            f"若用户提供提示（时代/文本片段/出处等），可参考但仍以图像为主。目标类型：{script_type}。"
            "请用中文 Markdown 分节输出：\n\n"
            "# 初步判读\n"
            "- 可能的文字体系（甲骨文/金文/篆书/隶书/敦煌文书等）与时代（可推测）。\n"
            "- 书写方向与版式（自右向左/自上而下、行列、行款）。\n\n"
            "# 字形要点与识别依据\n"
            "- 关键字形笔画/构件（如方形、人形、手形、卜辞结构、章草痕迹）。\n"
            "- 版面特征（行距、墨色、刻痕/纸纤维、装帧/残缺）。\n\n"
            "# 初步释文（不确定用□或？标注）\n"
            "- 给出尝试识读的分行文本，尽量标注可能异体或残缺。\n"
            "- 提供现代汉语释读或意译（若可）。\n\n"
            "# 字词考释与参照\n"
            "- 列出 Top 5 关键字词的可能读法/近似字，附依据。\n"
            "- 推荐参考书目或检索关键词（中文/英文）。\n\n"
            "# 进一步工作建议\n"
            "- 拍摄建议（光线/角度/近景/分区）。\n"
            "- 图像增强与去噪方向。\n\n"
            f"# 用户提示\n- {hint.strip() if hint else '（未提供）'}\n\n"
            "# 免责声明\n"
            "- 识别仅供参考，请参考专业文献与学术研究结论。\n"
        )

        client = get_openai_client()
        completion = client.chat.completions.create(
            model="ernie-4.5-turbo-vl",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            stream=False,
        )

        return completion.choices[0].message.content or "无法识别古文字内容"
    except Exception as e:
        return f"分析失败: {str(e)}"


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

        # Open image
        image = Image.open(image_file)

        # Analyze
        result = analyze_ancient_script(image, script_type, hint)

        # Save to database
        analysis = ScriptAnalysis.objects.create(
            image=image_file,
            script_type=script_type,
            hint=hint,
            result=result
        )

        return JsonResponse({
            'success': True,
            'result': result,
            'analysis_id': analysis.id
        })

    except Exception as e:
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
        image = Image.open(io.BytesIO(image_bytes))

        script_type = data.get('script_type', '甲骨文')
        hint = data.get('hint', '')

        # Analyze
        result = analyze_ancient_script(image, script_type, hint)

        # Save to database
        analysis = ScriptAnalysis.objects.create(
            image=None,  # Will save only text result
            script_type=script_type,
            hint=hint,
            result=result
        )

        return JsonResponse({
            'success': True,
            'result': result,
            'analysis_id': analysis.id
        })

    except Exception as e:
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
