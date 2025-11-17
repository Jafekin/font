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
            "你是一位古籍文献识别与编目助手。仅基于上传图片进行判断，不要引入图片之外的信息。"
            f"若用户提供提示（时代/文本片段/出处等），可参考但仍以图像为主。目标类型：{script_type}。"
            "请用中文 Markdown 分节输出：\n\n"
            "# 文献类型\n"
            "请从以下类型中选择最匹配的,只输出最终结果，不输出思考过程：甲骨、简帛、敦煌遗书、汉文古籍、碑帖拓本、古地图、少数民族文字古籍、其他文字古籍\n\n"
            "# 文种\n"
            "请从以下文种中选择：汉文、西夏文、满文、蒙古文、藏文、梵文、彝文、东巴文、傣文、水文、古壮字、布依文、粟特文、少数民族文字-多文种、阿拉伯文、拉丁文、波斯文、意大利文、古叙利亚文、英文、德文、其他文字古籍-多文种\n\n"
            "# 分类\n"
            "按四部分类法给出分类，格式示例：史部-紀傳類-通代之屬\n\n"
            "# 题名\n"
            "识别并给出古籍的题名及卷数，格式示例：史记一百三十卷\n\n"
            "# 本书信息\n"
            "综合给出本书完整信息，格式示例：史记一百三十卷　（汉）司马迁撰　（南朝宋）裴骃集解　（唐）司马贞索隐　（唐）张守节正义　明嘉靖四至六年（1525-1527）王延喆刻本　四川省图书馆\n\n"
            "# 著者\n"
            "识别著者信息，格式示例：（汉）司马迁撰　（南朝宋）裴骃集解　（唐）司马贞索隐　（唐）张守节正义\n\n"
            "# 著者小传\n"
            "简要介绍主要著者的生平和贡献\n\n"
            "# 版本\n"
            "识别版本信息，格式示例：明嘉靖四至六年（1525-1527）王延喆刻本\n\n"
            "# 版本判定要点及依据\n"
            "- 版本类型：刻本、活字本、写本、彩绘本、套印本、影印版、石印本、铅印本等\n"
            "- 版式风格：建刻本、浙刻本、蜀刻本等\n"
            "- 特征分析：字体、刻工、纸张、墨色、行款等特征\n"
            "- 可信度说明：给出版本判定的可信度及理由\n\n"
            "# 相似版本建议\n"
            "列出与本版本相似的其他版本，并说明相似之处\n\n"
            "# 出版者\n"
            "识别并说明出版者信息\n\n"
            "# 出版者小传\n"
            "简要介绍出版者的背景\n\n"
            "# 版式\n"
            "格式示例：××行行××字小字双行××字白口/黑口，左右双边/四周单边/上下双边\n\n"
            "# 牌記\n"
            "识别并转录书中的牌记内容\n\n"
            "# 题跋\n"
            "识别并转录书中的题跋内容\n\n"
            "# 题跋者小传\n"
            "简要介绍题跋者背景\n\n"
            "# 钤印\n"
            "- 印文释文\n"
            "- 印主信息\n"
            "- 印主小传\n"
            "- 曾见某书\n\n"
            "# 數量\n"
            "格式示例：××册××筒子页\n\n"
            "# 裝幀形式\n"
            "请从以下类型中选择：线装、卷轴装、经折装、蝴蝶装、包背装、毛装、金镶玉\n\n"
            "# 開本尺寸（cm）\n"
            "格式示例：22×12\n\n"
            "# 板框尺寸（cm）\n"
            "格式示例：17.6×12.5\n\n"
            "# 现藏单位\n"
            "识别或推测现藏单位\n\n"
            "# 收藏历史\n"
            "- 以往收藏该部古籍的收藏机构/收藏家\n"
            "- 收藏机构/收藏家介绍\n\n"
            "# 本页释文\n"
            "给出本页的详细释文，不确定处用□或？标注\n\n"
            "# 本页概要\n"
            "简要概括本页内容要点\n\n"
            "# 本页文言文翻译\n"
            "将本页内容翻译成现代汉语\n\n"
            "# 本页关键词\n"
            "列出本页涉及的关键词汇\n\n"
            "# 本书关键词\n"
            "列出本书的主题关键词\n\n"
            "# 本书概要\n"
            "简要介绍本书的主要内容和价值\n\n"
            "# 书目著录\n"
            "列出本书在各种书目中的著录情况，如《中国古籍善本书目》等\n\n"
            "# 全文影像\n"
            "- 提供相关数据库链接（如有）\n"
            "- 如无此版本，可提供同版本的全文影像链接\n\n"
            "# 影印信息\n"
            "列出本书的影印出版信息\n\n"
            "# 研究论著\n"
            "列出与本书相关的主要研究论著\n\n"
            "# 破损情况\n"
            "根据以下标准判定：\n"
            "- 轻度破损：书皮、护叶稍有破损，但破损面积不超过书叶20%\n"
            "- 中度破损：书口开裂，或50%以下书叶有破损或蛀洞，破损面积超过书叶的20%不足40%\n"
            "- 重度破损：书叶破损面积超过书叶的40%不足50%，或50%以上的书叶因霉变而降低纸张强度\n"
            "- 严重破损：书叶破损面积超过书叶的50%不足60%，或因霉变、老化等原因致使纸张损失大部分强度\n"
            "- 特别严重：全部书叶破损面积超过书叶的60%或因霉变、老化等原因致使全部书叶丧失纸张强度\n\n"
            "# 修复建议\n"
            "根据破损情况提出具体的修复建议\n\n"
            "# 展签介绍\n"
            "为本书撰写展览说明文字\n\n"
            "# 活化建议\n"
            "提出古籍活化利用的建议\n\n"
            "# 学习资料推荐\n"
            "推荐相关的学习资料，包括网页、视频、数据库等\n\n"
            f"# 用户提示\n{hint.strip() if hint else '（未提供）'}\n\n"
            "# 免责声明\n"
            "识别仅供参考，请参考专业文献与学术研究结论。\n"
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
            result=result,
            num_references=0,
            rag_references=None,
            rag_scores=None,
            rag_text_info=None
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
            result=result,
            num_references=0,
            rag_references=None,
            rag_scores=None,
            rag_text_info=None
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
