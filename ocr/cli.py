'''
Author        Jiahui Chen 1946847867@qq.com
Date          2025-11-19 15:31:00
LastEditTime  2025-11-19 16:05:26
Description   

'''
"""命令行版式识别工具（默认使用 PaddleOCR）。"""
import json
import argparse
import sys
from pathlib import Path
from ocr.paddle_pipeline import PaddleLayoutPipeline
from ocr import pipeline

# 保证可以在仓库根目录直接运行 python ocr/cli.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="基于 PaddleOCR 的版式识别")
    parser.add_argument("image", help="要识别的图片路径")
    parser.add_argument("--paddle-lang", default="chinese_cht",
                        help="PaddleOCR 的语言代码，如 chinese_cht/ch/chinese_cht")
    parser.add_argument("--paddle-gpu", action="store_true",
                        help="若安装了 paddlepaddle-gpu，可指定该参数启用 GPU")
    parser.add_argument("--paddle-use-angle", action="store_true",
                        help="需要启用文字方向分类时使用")
    return parser.parse_args()


def run_with_paddle(args):
    kwargs = {
        "lang": args.paddle_lang,
        "use_angle_cls": args.paddle_use_angle,
    }
    if args.paddle_gpu:
        kwargs["device"] = "gpu"
    else:
        kwargs.setdefault("device", "cpu")
    analyzer = PaddleLayoutPipeline(ocr_kwargs=kwargs)
    return analyzer.analyze_image(args.image)


def main():
    args = parse_args()
    if (args.paddle_lang == "chinese_cht" and not args.paddle_gpu
            and not args.paddle_use_angle):
        # 直接复用默认缓存的管线，避免重复初始化 PaddleOCR
        result = pipeline.analyze_image(args.image)
    else:
        result = run_with_paddle(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
