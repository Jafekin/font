"""简单示例：直接调用 PaddleLayoutPipeline 进行识别。"""
import argparse
import json

from ocr.paddle_pipeline import PaddleLayoutPipeline


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 版式识别示例")
    parser.add_argument("image", nargs="?", default="data/史记_1_100393_0065_b2425e.jpg",
                        help="需要识别的图片路径")
    parser.add_argument("--lang", default="chinese_cht", help="PaddleOCR 语言代码")
    parser.add_argument("--gpu", action="store_true", help="启用 GPU 推理")
    args = parser.parse_args()

    kwargs = {"lang": args.lang, "use_angle_cls": False}
    if args.gpu:
        kwargs["use_gpu"] = True

    analyzer = PaddleLayoutPipeline(ocr_kwargs=kwargs)
    result = analyzer.analyze_image(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
