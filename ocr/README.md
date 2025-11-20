<!--
 * @Author        Jiahui Chen 1946847867@qq.com
 * @Date          2025-11-19 15:32:18
 * @LastEditTime  2025-11-19 17:42:45
 * @Description   
 * 
-->
# recognition

基于 **PaddleOCR** 的版式识别试验场。当前内容包含：

- `pipeline.py`：默认入口，内部复用 `PaddleLayoutPipeline`。
- `paddle_pipeline.py`：核心逻辑，先运行 PaddleOCR，再根据检测框推断列数、行数、字数及版框。
- `cli.py`：命令行包装，可快速对单张图片进行分析。
- `ocr.py`：直接调用 PaddleOCR 原始 `predict` API 的示例脚本。

## 运行示例

```bash
python ocr/cli.py data/史记_1_100393_0065_b2425e.jpg

# 指定语言或启用 GPU
python ocr/cli.py data/史记_1_100393_0065_b2425e.jpg \
    --paddle-lang chinese_cht --paddle-gpu
```

输出说明：

- `num_columns`、`lines_per_column`：列数与每列的文本行数。
- `chars_per_line_median`：基于 OCR 文本长度估计的行均字数（中位数）。
- `border_color` 与 `borders`：白口/黑口以及四周单/双边框检测结果。
- `small_font`、`double_small_lines`：根据行高与间距给出的启发式判断。
- `columns_detail`：每列的原始识别文本与检测框，可用于调试。

## 进一步定制

- 若需要调整列/行划分策略，可直接修改 `paddle_pipeline.py` 中的 `_cluster_columns` 或双行检测阈值。
- 若想复用 OCR 结果用于别的任务，可查看 `columns_detail`，其中包含每一行的原始文本与四边形坐标。
