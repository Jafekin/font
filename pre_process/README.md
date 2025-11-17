# Ancient Book Image Preprocessor

一个基于 PyTorch 的古籍图片预处理系统，提供结构识别、干扰因素处理和行款信息提取。

## 功能概览

### 1. 结构识别 (Structure Detection)
- **版心识别**: 自动识别文字主要区域
- **栏线检测**: 识别竖向分栏线
- **页码定位**: 识别页码位置
- **题签检测**: 识别书页顶部题签
- **边框分类**: 识别单边框/双边框
- **文字区域**: 检测文字区域分布

### 2. 干扰因素处理 (Distortion Handling)
- **污渍检测**: 检测古籍上的污渍、墨迹等
- **模糊区域**: 识别图片中的模糊区域（拉普拉斯方差）
- **折页检测**: 识别书页折痕
- **图片清理**: 使用图像修复改善质量
- **质量评分**: 给出整体图片质量分数 (0-1)

### 3. 行款信息提取 (Line Information Extraction)
- **行数识别**: 自动计算书页有多少行文字
- **字数检测**: 估算每行有多少个字
- **行距分析**: 计算行间距的均匀度
- **字号估算**: 估算文字大小
- **排版类型**: 识别单栏/多栏排版
- **边框配置**: 分析边框结构 (单边/双边/无边)

## 系统架构

```
pre_process/
├── __init__.py                 # 包初始化
├── book_processor.py           # 主处理器（协调器）
├── structure_detector.py       # 结构检测模块
├── distortion_handler.py       # 干扰处理模块
├── line_info_extractor.py     # 行款信息提取模块
└── example_usage.py           # 使用示例
```

## 安装依赖

所需依赖已在 `requirements.txt` 中定义：

```bash
pip install -r requirements.txt
```

关键依赖：
- PyTorch >= 2.1.0
- OpenCV >= 4.5
- NumPy >= 1.20
- Pillow >= 10.0

## 快速开始

### 方式1: Python 脚本

```python
from pre_process import AncientBookProcessor

# 初始化处理器（自动检测 CUDA）
processor = AncientBookProcessor(use_cuda=True)

# 处理单个图片
result = processor.process('path/to/ancient_book.jpg')

# 访问结果
if result['success']:
    summary = result['summary']
    print(f"检测行数: {summary['layout_info']['num_lines']}")
    print(f"每行字数: {summary['layout_info']['chars_per_line']}")
    print(f"图片质量: {summary['image_quality']['overall_score']:.1%}")
```

### 方式2: 命令行工具

处理单个图片：
```bash
python pre_process/example_usage.py path/to/image.jpg --output results/
```

处理整个目录：
```bash
python pre_process/example_usage.py data/books/ --output results/
```

### 方式3: Django REST API

启动 Django 服务器后，访问以下 API：

**上传图片进行预处理:**
```bash
curl -X POST -F "image=@image.jpg" http://localhost:8000/api/preprocess
```

**Base64 图片预处理:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"data:image/png;base64,..."}' \
  http://localhost:8000/api/preprocess-base64
```

**获取处理器信息:**
```bash
curl http://localhost:8000/api/processor-info
```

## 输出结果结构

### 顶级字段

```json
{
  "success": true,
  "image_shape": [1000, 800, 3],
  "device": "cuda",
  "structures": {...},
  "distortions": {...},
  "line_info": {...},
  "summary": {...},
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Structures (结构信息)

```json
{
  "version_heart": {
    "found": true,
    "coords": {"x": 100, "y": 50, "width": 600, "height": 800},
    "confidence": 0.95
  },
  "column_lines": [
    {"type": "vertical", "x1": 300, "y1": 0, "x2": 300, "y2": 800}
  ],
  "page_numbers": [
    {"region": "bottom_right", "x": 700, "y": 900}
  ],
  "borders": {
    "top": true,
    "bottom": true,
    "left": true,
    "right": true,
    "type": "double"
  }
}
```

### Distortions (干扰信息)

```json
{
  "quality_score": 0.85,
  "stains_info": [
    {"x": 150, "y": 200, "width": 20, "height": 30, "area": 600}
  ],
  "blur_regions": [
    {"x": 400, "y": 300, "variance": 8.5, "blur_level": "high"}
  ],
  "fold_marks": [
    {"x1": 500, "y1": 0, "x2": 500, "y2": 800, "type": "vertical"}
  ]
}
```

### Line Info (行款信息)

```json
{
  "num_lines": 12,
  "chars_per_line": 16,
  "line_spacing": 65.5,
  "char_size_estimate": 48,
  "layout_type": "single_column",
  "borders": {
    "configuration": "complete_double",
    "top": {"detected": true, "type": "double", "line_count": 2}
  },
  "line_details": [
    {
      "line_id": 0,
      "y_position": 100,
      "height": 48,
      "char_count": 16,
      "confidence": 0.92
    }
  ]
}
```

### Summary (分析摘要)

```json
{
  "page_type": "single_column_text",
  "image_quality": {
    "overall_score": 0.85,
    "stain_count": 2,
    "blur_regions": 1,
    "fold_marks": 0
  },
  "layout_info": {
    "num_lines": 12,
    "chars_per_line": 16,
    "estimated_char_size": 48
  },
  "recommendations": [
    "图片质量良好，可以进行进一步的文字识别"
  ]
}
```

## 性能参数

| 指标 | CPU | GPU (CUDA) |
|-----|-----|-----------|
| 处理速度 | 1-5秒 | 0.5-2秒 |
| 内存占用 | ~300MB | ~600MB |
| 图片大小 | 自适应 | 自适应 |

## 处理流程

```
输入图片
    ↓
[图片加载和预处理]
    ↓
[结构检测] ────→ 版心、栏线、页码、题签、边框
    ↓
[干扰处理] ────→ 污渍、模糊、折页检测和修复
    ↓
[行款提取] ────→ 行数、字数、边框配置
    ↓
[生成摘要] ────→ 建议和质量评分
    ↓
输出结果 JSON
```

## 配置和调优

### 设备选择

```python
# 自动检测 CUDA
processor = AncientBookProcessor(use_cuda=True)

# 强制使用 CPU
processor = AncientBookProcessor(device='cpu')

# 强制使用 CUDA
processor = AncientBookProcessor(device='cuda')
```

### 分析深度

```python
# 完整分析（默认）
result = processor.process(image_path, full_analysis=True)

# 基础分析（更快）
result = processor.process(image_path, full_analysis=False)
```

## 参数调整

修改以下文件中的参数以调整检测灵敏度：

### structure_detector.py
- `cv2.Canny()` 参数：调整边缘检测阈值
- Hough 变换参数：调整线条检测灵敏度

### distortion_handler.py
- `blur_threshold = 10`：降低值更敏感
- 污渍大小范围：`if 20 < area < 1000:`

### line_info_extractor.py
- `threshold = 0.2`：行聚类距离
- 调整行款特征提取逻辑

## 扩展开发

### 添加自定义检测器

```python
from pre_process.structure_detector import StructureDetector

class CustomDetector(StructureDetector):
    def detect_custom_element(self, image):
        # 实现自定义检测逻辑
        pass
```

### 集成深度学习模型

```python
class NeuralNetworkDetector:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
    
    def detect(self, image):
        # 使用神经网络进行检测
        pass
```

## 故障排除

| 问题 | 原因 | 解决方案 |
|-----|------|--------|
| 导入错误 | 包不在路径中 | 添加到 PYTHONPATH |
| CUDA 内存不足 | GPU 内存不够 | 使用 CPU 或缩小图片 |
| 检测不准 | 图片质量差 | 增加图片分辨率 |
| 处理缓慢 | 未使用 GPU | 安装 CUDA，设置 use_cuda=True |
| 行数识别错误 | 文字太小或太大 | 调整参数 |

## 常见问题 (FAQ)

**Q: 支持哪些图片格式？**
A: JPG, PNG, BMP, TIFF 等 OpenCV 支持的格式

**Q: 最大支持多大的图片？**
A: 取决于可用内存，通常 50MB 以下无问题

**Q: 如何提高准确率？**
A: 确保原始图片质量好，调整相关参数，可训练专门的模型

**Q: 可以批量处理吗？**
A: 支持，参见 `example_usage.py` 的目录处理方式

**Q: 可以在生产环境使用吗？**
A: 可以，建议配合错误处理和日志系统

## 许可证

MIT License

## 更新日志

### v1.0.0 (2025-01-15)
- ✅ 初始版本发布
- ✅ 支持结构检测、干扰处理、行款提取
- ✅ Django API 集成
- ✅ CPU/GPU 支持

## 联系与支持

有问题或建议，请查看项目文档或提交 Issue。

---

**最后更新**: 2025-01-15  
**版本**: 1.0.0

