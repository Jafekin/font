# 古文字识别助手：关键流程示意

> 说明：本文件使用 Mermaid 流程图描述项目中的若干关键步骤，可在 VS Code / GitHub / mermaid.live 中预览或导出为 PNG/SVG，用于论文或报告配图。

---

## 1. 古籍影像总体处理流程

```mermaid
flowchart LR
    A[古籍影像采集/上传] --> B[文件接收与存储\nDjango: app/views.py]
    B --> C[版式与OCR预处理\nocr/paddle_pipeline.py]
    C --> D[特征提取\n几何/纹理/版面/钤印]
    D --> E[向量编码\nrag/embeddings.py]
    E --> F[样本库与索引构建\nembeddings.npy + metadata.json]
    F --> G[相似检索与聚类\nrag/retriever.py]
    G --> H[RAG Prompt 组装\nrag/prompt.py]
    H --> I[大模型分析与结构化输出\nrag/pipeline.py]
    I --> J[结果存储与展示\n分析报告、可视化]
```

---

## 2. 特征提取与破损候选区域检测流程

```mermaid
flowchart LR
    A[输入古籍页面图像] --> B[版面分析\n列数/行数/版心区域\nocr/paddle_pipeline.py]
    B --> C[文字区与白口区划分]
    C --> D[局部预处理\n灰度化/归一化]
    D --> E[阈值分割与边缘检测\nOtsu/Canny 等]
    E --> F[连通域分析\n筛选异常区域]
    F --> G[候选区域过滤\n去除文字与正常纹理]
    G --> H[几何特征计算\n面积比例/长宽比/边界曲折度]
    H --> I[纹理特征计算\nGLCM/LBP 等]
    I --> J[视觉嵌入向量\n预训练视觉模型]
    J --> K[综合破损特征描述\n用于聚类与分级]
```

---

## 3. 版本指纹构建与相似版本检索

```mermaid
flowchart LR
    A[古籍页面图像] --> B[版面参数提取\n列数/行数/行高/行均字数]
    B --> C[版心边框与白口/黑口识别]
    C --> D[小字/注释结构检测]
    D --> E[字形局部片段裁剪]
    E --> F[版面结构向量编码]
    E --> G[字形风格向量编码\nChinese-CLIP 等]
    F --> H[版本特征拼接\n版式 + 字形]
    G --> H
    H --> I[向量索引写入\nrag/index]
    I --> J[新页面查询时\n相似版本检索]
    J --> K[相似样本集合\n供版本判定与聚类]
```

---

## 4. 钤印识别与实体聚类流程

```mermaid
flowchart LR
    A[输入页面图像] --> B[颜色空间转换\nRGB -> HSV/YCbCr]
    B --> C[红色高饱和区域初筛]
    C --> D[轮廓与形状分析\n矩形/方形/椭圆]
    D --> E[尺寸与位置过滤\n排除噪声]
    E --> F[印章区域裁剪与归一化]
    F --> G[视觉特征提取\nCNN/CLIP 向量]
    G --> H[印章向量聚类\n同一印章多实例合并]
    H --> I[与著录信息对照\n收藏家/机构命名]
    I --> J[生成钤印实体库\n印章-古籍-版本-馆藏关联]
```

---

## 5. 著录识别与 RAG 语义增强流程

```mermaid
flowchart LR
    A[页面图像 + OCR 文本] --> B[标题/卷端/牌记定位]
    B --> C[初步字段抽取\n题名/卷次/著者线索]
    C --> D[向量检索相似页面\nrag/retriever.py]
    D --> E[汇总上下文元数据\n题名/版本/藏地等]
    E --> F[构造 RAG Prompt\nrag/prompt.py]
    F --> G[调用大模型\n生成结构化著录]
    G --> H[字段规范化与校验\n对照现有目录]
    H --> I[写入著录知识库\n支持检索与分析]
```

---

如需，我可以再单独为“破损分级”或“整体系统架构（前端–Django–RAG–LLM）”画一张更偏宏观的架构图，或者帮你把这些 Mermaid 渲染成 PNG 并示意如何插入到论文/报告中。