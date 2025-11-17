# RAG 使用示例

## 快速开始

### 1. 构建知识库索引

首先，准备图片和对应的文字信息文件：

```
media/uploads/
├── image1.jpg
├── image1.json          # 对应的文字信息（JSON格式）
├── image2.png
└── image2.txt           # 对应的文字信息（文本格式）
```

**文字信息文件格式示例**：

`image1.json`:
```json
{
  "description": "这是一张史记的古籍图片，记录了司马迁的史记内容...",
  "title": "史记一百三十卷",
  "author": "（汉）司马迁",
  "edition": "北宋刻本"
}
```

`image2.txt`:
```
这是一张甲骨文图片，记录了商代晚期的卜辞内容。
主要内容包括：祭祀、田猎、征伐等。
```

然后运行索引构建脚本：

```bash
python scripts/build_index.py \
    --image-dir media/uploads \
    --index-path rag/index
```

### 2. 使用RAG管道分析图片

```python
from rag.pipeline import RAGPipeline

# 初始化管道
pipeline = RAGPipeline(index_path="rag/index")

# 用户上传图片后，进行分析
result = pipeline.run(
    image_path="/Users/jafekin/Codes/Python Projects/font/data/名录 史记2025-11-6/A史记 集解本 1北宋刻本/【1】00393 史记一百三十卷 （汉）司马迁撰 （南朝宋）裴骃集解 北宋刻本（卷一至四、八至一百三十配南宋初建阳刻本） 北京大学图书馆/0000004a.jpg",
    script_type="汉文古籍",
    hint="用户提供的提示信息（可选）",
    k=5  # 检索top-5个相似图片
)

# 检查结果
if result['success']:
    # LLM的分析结果
    analysis = result['analysis']
    print("分析结果:", analysis)
    
    # 检索到的文字信息（这些信息会被输入给LLM）
    text_info = result['retrieved_text_info']
    print("\n检索到的文字信息:")
    for i, info in enumerate(text_info, 1):
        print(f"{i}. {info[:100]}...")  # 显示前100个字符
    
    # 参考来源
    references = result['retrieved_references']
    print(f"\n参考来源: {references}")
    
    # 相似度分数
    scores = result['retrieval_scores']
    print(f"相似度分数: {scores}")
else:
    print(f"错误: {result['error']}")
```

### 3. 仅检索相似图片（不调用LLM）

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(index_path="rag/index")

# 搜索相似图片
similar_images = pipeline.search_similar(
    query_image_path="/Users/jafekin/Codes/Python Projects/font/data/名录 史记2025-11-6/A史记 集解本 1北宋刻本/【1】00393 史记一百三十卷 （汉）司马迁撰 （南朝宋）裴骃集解 北宋刻本（卷一至四、八至一百三十配南宋初建阳刻本） 北京大学图书馆/0000004a.jpg",
    k=5
)

for item in similar_images:
    print(f"图片ID: {item['id']}")
    print(f"图片路径: {item['image_path']}")
    print(f"文字信息: {item['text_info']}")
    print(f"相似度: {item['score']:.3f}")
    print("---")
```

### 4. 获取图片对应的文字信息

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(index_path="rag/index")

# 直接获取图片对应的文字信息
text_info_list = pipeline.get_text_info_for_image("image.jpg", k=1)

if text_info_list:
    print(f"文字信息: {text_info_list[0]}")
else:
    print("未找到对应的文字信息")
```

## 工作流程说明

1. **构建知识库**：
   - 将图片和对应的文字信息组织好
   - 运行`build_index.py`构建索引
   - 索引会保存图片的向量表示和对应的文字信息

2. **用户上传图片**：
   - 用户通过前端上传图片

3. **检索相似图片**：
   - 系统使用txtai检索知识库中与用户图片相似的图片
   - 返回相似图片及其对应的文字信息

4. **构建增强提示词**：
   - 将检索到的文字信息作为上下文
   - 与用户提示一起构建增强的提示词

5. **LLM分析**：
   - 将增强的提示词和用户图片一起输入给LLM
   - LLM基于检索到的信息进行分析和回答

6. **返回结果**：
   - 返回LLM的分析结果
   - 同时返回检索到的文字信息和参考来源

## 注意事项

1. **文字信息文件命名**：
   - 图片文件：`image.jpg`
   - 对应的文字信息文件：`image.json` 或 `image.txt`
   - 文件名（不含扩展名）必须相同

2. **文字信息格式**：
   - JSON格式：脚本会自动提取`description`、`text`、`content`等字段
   - 文本格式：直接读取文件内容
   - 如果找不到文字信息文件，会使用图片文件名作为默认信息

3. **索引更新**：
   - 添加新图片后，需要重新运行`build_index.py`更新索引
   - 或者实现增量索引功能（未来扩展）

4. **性能优化**：
   - 使用批量处理可以提高索引构建速度
   - 调整`--batch-size`参数（默认32）
   - 确保有足够的GPU内存（如果有GPU）

