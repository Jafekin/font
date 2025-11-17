> *“把历史文献里的细小纹理交给机器看见，把识别到的故事交还给人。”*

# 古文字识别助手 · Django

跨越千年的手稿，需要一个足够细腻的数字伙伴。这个项目把 Django、视觉大模型与检索增强生成（RAG）放在同一个舞台，让甲骨文、敦煌文书、金石拓片都能在浏览器中获得即时的释读和结构化报告。

---

## 目录
- [古文字识别助手 · Django](#古文字识别助手--django)
  - [目录](#目录)
  - [功能亮点](#功能亮点)
  - [系统架构](#系统架构)
  - [快速上手](#快速上手)
  - [环境变量](#环境变量)
  - [运行与调试](#运行与调试)
  - [API 速览](#api-速览)
  - [RAG 工作流](#rag-工作流)
  - [项目结构](#项目结构)
  - [常见问题](#常见问题)
  - [路线图](#路线图)
  - [许可证](#许可证)

---

## 功能亮点

| 功能 | 描述 |
| --- | --- |
| 多模态识别 | 上传 PNG/JPG 即刻触发 ERNIE-4.5-Turbo-VL，返回结构化 Markdown 或 JSON 报告。 |
| 提示增强 | 支持年代、出处等自定义提示词，指导模型聚焦正确语境。 |
| 历史留痕 | 自动记录分析历史，方便专家复盘与比对。 |
| 宣纸风界面 | 原生模板提供素雅的水墨 UI，适合展陈或教学场景。 |
| RAG 加持 | CLIP 向量、Faiss 检索与 Prompt 拼装，提供语境增强与引用依据。 |

---

## 系统架构

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│ 前端上传层  │ ─→ │ Django 视图层 │ ─→ │ RAG Pipeline │ ─→ │ ERNIE-VL 模型 │
└────────────┘     └──────────────┘     └──────────────┘     └────────────┘
        ↑                    │                      │                    ↓
      历史记录  ←────────────┘                      └── Faiss 索引 / CLIP 向量
```

关键模块：
- **app/views.py**：HTTP 入口、文件解析、结果持久化。
- **rag/embeddings.py**：惰性加载 CLIP，统一生成 512 维向量。
- **rag/retriever.py**：封装 Faiss/文献检索逻辑。
- **rag/prompt.py**：根据检索上下文构造 Markdown 或 JSON Prompt。
- **rag/pipeline.py**：将检索、提示、LLM 调用串联，返回引用、得分、上下文片段等完整产物。

---

## 快速上手

```bash
# 1. 克隆项目
git clone <repo-url> font && cd font

# 2. 创建虚拟环境（推荐同目录，不覆盖系统 Python）
python3 -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 初始化数据库
.venv/bin/python manage.py migrate

# 5.（可选）创建管理员
.venv/bin/python manage.py createsuperuser

# 6. 启动开发服务器
.venv/bin/python manage.py runserver 0.0.0.0:8000
```

访问：
- 前台界面：http://localhost:8000
- Django Admin：http://localhost:8000/admin

---

## 环境变量

将 `.env.example` 复制为 `.env`，填入以下字段：

| 变量 | 示例值 | 说明 |
| --- | --- | --- |
| `DEBUG` | `True` | 生产务必改为 `False` |
| `SECRET_KEY` | `change-me` | 生成随机字符串，防止安全风险 |
| `OPENAI_API_KEY` | `your-baai-key` | 对接文心大模型的密钥 |
| `OPENAI_BASE_URL` | `https://aistudio.baidu.com/llm/lmapi/v3` | 可替换成私有化接入网关 |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | 生产环境填入真实域名 |

> 生产部署记得额外设置 `CSRF_TRUSTED_ORIGINS`、`SECURE_HSTS_SECONDS` 等安全参数。

---

## 运行与调试

```bash
# 运行单元测试
.venv/bin/python -m pytest tests

# 构建 / 重建 Faiss 索引
.venv/bin/python scripts/build_index.py --source media/uploads --out rag/faiss.index

# 启动 VS Code 默认任务（server）
# Cmd+Shift+B / Ctrl+Shift+B -> Django: Run Server
```

常用管理命令：
- `python manage.py shell`：快速验证模型或 ORM。
- `python manage.py collectstatic`：部署前收集静态文件。
- `python manage.py dumpdata app.ScriptAnalysis`：导出历史分析记录备份。

---

## API 速览

| Endpoint | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 首页上传界面 |
| `/api/analyze` | POST (multipart) | 上传图片触发分析，返回 Markdown 报告 |
| `/api/analyze-base64` | POST (JSON) | 传入 Base64 图片，适合移动端或前端截图 |
| `/api/history` | GET | 获取最近 20 条分析记录 |

示例（multipart）：

```python
import requests

with open("tests/image/22.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/analyze",
        files={"image": f},
        data={"script_type": "甲骨文", "hint": "商晚期 卜辞"}
    )
print(resp.json())
```

---

## RAG 工作流

1. **嵌入**：`scripts/build_index.py` 遍历 `media/uploads`，调用 `rag/embeddings.py` 生成图文向量。
2. **检索**：`rag/retriever.py` 通过 Faiss Top-K 搜索相似版本、题跋或拓片。
3. **提示词**：`rag/prompt.py` 将检索到的上下文拼入高级 Prompt，可切换 Markdown / JSON 模板。
4. **生成**：`rag/pipeline.py` 调用 `analyze_ancient_script`，输出分析正文、引用列表、分数、上下文片段数量等。

快速调用：

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(index_path="rag/faiss.index")
result = pipeline.run(
    image_path="media/uploads/2025/11/05/example.png",
    script_type="甲骨文",
    hint="王卜辞"
)
print(result["analysis"])
print(result["num_references"], result["retrieved_references"])
```

---

## 项目结构

```
font/
├── app/                 # Django 主应用（模型、视图、模板）
├── config/              # 全局设置、路由、WSGI 入口
├── rag/                 # 向量、检索、Prompt、Pipeline
├── scripts/             # 数据与索引构建脚本
├── media/uploads/       # 用户上传图片
├── tests/               # Pytest 测试与示例图片
├── requirements.txt     # 依赖列表
└── README.md            # 项目前言
```

---

## 常见问题

**Q: 首次运行报错 `OpenAI library is not installed`？**  
A: 重新执行 `pip install -r requirements.txt`，或确认虚拟环境已激活。

**Q: 上传图片后报 500，提示 `analysis failed`？**  
A: 检查 `.env` 中的密钥与 Base URL 是否可访问；必要时在 `app/views.py` 中打开日志。

**Q: Faiss 索引过大或内存不足？**  
A: 调整 `scripts/build_index.py` 的分桶策略，或考虑使用 Milvus/Weaviate 这样的向量数据库。

---

## 路线图

- [ ] JSON-LD 标注管线（断裂等级、修复建议、版本指纹）。
- [ ] 向量索引增量更新与自动重建。
- [ ] RAG 结果缓存与引用可视化。
- [ ] Celery 异步任务，将批量识别与索引构建解耦。
- [ ] 多语言界面与 API 结果翻译。

---

## 许可证

MIT License © 古文字识别助手团队

欢迎通过 Issue / PR 分享你的发现、想法或下一步需求。
