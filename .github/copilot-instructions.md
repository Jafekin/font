# GitHub Copilot 自定义指示

## 项目信息

**项目类型**: Django Web 应用  
**项目名称**: 古文字识别助手  
**开发语言**: Python 3.13  
**框架**: Django 4.2.7

## 项目概述

这是一个基于 Django 框架的古文字（甲骨文、敦煌文书等）识别和释读助手。原本是 Gradio 应用，已完整迁移到 Django 框架。

### 核心功能

✅ 上传和处理古文字图片  
✅ 集成百度文心大模型进行 AI 识别  
✅ Markdown 格式的智能分析报告  
✅ 分析历史记录数据库存储  
✅ 宣纸水墨风的现代 UI 设计

### 技术栈

- **后端**: Django 4.2.7 + SQLite/PostgreSQL
- **前端**: HTML5 + Vanilla JavaScript + CSS3
- **AI/ML**: OpenAI SDK + 百度文心大模型
- **图像处理**: Pillow
- **依赖管理**: pip + requirements.txt

## 文件结构

```
font/
├── manage.py                    # Django CLI
├── requirements.txt             # 依赖列表
├── .env                         # 环境变量配置
├── .gitignore                   # Git 忽略配置
├── README.md                    # 完整文档
├── QUICKSTART.md               # 快速开始指南
│
├── config/                      # 项目配置
│   ├── settings.py             # Django 设置
│   ├── urls.py                 # URL 路由
│   └── wsgi.py                 # WSGI 应用入口
│
├── app/                         # 主应用
│   ├── templates/index.html    # 前端页面
│   ├── static/                 # 静态文件目录
│   ├── models.py               # 数据模型 (ScriptAnalysis)
│   ├── views.py                # 视图和 API 逻辑
│   ├── urls.py                 # 应用级 URL 路由
│   ├── admin.py                # Django 后台配置
│   ├── apps.py                 # 应用配置
│   └── migrations/             # 数据库迁移文件
│
├── media/                       # 用户上传文件存储
│   └── uploads/               # 图片上传目录
│
└── .vscode/
    └── tasks.json             # VS Code 任务配置
```

## 主要模块说明

### models.py

定义 `ScriptAnalysis` 模型，存储分析记录：

- image: 上传的图片
- script_type: 古文字类型
- hint: 用户提示
- result: AI 分析结果
- created_at/updated_at: 时间戳

### views.py

实现核心逻辑：

- `analyze_ancient_script()`: AI 识别函数
- `index()`: 主页
- `analyze()`: 文件上传 API
- `analyze_base64()`: Base64 图片 API
- `history()`: 历史记录 API

### templates/index.html

现代响应式前端界面：

- 图片上传组件
- 参数配置表单
- 结果显示区域
- 历史记录面板

## 关键配置

### .env 环境变量

```env
DEBUG=True
SECRET_KEY=your-secret-key
OPENAI_API_KEY=百度API密钥
OPENAI_BASE_URL=https://aistudio.baidu.com/llm/lmapi/v3
ALLOWED_HOSTS=localhost,127.0.0.1
```

### settings.py 关键设置

- `INSTALLED_APPS`: 包含 Django 内置应用和自定义 app
- `MIDDLEWARE`: 包括 CORS 中间件
- `TEMPLATES`: 模板配置
- `DATABASES`: SQLite 数据库
- `MEDIA_ROOT/URL`: 上传文件路径

## 开发工作流

### 启动开发环境

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 运行开发服务器
python manage.py runserver

# 3. 访问应用
# 主页: http://localhost:8000
# 后台: http://localhost:8000/admin
```

### 常见任务

#### 创建数据库迁移

```bash
python manage.py makemigrations
python manage.py migrate
```

#### 创建后台用户

```bash
python manage.py createsuperuser
```

#### 收集静态文件（生产）

```bash
python manage.py collectstatic
```

#### 启动 Shell

```bash
python manage.py shell
```

## API 接口

### POST /api/analyze

上传图片进行分析

```
参数:
- image: 图片文件
- script_type: 文字类型
- hint: 用户提示

响应:
{
  "success": true,
  "result": "分析结果 Markdown",
  "analysis_id": 1
}
```

### POST /api/analyze-base64

上传 Base64 图片

```
JSON body:
{
  "image": "data:image/png;base64,...",
  "script_type": "甲骨文",
  "hint": "可选提示"
}
```

### GET /api/history

获取最近 20 条分析记录

## 扩展和修改指南

### 添加新的古文字类型

编辑 `app/models.py`:

```python
SCRIPT_TYPE_CHOICES = [
    ('甲骨文', '甲骨文'),
    ('新类型', '显示名称'),  # 添加这行
]
```

编辑 `app/templates/index.html`:

```html
<select id="scriptType">
  <option value="甲骨文">甲骨文</option>
  <option value="新类型">新类型</option>
  <!-- 添加这行 -->
</select>
```

### 修改 AI 提示词

编辑 `app/views.py` 中的 `analyze_ancient_script()` 函数，修改 `prompt_text` 变量。

### 自定义样式

编辑 `app/templates/index.html` 中的 `<style>` 标签，或在 `app/static/` 创建 CSS 文件。

## 生产部署

### 使用 Gunicorn + Nginx

```bash
# 1. 安装 Gunicorn
pip install gunicorn

# 2. 启动应用
gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 4

# 3. 配置 Nginx 反向代理（详见 README.md）
```

### 使用 Docker

创建 `Dockerfile`:

```dockerfile
FROM python:3.13
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## 依赖管理

### 更新依赖

```bash
# 查看过时的包
pip list --outdated

# 更新特定包
pip install --upgrade Django

# 重新生成 requirements.txt
pip freeze > requirements.txt
```

### 常见依赖问题

- Pillow: 图像处理库，需要系统依赖
- OpenAI: 百度 API SDK
- Django: Web 框架
- python-dotenv: 环境变量管理

## 测试

### 运行测试

```bash
# 所有测试
python manage.py test

# 特定应用
python manage.py test app

# 单个测试文件
python manage.py test app.tests.test_views
```

### 创建测试

在 `app/tests.py` 中创建：

```python
from django.test import TestCase
from app.models import ScriptAnalysis

class ScriptAnalysisTestCase(TestCase):
    def test_model_creation(self):
        # 测试代码
        pass
```

## 调试技巧

### 启用 Django Debug Toolbar

```bash
pip install django-debug-toolbar
```

在 `settings.py` 中添加：

```python
INSTALLED_APPS += ['debug_toolbar']
MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
```

### 查看数据库查询

```python
from django.db import connection
from django.test.utils import CaptureQueriesContext

with CaptureQueriesContext(connection) as context:
    # 代码
    print(context)
```

## 性能优化建议

1. **数据库**: 添加索引，使用 PostgreSQL 替代 SQLite
2. **缓存**: 启用 Redis 缓存
3. **异步**: 使用 Celery 处理长时任务
4. **前端**: 压缩 CSS/JS，使用 CDN
5. **API**: 实现分页，添加速率限制

## 常见问题和解决方案

### 1. 导入错误

```
虚拟环境未激活或依赖未安装
→ 运行: source .venv/bin/activate && pip install -r requirements.txt
```

### 2. 数据库错误

```
迁移未应用
→ 运行: python manage.py migrate
```

### 3. 静态文件未加载

```
静态文件未收集
→ 运行: python manage.py collectstatic --noinput
```

### 4. 端口被占用

```
修改运行命令: python manage.py runserver 8001
```

## 版本历史

- **v1.0.0** (2025-10-31): 从 Gradio 迁移到 Django，完整功能实现

## 许可证

MIT License

## 联系方式

项目维护和支持请查看 README.md 或 QUICKSTART.md 文档。
