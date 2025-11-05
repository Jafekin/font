# 📜 古文字识别助手 - Django 版本

一个基于 Django 框架的古文字（甲骨文、敦煌文书等）识别和释读助手，集成了百度文心大模型的视觉识别能力。

## 功能特性

✨ **核心功能**
- 🖼️ 上传高清古文字图片（PNG/JPG）
- 🤖 基于文心大模型的智能识别与释读
- 📋 支持多种古文字体系（甲骨文、敦煌文书、金文、篆书、隶书）
- 💡 提供用户提示支持（时代、文本片段、出处等）
- 📊 完整的分析报告（Markdown 格式）
- 💾 分析历史记录存储
- 🎨 宣纸水墨风设计界面

## 安装与配置

### 1. 环境准备

```bash
# 克隆或进入项目目录
cd /Users/jafekin/Codes/Python\ Projects/font

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境变量配置

复制示例文件并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下内容：

```env
DEBUG=True
SECRET_KEY=your-secret-key-change-this-in-production
OPENAI_API_KEY=5124588afef0ea3dba67365a5b9cd429283ba74d
OPENAI_BASE_URL=https://aistudio.baidu.com/llm/lmapi/v3
ALLOWED_HOSTS=localhost,127.0.0.1
```

> ⚠️ **生产环境注意**：
> - 更改 `SECRET_KEY` 为一个安全的随机字符串
> - 设置 `DEBUG=False`
> - 配置正确的 `ALLOWED_HOSTS`

### 3. 数据库迁移

```bash
python manage.py migrate
```

### 4. 创建超级用户（可选）

```bash
python manage.py createsuperuser
```

## 快速开始

### 1. 启动开发服务器

```bash
# 使用命令行
python manage.py runserver

# 或使用 VS Code 任务
# 快捷键: Ctrl+Shift+B 或 Cmd+Shift+B (macOS)
```

### 2. 访问应用

主页: **http://localhost:8000**

后台管理: **http://localhost:8000/admin**

**登录凭证**:
- 用户名: `admin`
- 密码: `password123`

### 3. 使用应用

1. 在主页上传古文字图片（PNG/JPG）
2. 选择文字类型和提供可选提示
3. 点击"开始识别古文字"
4. 系统将调用百度文心大模型进行识别
5. 查看分析结果和历史记录

## 项目结构

```
font/
├── manage.py                 # Django 管理脚本
├── requirements.txt          # Python 依赖
├── .env.example             # 环境变量示例
├── README.md                # 本文件
│
├── config/                  # Django 配置
│   ├── settings.py          # 全局设置
│   ├── urls.py              # URL 路由
│   ├── wsgi.py              # WSGI 应用
│   └── __init__.py
│
├── app/                     # 主应用
│   ├── templates/
│   │   └── index.html       # 前端界面
│   ├── static/              # 静态文件（CSS/JS）
│   ├── migrations/          # 数据库迁移
│   ├── models.py            # 数据模型
│   ├── views.py             # 业务逻辑
│   ├── urls.py              # 应用路由
│   ├── admin.py             # 后台管理配置
│   ├── apps.py              # 应用配置
│   └── __init__.py
│
└── media/                   # 用户上传文件
    └── uploads/
```

## API 接口

### 1. 主页
- **URL**: `/`
- **方法**: GET
- **功能**: 返回前端界面

### 2. 分析接口
- **URL**: `/api/analyze`
- **方法**: POST
- **参数**:
  - `image`: 图片文件 (必需)
  - `script_type`: 文字类型 (可选，默认：甲骨文)
  - `hint`: 用户提示 (可选)

**请求示例**:
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'script_type': '甲骨文',
        'hint': '商代卜辞'
    }
    response = requests.post('http://localhost:8000/api/analyze', files=files, data=data)
    print(response.json())
```

**响应示例**:
```json
{
  "success": true,
  "result": "# 初步判读\n- 可能的文字体系：甲骨文...",
  "analysis_id": 1
}
```

### 3. Base64 分析接口
- **URL**: `/api/analyze-base64`
- **方法**: POST
- **参数** (JSON):
  - `image`: Base64 图片数据 (必需)
  - `script_type`: 文字类型 (可选)
  - `hint`: 用户提示 (可选)

### 4. 历史记录接口
- **URL**: `/api/history`
- **方法**: GET
- **响应**: 最近 20 条分析记录

## 技术栈

- **后端框架**: Django 4.2
- **数据库**: SQLite（开发）/ PostgreSQL（生产推荐）
- **AI 模型**: 百度文心大模型（ERNiE-4.5-Turbo-VL）
- **API 库**: OpenAI Python SDK
- **图像处理**: Pillow
- **前端**: Vanilla JavaScript + CSS3

## 配置文件说明

### settings.py 主要配置项

```python
# 上传文件大小限制（默认 50MB）
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800
FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800

# OpenAI/百度 API 配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

# 上传文件存储位置
MEDIA_ROOT = BASE_DIR / 'media'
MEDIA_URL = '/media/'
```

## 故障排除

### 问题 1: 导入 Pillow 失败

```bash
pip install --upgrade Pillow
```

### 问题 2: OpenAI API 错误

检查：
1. `OPENAI_API_KEY` 是否正确
2. 网络连接是否正常
3. API 配额是否充足

### 问题 3: 图片上传失败

- 检查文件大小是否超过限制
- 确认文件格式为 PNG 或 JPG
- 检查 `media/` 文件夹权限

## 性能优化建议

1. **生产部署**:
   - 使用 PostgreSQL 替代 SQLite
   - 配置 Nginx 反向代理
   - 使用 Gunicorn 或 uWSGI 作为应用服务器
   - 启用静态文件缓存

2. **API 优化**:
   - 添加请求缓存
   - 实现异步任务队列（Celery）
   - 添加速率限制

3. **前端优化**:
   - 压缩和最小化 CSS/JS
   - 使用 CDN 加速
   - 启用 Gzip 压缩

## 扩展功能建议

- 🔐 用户认证和权限管理
- 📧 分析结果邮件通知
- 🏷️ 标签和分类系统
- 🔄 导出功能（PDF/Word）
- 📱 移动端适配优化
- 🌍 多语言支持

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**: 2025年10月31日

