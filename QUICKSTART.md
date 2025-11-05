# Django 古文字识别应用 - 快速指南

## 📋 项目概述

这是一个基于 Django 框架的现代化古文字识别和释读应用。

**核心功能**:
✅ 古文字图片上传和识别  
✅ AI 驱动的智能分析和释读  
✅ 多种古文字体系支持  
✅ 宣纸水墨风格 UI  
✅ 分析历史记录管理  

---

## 🚀 快速启动

### 环境配置（第一次运行）

```bash
# 1. 进入项目目录
cd /Users/jafekin/Codes/Python\ Projects/font

# 2. 激活虚拟环境（已创建）
source .venv/bin/activate

# 3. 安装依赖（已完成）
pip install -r requirements.txt

# 4. 配置环境变量（已完成）
# 检查 .env 文件配置

# 5. 初始化数据库（已完成）
python manage.py migrate

# 6. 创建超级用户（已完成）
python manage.py createsuperuser
```

### 启动应用

**方法 1: 命令行**
```bash
python manage.py runserver
```

**方法 2: VS Code**
- 快捷键: `Cmd+Shift+B` (macOS)
- 或在 VS Code 中选择 `Django: Run Server` 任务

**方法 3: 生产环境**
```bash
gunicorn config.wsgi:application --bind 0.0.0.0:8000
```

---

## 🌐 访问地址

| 页面 | URL | 说明 |
|------|-----|------|
| 主页 | http://localhost:8000 | 古文字识别界面 |
| 后台 | http://localhost:8000/admin | 管理后台 |
| API | http://localhost:8000/api/analyze | 图片分析接口 |

### 登录凭证

```
用户名: admin
密码: password123
```

---

## 📁 项目结构

```
font/
├── manage.py                    # Django 管理脚本
├── requirements.txt             # Python 依赖
├── .env                         # 环境变量
├── README.md                    # 完整文档
├── QUICKSTART.md               # 本文件
│
├── config/                      # 配置目录
│   ├── settings.py             # Django 设置
│   ├── urls.py                 # 主 URL 配置
│   └── wsgi.py                 # WSGI 应用
│
├── app/                         # 主应用
│   ├── templates/
│   │   └── index.html          # 前端页面
│   ├── static/                 # 静态文件
│   ├── models.py               # 数据模型
│   ├── views.py                # 视图逻辑
│   ├── urls.py                 # URL 路由
│   ├── admin.py                # 后台配置
│   └── migrations/             # 数据库迁移
│
└── .vscode/
    └── tasks.json              # VS Code 任务配置
```

---

## 🔧 常用命令

```bash
# 启动服务器
python manage.py runserver

# 创建迁移
python manage.py makemigrations

# 应用迁移
python manage.py migrate

# 进入 Django Shell
python manage.py shell

# 创建超级用户
python manage.py createsuperuser

# 查看数据库
python manage.py dbshell

# 收集静态文件
python manage.py collectstatic

# 清理旧会话
python manage.py clearsessions
```

---

## 🛠️ 配置说明

### .env 文件

```env
DEBUG=True                          # 调试模式
SECRET_KEY=your-secret-key          # 密钥
OPENAI_API_KEY=your-api-key         # 百度 API 密钥
OPENAI_BASE_URL=api-endpoint        # API 端点
ALLOWED_HOSTS=localhost,127.0.0.1   # 允许的主机
```

### settings.py 关键设置

```python
# 上传文件大小限制
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB

# 数据库
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 时区和语言
LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
```

---

## 📱 API 使用示例

### 图片分析 API

```bash
# 使用 curl
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@image.jpg" \
  -F "script_type=甲骨文" \
  -F "hint=商代卜辞"

# 使用 Python
import requests

with open('image.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'script_type': '甲骨文',
        'hint': '商代卜辞'
    }
    response = requests.post(
        'http://localhost:8000/api/analyze',
        files=files,
        data=data
    )
    print(response.json())
```

**响应格式**:
```json
{
  "success": true,
  "result": "# 初步判读\n...",
  "analysis_id": 1
}
```

---

## 🐛 故障排除

### 问题 1: 无法连接到数据库

```bash
# 解决方案：重新迁移
python manage.py migrate
```

### 问题 2: OpenAI API 错误

- 检查 `.env` 中的 `OPENAI_API_KEY` 是否正确
- 确保网络连接正常
- 验证 API 配额未用完

### 问题 3: 静态文件未加载

```bash
# 收集静态文件
python manage.py collectstatic --noinput
```

### 问题 4: 导入错误

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt
```

---

## 📊 性能优化

### 生产环境部署

```bash
# 1. 使用 PostgreSQL
pip install psycopg2-binary

# 2. 使用 Gunicorn
pip install gunicorn
gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 4

# 3. 使用 Nginx 反向代理
# 配置 Nginx 转发请求到 Gunicorn

# 4. 启用缓存
# 配置 Redis 或 Memcached
```

### 代码级优化

- 使用数据库索引加快查询
- 添加请求缓存
- 实现异步任务队列（Celery）
- 压缩静态文件

---

## 📚 扩展功能建议

- 🔐 用户认证和权限管理
- 📧 结果邮件通知
- 🏷️ 标签和分类系统
- 📄 导出功能（PDF/Word）
- 🌍 多语言支持
- 📊 数据分析仪表板
- 🎨 更多主题切换

---

## 📞 技术支持

### 遇到问题？

1. 检查日志输出
2. 查看 Django 错误页面
3. 参考完整文档 (README.md)
4. 检查依赖版本兼容性

### 依赖版本

```
Python: 3.13.2
Django: 4.2.7
Pillow: 12.0.0
OpenAI: 1.3.9
```

---

## 📝 常见问题

**Q: 如何添加新的古文字类型？**

A: 编辑 `app/models.py` 中的 `SCRIPT_TYPE_CHOICES`：
```python
SCRIPT_TYPE_CHOICES = [
    ('甲骨文', '甲骨文'),
    ('你的类型', '显示名称'),
]
```

**Q: 如何修改 API 密钥？**

A: 编辑 `.env` 文件，修改 `OPENAI_API_KEY` 字段，重启服务器。

**Q: 支持哪些图片格式？**

A: PNG、JPG、JPEG、GIF、BMP 等常见格式。

---

## 🎉 完成！

项目已成功配置并运行。访问 http://localhost:8000 开始使用吧！

祝你使用愉快！ 📜

