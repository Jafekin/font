# 开发者速查表 - Django 古文字识别应用

## 🔗 快速链接

| 链接 | URL |
|------|-----|
| 应用主页 | http://localhost:8000 |
| 后台管理 | http://localhost:8000/admin |
| API - 分析 | POST http://localhost:8000/api/analyze |
| API - 历史 | GET http://localhost:8000/api/history |

## 👤 登录信息

```
用户名: admin
密码: password123
```

## 📂 关键文件位置

```
app/
├── models.py           # 数据模型定义
├── views.py            # 业务逻辑和 API
├── urls.py             # URL 路由配置
├── admin.py            # 后台管理配置
└── templates/
    └── index.html      # 前端页面

config/
├── settings.py         # Django 配置
├── urls.py             # 主 URL 配置
└── wsgi.py             # WSGI 应用入口
```

## 🚀 常见操作

### 启动应用

```bash
# 方式 1: 命令行
python manage.py runserver

# 方式 2: VS Code (Cmd+Shift+B)
# 方式 3: 后台运行
python manage.py runserver &
```

### 停止应用

```bash
# Ctrl+C 停止前台运行
# 或查找进程后杀死
kill -9 $(lsof -t -i :8000)
```

### 数据库操作

```bash
# 创建迁移
python manage.py makemigrations

# 应用迁移
python manage.py migrate

# 回滚迁移
python manage.py migrate app 0001

# 进入 Shell
python manage.py shell

# 删除数据库（危险！）
rm db.sqlite3
python manage.py migrate
```

### 创建用户

```bash
# 交互式创建
python manage.py createsuperuser

# 命令行创建
python manage.py createsuperuser --noinput \
  --username admin \
  --email admin@example.com
```

## 📝 修改指南

### 添加新的古文字类型

1. **编辑 models.py**:
```python
SCRIPT_TYPE_CHOICES = [
    ('甲骨文', '甲骨文'),
    ('新类型', '新类型'),  # 添加这行
]
```

2. **编辑 index.html**:
```html
<select id="scriptType">
    <option value="甲骨文">甲骨文</option>
    <option value="新类型">新类型</option>  <!-- 添加这行 -->
</select>
```

3. **运行迁移**:
```bash
python manage.py makemigrations
python manage.py migrate
```

### 修改 AI 提示词

编辑 `app/views.py` 中的 `analyze_ancient_script()` 函数：

```python
prompt_text = (
    "你的新提示词在这里..."
)
```

### 自定义样式

编辑 `app/templates/index.html` 中的 CSS：

```html
<style>
    :root {
        --primary: #8c3b1a;  /* 修改主色 */
        --accent: #c72e2e;   /* 修改强调色 */
    }
</style>
```

## 🔍 常见问题速解

| 问题 | 解决方案 |
|------|--------|
| 无法连接数据库 | `python manage.py migrate` |
| API 返回 401 错误 | 检查 `.env` 中的 API 密钥 |
| 静态文件未加载 | `python manage.py collectstatic` |
| 图片上传失败 | 检查文件大小和格式 |
| 端口被占用 | `python manage.py runserver 8001` |

## 🧪 测试 API

### 使用 curl 测试分析 API

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@image.jpg" \
  -F "script_type=甲骨文" \
  -F "hint=商代卜辞"
```

### 使用 Python 测试

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze',
        files={'image': f},
        data={
            'script_type': '甲骨文',
            'hint': 'optional hint'
        }
    )
    print(response.json())
```

### 使用 curl 获取历史记录

```bash
curl http://localhost:8000/api/history
```

## 📊 查看数据库

### 使用 Django Shell

```bash
python manage.py shell

# 获取所有分析记录
from app.models import ScriptAnalysis
records = ScriptAnalysis.objects.all()

# 获取最近的记录
latest = ScriptAnalysis.objects.latest('created_at')

# 按类型过滤
jiaguwen = ScriptAnalysis.objects.filter(script_type='甲骨文')

# 删除记录
record = ScriptAnalysis.objects.get(id=1)
record.delete()
```

### 使用 Django 后台

访问 http://localhost:8000/admin 登录后直接管理

## 🔐 安全检查清单

- [ ] 修改 `settings.py` 中的 `SECRET_KEY`
- [ ] 设置 `DEBUG = False` 在生产环境
- [ ] 配置 `ALLOWED_HOSTS` 为实际域名
- [ ] 使用强密码替换默认密码
- [ ] 启用 HTTPS
- [ ] 定期备份数据库

## 📦 依赖管理

```bash
# 查看已安装的包
pip list

# 查看过时的包
pip list --outdated

# 更新所有包
pip install --upgrade -r requirements.txt

# 安装特定版本
pip install Django==4.2.7

# 生成 requirements.txt
pip freeze > requirements.txt
```

## 🐛 调试技巧

### 启用 Django Debug Toolbar

```bash
pip install django-debug-toolbar

# 在 settings.py 中添加：
INSTALLED_APPS += ['debug_toolbar']
MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
INTERNAL_IPS = ['127.0.0.1']
```

### 查看数据库查询

```python
from django.db import connection
from django.test.utils import CaptureQueriesContext

with CaptureQueriesContext(connection) as context:
    # 代码
    pass

print(f"Queries: {len(context)}")
for query in context:
    print(query['sql'])
```

### 打印日志

```python
import logging

logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## 🚢 部署检查清单

**本地开发**
- [ ] 应用正常运行
- [ ] API 端点正常
- [ ] 数据库迁移完成
- [ ] 所有依赖已安装

**生产前**
- [ ] `DEBUG = False`
- [ ] `SECRET_KEY` 已更改
- [ ] 数据库备份完成
- [ ] 静态文件已收集
- [ ] HTTPS 已启用
- [ ] 日志已配置

**生产后**
- [ ] 定期检查日志
- [ ] 监控服务器性能
- [ ] 定期备份数据库
- [ ] 及时更新依赖包

## 🔗 相关文档

- `README.md` - 完整技术文档
- `QUICKSTART.md` - 快速开始指南
- `DEPLOYMENT_SUMMARY.md` - 部署总结
- `.github/copilot-instructions.md` - AI 助手指令

## 📞 技术支持

遇到问题？

1. 查看相关文档
2. 检查错误日志
3. 查看 Django 错误页面
4. 使用 Django Shell 调试

---

**最后更新**: 2025-10-31  
**作者**: Copilot  
**版本**: 1.0.0

