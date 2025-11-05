# 🔧 故障排除指南

## 已解决的问题

### 问题 1: OpenAI Client 初始化错误

**错误信息**:
```
Client.__init__() got an unexpected keyword argument 'proxies'
```

**原因**:
- OpenAI 库版本过旧（1.3.9）
- 新版本 API 签名与旧版本不兼容

**解决方案**:
✅ 已升级 OpenAI 到 2.6.1 版本

**操作步骤**:
```bash
# 升级 OpenAI 库
pip install --upgrade openai

# 或指定版本
pip install openai>=1.40.0
```

**验证**:
```bash
python -c "import openai; print(openai.__version__)"
```

---

## 常见问题列表

### 1. 无法连接到百度 API

**症状**:
```
分析失败: Connection refused / Connection timeout
```

**检查项**:
- [ ] 检查 `.env` 中的 `OPENAI_API_KEY` 是否正确
- [ ] 检查 `OPENAI_BASE_URL` 是否为 `https://aistudio.baidu.com/llm/lmapi/v3`
- [ ] 检查网络连接是否正常
- [ ] 检查 API 配额是否充足

**解决方案**:
```bash
# 1. 验证环境变量
cat .env | grep OPENAI

# 2. 测试 API 连接
python -c "
from openai import OpenAI
client = OpenAI(
    api_key='your_key',
    base_url='https://aistudio.baidu.com/llm/lmapi/v3'
)
print('Connected successfully!')
"
```

### 2. 图片上传失败

**症状**:
```
分析失败: File size exceeds maximum allowed size
```

**检查项**:
- [ ] 文件大小是否超过 50MB
- [ ] 文件格式是否为 PNG/JPG/JPEG
- [ ] 文件是否损坏

**解决方案**:
```bash
# 在 settings.py 中增加文件大小限制
DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600
```

### 3. 数据库错误

**症状**:
```
OperationalError: no such table: app_scriptanalysis
```

**解决方案**:
```bash
# 运行迁移
python manage.py migrate
```

### 4. 静态文件未加载

**症状**:
- 页面样式错乱
- 图标无法显示

**解决方案**:
```bash
# 收集静态文件
python manage.py collectstatic --noinput

# 或在 settings.py 中禁用调试模式下的收集
# DEBUG = False
```

### 5. 端口被占用

**症状**:
```
Error: That port is already in use.
```

**解决方案**:
```bash
# 使用不同的端口
python manage.py runserver 8001

# 或杀死占用端口的进程
kill -9 $(lsof -t -i :8000)
```

### 6. 虚拟环境问题

**症状**:
```
ModuleNotFoundError: No module named 'django'
```

**解决方案**:
```bash
# 激活虚拟环境
source .venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt
```

### 7. 导入错误

**症状**:
```
ImportError: cannot import name 'OpenAI' from 'openai'
```

**原因**: OpenAI 库版本不兼容

**解决方案**:
```bash
# 卸载并重新安装
pip uninstall openai -y
pip install openai>=1.40.0
```

### 8. 权限错误

**症状**:
```
PermissionError: [Errno 13] Permission denied: 'media/uploads/...'
```

**解决方案**:
```bash
# 修改目录权限
chmod -R 755 media/
chmod -R 755 app/static/
```

---

## 调试技巧

### 启用详细日志

在 `settings.py` 中添加：

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}
```

### 使用 Django Shell 调试

```bash
python manage.py shell

# 导入模型
from app.models import ScriptAnalysis

# 查询数据
records = ScriptAnalysis.objects.all()
print(records)

# 测试函数
from app.views import analyze_ancient_script
result = analyze_ancient_script(image, '甲骨文', '')
```

### 检查环境变量

```bash
# 查看所有环境变量
env | grep OPENAI

# 或在 Python 中检查
import os
print(os.getenv('OPENAI_API_KEY'))
```

### 测试 API 端点

```bash
# 使用 curl 测试
curl -X GET http://localhost:8000/api/history

# 或使用 Python
import requests
response = requests.get('http://localhost:8000/api/history')
print(response.json())
```

---

## 性能问题排查

### 服务器响应缓慢

**检查项**:
- [ ] 数据库查询是否过多
- [ ] AI 模型响应时间
- [ ] 网络延迟

**优化方案**:
```python
# 添加缓存
from django.views.decorators.cache import cache_page

@cache_page(60 * 5)  # 缓存 5 分钟
def history(request):
    ...

# 使用数据库连接池
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'CONN_MAX_AGE': 600,
    }
}
```

### 内存泄漏

**检查项**:
- [ ] 是否有未关闭的文件
- [ ] 是否有大对象未释放

**优化方案**:
```python
# 使用上下文管理器
with open(file_path, 'rb') as f:
    image = Image.open(f)
    # 处理图像

# 显式垃圾回收
import gc
gc.collect()
```

---

## 网络相关问题

### 跨域 (CORS) 错误

**症状**:
```
Access to XMLHttpRequest blocked by CORS policy
```

**解决方案**:
在 `settings.py` 中配置：

```python
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:8000',
    'https://yourdomain.com',
]
```

### SSL/TLS 证书错误

**症状**:
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**解决方案**:
```python
# 临时禁用（开发环境）
import urllib3
urllib3.disable_warnings()

# 或正确配置证书
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
```

---

## 数据库问题

### 数据库锁定

**症状**:
```
database is locked
```

**解决方案**:
```bash
# 对于 SQLite，删除锁文件
rm db.sqlite3-journal

# 或迁移到 PostgreSQL（生产环境推荐）
```

### 数据库连接失败

**症状**:
```
could not connect to server
```

**解决方案**:
```bash
# 检查数据库连接参数
python manage.py dbshell

# 重新创建数据库
rm db.sqlite3
python manage.py migrate
```

---

## 快速恢复步骤

### 完全重置应用

```bash
# 1. 停止服务器 (Ctrl+C)

# 2. 删除数据库
rm db.sqlite3

# 3. 重新迁移
python manage.py migrate

# 4. 重新创建超级用户
python manage.py createsuperuser

# 5. 重启服务器
python manage.py runserver
```

### 清除缓存

```bash
# Django 缓存
python manage.py clear_cache

# 浏览器缓存
# 按 Cmd+Shift+R (macOS) 或 Ctrl+Shift+R (Windows/Linux) 硬刷新
```

### 更新依赖

```bash
# 更新所有依赖
pip install --upgrade -r requirements.txt

# 冻结当前依赖版本
pip freeze > requirements.txt
```

---

## 获取帮助

### 查看错误日志

```bash
# Django 日志
# 错误信息通常显示在终端和浏览器的调试页面

# 数据库日志
python manage.py dbshell
```

### 常用命令

```bash
# 检查项目配置
python manage.py check

# 运行测试
python manage.py test

# 显示所有 SQL 查询
python manage.py shell
from django.db import connection
connection.queries
```

### 联系支持

- 查看项目文档: `README.md`, `QUICKSTART.md`
- 检查 GitHub Issues
- 查看 Django 官方文档
- 查看 OpenAI 官方文档

---

## 已知限制

- 🖼️ **图片大小**: 最大 50MB（可配置）
- ⏱️ **API 超时**: 30 秒
- 📊 **数据库**: SQLite（开发）、建议生产用 PostgreSQL
- 🌐 **并发**: 开发服务器单线程

---

**最后更新**: 2025-10-31  
**维护者**: Copilot  
**状态**: ✅ 维护中

