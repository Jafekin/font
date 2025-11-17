# RAG v2.0 测试套件

本目录包含 RAG（检索增强生成）v2.0 txtai 实现的完整测试。

## 📋 测试文件

### 1. `test_rag_v2.py` - 单元测试

单元测试验证所有模块和函数的基本功能：

```bash
# 运行所有单元测试
python -m unittest tests.test_rag_v2 -v

# 运行特定测试类
python -m unittest tests.test_rag_v2.TestEmbeddings -v

# 运行特定测试方法
python -m unittest tests.test_rag_v2.TestEmbeddings.test_get_text_embedding_empty_string -v
```

**涵盖范围**：
- ✅ 模块导入
- ✅ 函数签名
- ✅ 错误处理
- ✅ 类定义
- ✅ 文档字符串
- ✅ 向后兼容性

### 2. `test_rag_integration.py` - 集成测试

集成测试展示各个模块的实际使用方式：

```bash
# 运行集成测试
python tests/test_rag_integration.py

# 或者
python -m tests.test_rag_integration
```

**涵盖范围**：
- ✅ 模块导入（所有 RAG 子模块）
- ✅ 向量化功能（文本嵌入）
- ✅ 提示词生成
- ✅ 类方法检查
- ✅ 向后兼容性验证
- ✅ 文档完整性

## 🚀 快速开始

### 方式 1：运行单元测试（推荐）

```bash
cd /Users/jafekin/Codes/Python\ Projects/font

# 激活虚拟环境
source .venv/bin/activate

# 运行所有单元测试
python -m unittest tests.test_rag_v2 -v
```

### 方式 2：运行集成测试

```bash
cd /Users/jafekin/Codes/Python\ Projects/font
python tests/test_rag_integration.py
```

### 方式 3：运行所有测试

```bash
cd /Users/jafekin/Codes/Python\ Projects/font
python -m unittest discover -s tests -p "test_*.py" -v
```

## 📊 测试覆盖范围

### embeddings.py 模块

| 测试项 | 状态 |
|-------|------|
| 导入检查 | ✅ |
| `get_image_embedding()` 函数 | ✅ |
| `get_text_embedding()` 函数 | ✅ |
| `batch_image_embeddings()` 函数 | ✅ |
| `batch_text_embeddings()` 函数 | ✅ |
| 文件不存在错误 | ✅ |
| 空字符串错误 | ✅ |
| 空列表错误 | ✅ |

### retriever.py 模块

| 测试项 | 状态 |
|-------|------|
| `TxtaiRetriever` 类 | ✅ |
| `FaissRetriever` 别名 | ✅ |
| `search()` 方法 | ✅ |
| `search_by_vector()` 方法 | ✅ |
| `batchsearch()` 方法 | ✅ |
| 向后兼容性 | ✅ |

### pipeline.py 模块

| 测试项 | 状态 |
|-------|------|
| `RAGPipeline` 类 | ✅ |
| `run()` 方法 | ✅ |
| `search_similar()` 方法 | ✅ |
| `batch_analyze()` 方法 | ✅ |

### prompt.py 模块

| 测试项 | 状态 |
|-------|------|
| `PROMPT_TEXT` 模板 | ✅ |
| `get_prompt()` 函数 | ✅ |
| 古籍元数据字段 | ✅ |
| 文献类型 | ✅ |
| 文种分类 | ✅ |
| 破损等级 | ✅ |

## 🔍 详细测试说明

### TestEmbeddings 类

```python
test_imports()                           # 检查所有函数导入
test_get_image_embedding_file_not_found()  # 测试文件不存在错误
test_get_text_embedding_empty_string()     # 测试空字符串错误
test_get_text_embedding_whitespace_only()  # 测试空格字符串错误
test_batch_image_embeddings_empty_list()   # 测试空列表错误
test_batch_text_embeddings_empty_list()    # 测试空列表错误
```

### TestRetriever 类

```python
test_imports()                           # 检查类导入
test_backward_compatibility()            # 测试 FaissRetriever 兼容性
test_txtai_retriever_methods()           # 检查方法存在
```

### TestPipeline 类

```python
test_imports()                           # 检查类导入
test_pipeline_methods()                  # 检查方法存在
```

### TestPrompt 类

```python
test_prompt_text_exists()                # 检查提示词模板存在
test_prompt_text_contains_fields()       # 检查必需字段
test_get_prompt_function_exists()        # 检查函数存在
test_get_prompt_format()                 # 检查函数输出
```

### TestIntegration 类

```python
test_all_modules_importable()            # 所有模块可导入
test_no_circular_imports()               # 无循环导入
test_type_hints()                        # 检查类型提示
```

### TestDocumentation 类

```python
test_doc_files_exist()                   # 检查文档文件
test_embeddings_docstring()              # 检查模块文档
test_function_docstrings()               # 检查函数文档
```

## 📈 测试结果示例

### 单元测试输出

```
test_batch_image_embeddings_empty_list (test.test_rag_v2.TestEmbeddings) ... ok
test_batch_text_embeddings_empty_list (test.test_rag_v2.TestEmbeddings) ... ok
test_get_image_embedding_file_not_found (test.test_rag_v2.TestEmbeddings) ... ok
test_get_text_embedding_empty_string (test.test_rag_v2.TestEmbeddings) ... ok
test_get_text_embedding_whitespace_only (test.test_rag_v2.TestEmbeddings) ... ok
test_imports (test.test_rag_v2.TestEmbeddings) ... ok
...
----------------------------------------------------------------------
Ran 35 tests in 0.123s

OK
```

### 集成测试输出

```
============================================================
📊 测试向量化模块 (embeddings.py)
============================================================

✓ 测试单个文本向量...
  • 输入: '甲骨文'
  • 向量维度: 512
  • 向量样本: [0.123, -0.456, 0.789, ...]...

✓ 测试批量文本向量...
  • 输入数量: 3
  • 输出向量数: 3
  • 每个向量维度: 512

✅ 向量化模块测试通过

============================================================
📊 测试结果总结
============================================================

✅ PASS - 模块导入
✅ PASS - 类和方法
✅ PASS - 向后兼容性
✅ PASS - 提示词模块
✅ PASS - 向量化模块
✅ PASS - 文档文件

总体: 6/6 测试通过
============================================================

🎉 所有测试通过！RAG v2.0 已就绪。
```

## 🐛 故障排查

### 问题 1：导入错误

```
ModuleNotFoundError: No module named 'rag'
```

**解决**：
```bash
# 确保在项目根目录运行
cd /Users/jafekin/Codes/Python\ Projects/font

# 或显式指定 Python 路径
PYTHONPATH=/Users/jafekin/Codes/Python\ Projects/font python tests/test_rag_integration.py
```

### 问题 2：txtai 导入失败

```
ImportError: No module named 'txtai'
```

**解决**：
- 检查 `thirdparty/txtai` 是否存在
- 或安装 pip 版本：`pip install txtai`

### 问题 3：测试超时

某些测试可能首次运行时较慢（下载模型）：
- 首次运行需要下载 CLIP 模型，请耐心等待
- 后续运行会使用缓存

## 💡 扩展和自定义

### 添加新的单元测试

在 `test_rag_v2.py` 中添加：

```python
class TestNewFeature(unittest.TestCase):
    """新功能单元测试"""
    
    def test_something(self):
        """测试说明"""
        # 测试代码
        self.assertTrue(condition)
```

### 添加新的集成测试

在 `test_rag_integration.py` 中添加：

```python
def test_new_functionality():
    """新功能集成测试"""
    print("\n✓ 测试新功能...")
    # 测试代码
    print("✅ 新功能测试通过")
    return True
```

## 📝 最佳实践

1. **定期运行测试** - 在每次修改代码后运行测试
2. **使用 -v 标志** - 获取详细输出便于调试
3. **隔离测试** - 每个测试应该独立，不依赖其他测试
4. **检查输出** - 关注警告和错误消息

## 📚 相关文档

- [RAG v2.0 完整文档](../rag/README_NEW.md)
- [迁移指南](../rag/MIGRATION_GUIDE.md)
- [实现总结](../rag/IMPLEMENTATION_SUMMARY.md)

## 🎯 下一步

1. ✅ 运行单元测试验证基本功能
2. ✅ 运行集成测试验证整体架构
3. 📋 在 Django 应用中进行端到端测试
4. 📈 性能测试和基准测试
5. 🚀 生产部署

---

**更新日期**: 2025-11-13  
**版本**: v2.0.0  
**状态**: ✅ 测试通过

