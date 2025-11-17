# 史记图片处理脚本使用说明

## 功能

`process_shiji_images.py` 脚本用于处理史记数据集中的图片：

1. **遍历所有图片**：扫描指定目录下的所有图片文件
2. **解析目录信息**：从目录路径中提取版本、藏馆、作者等信息
3. **重命名图片**：生成有意义的文件名（格式：`史记_{版本类型}_{编号}_{序号}_{hash}.jpg`）
4. **生成JSON元数据**：为每个图片生成对应的JSON文件，包含：
   - 原始路径信息
   - 从目录解析的版本、藏馆等信息
   - 符合prompt.py中定义的所有字段结构
5. **统一输出**：将所有处理后的图片和JSON文件复制到单一的输出目录

## 使用方法

### 基本用法

```bash
python scripts/process_shiji_images.py \
    --data-dir "data/名录 史记2025-11-6" \
    --output-dir "data/processed_shiji"
```

### 参数说明

- `--data-dir`: 数据目录路径（默认: `data/名录 史记2025-11-6`）
- `--output-dir`: **必须指定**，所有处理后的文件将复制到此目录
- `--no-rename`: 不重命名图片，只生成JSON文件（使用原文件名）
- `--no-backup`: 不备份原文件

### 示例

#### 1. 处理并重命名所有图片

```bash
python scripts/process_shiji_images.py \
    --data-dir "data/名录 史记2025-11-6" \
    --output-dir "data/processed_shiji"
```

#### 2. 不重命名，只生成JSON文件

```bash
python scripts/process_shiji_images.py \
    --data-dir "data/名录 史记2025-11-6" \
    --output-dir "data/processed_shiji" \
    --no-rename
```

#### 3. 不备份原文件

```bash
python scripts/process_shiji_images.py \
    --data-dir "data/名录 史记2025-11-6" \
    --output-dir "data/processed_shiji" \
    --no-backup
```

## 输出结果

### 文件结构

处理完成后，所有文件将统一放在输出目录中：

```
data/processed_shiji/
├── 史记_北宋刻本_00393_0001_a1b2c3.jpg
├── 史记_北宋刻本_00393_0001_a1b2c3.json
├── 史记_北宋刻本_00393_0002_d4e5f6.jpg
├── 史记_北宋刻本_00393_0002_d4e5f6.json
├── ...
```

### JSON文件结构

每个JSON文件包含完整的元数据信息，主要字段包括：

```json
{
  "path_info": {
    "original_path": "原始路径",
    "original_directory": "原始目录",
    "original_filename": "原始文件名",
    "output_path": "输出路径",
    "output_filename": "输出文件名",
    "category": "集解本",
    "catalog_number": "【1】00393"
  },
  "document_metadata": {
    "document_type": {"value": "汉文古籍", "confidence": 1.0},
    "language": {"value": "汉文", "confidence": 1.0},
    "classification": {"value": "史部-紀傳類-通代之屬", "confidence": 1.0}
  },
  "title": {
    "title_text": "史记一百三十卷",
    "confidence": 1.0
  },
  "author_and_editors": {
    "author": "（汉）司马迁撰",
    "editor_commentator": ["（南朝宋）裴骃集解"],
    "confidence": 1.0
  },
  "edition_information": {
    "edition_type": "北宋刻本",
    "publication_period": "北宋",
    "confidence": 0.8
  },
  "collection_and_provenance": {
    "current_location": "北京大学图书馆",
    "confidence": 1.0
  },
  ...
}
```

## 文件名格式

重命名后的文件名格式：

```
史记_{版本类型}_{编号}_{序号}_{原文件名hash}.jpg
```

示例：
- `史记_北宋刻本_00393_0001_a1b2c3.jpg`
- `史记_明正德十三年_03474_0008_d4e5f6.jpg`

### 文件名组成部分

1. **版本类型**：从目录路径解析的版本信息（如"北宋刻本"、"明正德十三年"）
2. **编号**：从目录名提取的编号（如"00393"、"03474"）
3. **序号**：处理顺序（4位数字，从0001开始）
4. **原文件名hash**：原文件名的MD5哈希值前6位，确保唯一性

## 注意事项

1. **输出目录必须指定**：所有处理后的文件将复制到指定的输出目录
2. **原文件保留**：原文件不会被删除，如果需要备份，会在原目录创建`.backup_`开头的备份文件
3. **文件名唯一性**：如果生成的文件名重复，会自动添加序号后缀
4. **JSON文件**：每个图片都会生成对应的JSON文件，文件名与图片文件名相同（扩展名不同）

## 处理流程

1. 扫描数据目录，找到所有图片文件
2. 对每个图片：
   - 解析所在目录的信息（版本、藏馆等）
   - 生成新的文件名（如果启用重命名）
   - 生成完整的JSON元数据
   - 复制图片到输出目录
   - 保存JSON文件到输出目录
   - 在原目录创建备份（如果启用）
3. 显示处理统计信息

## 后续使用

处理完成后，可以使用 `build_index.py` 脚本构建RAG索引：

```bash
python scripts/build_index.py \
    --image-dir "data/processed_shiji" \
    --index-path "rag/index"
```

由于所有文件都在单一目录中，且每个图片都有对应的JSON文件，`build_index.py` 会自动读取JSON文件中的文字信息。

