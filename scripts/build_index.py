"""
使用txtai构建图片+文字信息的RAG知识库索引

该脚本会：
1. 扫描指定目录下的图片文件
2. 读取每张图片对应的文字信息（从JSON文件或元数据）
3. 使用txtai构建联合索引，支持图片和文字的统一检索

使用方法：
    python scripts/build_index.py --image-dir media/uploads --index-path rag/index
"""
import os
import sys
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

# 修复macOS上的OpenMP库冲突问题
# 如果多个库都链接了OpenMP，需要设置这个环境变量
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 允许txtai保存pickle数据（用于保存索引配置）
# 这是安全的，因为我们处理的是本地数据
if 'ALLOW_PICKLE' not in os.environ:
    os.environ['ALLOW_PICKLE'] = 'True'

# 添加项目根目录到路径，以便导入rag模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from txtai.embeddings import Embeddings
    TXTAI_AVAILABLE = True
except ImportError:
    TXTAI_AVAILABLE = False
    print("警告: txtai未安装，将使用假模式")

# 检查sentence-transformers是否可用
try:
    import sentence_transformers  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 配置
# 默认使用CLIP模型，支持图片和文本的统一向量空间
# 注意：使用clip-ViT-B-32（如txtai示例所示），而不是clip-vit-base-patch32
MODEL_NAME = "clip-ViT-B-32"  # CLIP模型，支持图片和文本
# MODEL_NAME = "model/saved_model"


def ensure_model_assets(model_dir: Path) -> None:
    """确保本地模型目录包含sentence-transformers加载所需的资产。"""
    required_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "preprocessor_config.json"
    ]

    parent_dir = model_dir.parent
    for filename in required_files:
        target = model_dir / filename
        if target.exists():
            continue

        candidate_sources = [
            parent_dir / filename,
            parent_dir / "saved_model" / filename
        ]

        for source in candidate_sources:
            if source.exists():
                try:
                    shutil.copy2(source, target)
                    print(f"缺失的模型文件 {filename} 已从 {source} 复制至 {target}")
                except Exception as copy_error:
                    print(f"警告: 无法复制模型文件 {filename}: {copy_error}")
                break
        else:
            print(f"警告: 本地模型缺少必要文件 {filename}，请确保模型导出完整。")

    # 修正CLIP配置缺失的hidden_size字段
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as cfg_file:
                config_data = json.load(cfg_file)

            if "hidden_size" not in config_data:
                text_cfg = config_data.get("text_config", {})
                inferred_hidden_size = text_cfg.get("hidden_size")
                if inferred_hidden_size:
                    config_data["hidden_size"] = inferred_hidden_size
                    with open(config_path, "w", encoding="utf-8") as cfg_file:
                        json.dump(config_data, cfg_file,
                                  ensure_ascii=False, indent=2)
                    print("missing hidden_size 已根据 text_config.hidden_size 补全")
                else:
                    print("警告: 无法从 text_config 推断 hidden_size，请检查模型配置。")
        except Exception as config_error:
            print(f"警告: 无法修复模型配置 {config_path}: {config_error}")


def ensure_sentence_transformer_dir(model_dir: Path) -> Path:
    """确保本地CLIP模型可以被sentence-transformers识别，返回转换后的目录。"""
    st_dir = model_dir / "sentence_transformer"
    modules_file = st_dir / "modules.json"

    if modules_file.exists():
        return st_dir

    try:
        from sentence_transformers import SentenceTransformer, models  # type: ignore
    except ImportError as import_error:  # pragma: no cover - 已在前面检查
        raise RuntimeError("需要sentence-transformers库以加载本地模型") from import_error

    word_embedding_model = models.Transformer(str(model_dir))
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )

    st_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model])
    st_model.save(str(st_dir))
    print(f"已将本地模型转换为sentence-transformers格式: {st_dir}")

    return st_dir


def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    将metadata中的嵌套字典和列表转换为字符串，确保所有值都是基本类型

    Args:
        metadata: 原始metadata字典

    Returns:
        扁平化的metadata字典，所有值都是基本类型
    """
    flattened = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            # 将字典转换为JSON字符串
            flattened[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, list):
            # 将列表转换为JSON字符串
            flattened[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            # None值转换为空字符串
            flattened[key] = ''
        else:
            # 其他基本类型直接使用
            flattened[key] = value
    return flattened


def _generate_key_variants(value: str, base_dir: Optional[Path] = None) -> List[str]:
    """生成用于匹配的多种路径表示。"""
    if not value:
        return []

    variants = set()
    raw = str(value).replace('\\', '/').strip()
    if not raw:
        return []

    variants.add(raw)
    variants.add(raw.lstrip('./'))
    variants.add(Path(raw).name)

    try:
        as_path = Path(raw)
        if base_dir and as_path.is_absolute():
            try:
                rel = as_path.relative_to(base_dir)
                variants.add(str(rel).replace('\\', '/'))
            except ValueError:
                pass
    except Exception:
        pass

    if base_dir:
        base_str = str(base_dir).replace('\\', '/')
        if raw.startswith(base_str):
            variants.add(raw[len(base_str):].lstrip('/'))

    return [variant for variant in variants if variant]


def build_json_lookup(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """扫描目录，构建图片路径到JSON数据的映射。"""
    lookup: Dict[str, Dict[str, Any]] = {}
    if not base_dir.exists():
        return lookup

    for json_path in base_dir.rglob('*.json'):
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
        except Exception:
            continue

        keys: List[str] = []
        # JSON文件自身的各种表示
        keys.extend(_generate_key_variants(str(json_path), base_dir))
        keys.extend(_generate_key_variants(
            str(json_path.relative_to(base_dir)), base_dir))
        keys.append(json_path.name)

        path_info = data.get('path_info', {}) if isinstance(data, dict) else {}
        for field in ('original_path', 'original_relative_path', 'output_path', 'output_filename', 'original_filename'):
            field_value = path_info.get(field)
            if field_value:
                if isinstance(field_value, list):
                    for item in field_value:
                        keys.extend(_generate_key_variants(
                            str(item), base_dir))
                else:
                    keys.extend(_generate_key_variants(
                        str(field_value), base_dir))

        for key in keys:
            normalized = key.replace('\\', '/').strip().lstrip('./')
            if normalized:
                lookup[normalized] = data

    return lookup


def build_default_text_info(image_path: Path, metadata: Dict[str, Any]) -> str:
    """根据已有元数据构建兜底的文字描述。"""
    parts: List[str] = [f"图片文件: {image_path.name}"]

    title = metadata.get('title')
    if title:
        parts.append(f"标题: {title}")

    directory = metadata.get('directory')
    if directory:
        parts.append(f"目录: {directory}")

    category = metadata.get('category')
    if category:
        parts.append(f"类别: {category}")

    catalog_number = metadata.get('catalog_number')
    if catalog_number:
        parts.append(f"编号: {catalog_number}")

    location = metadata.get('current_location')
    if location:
        parts.append(f"现藏: {location}")

    return "。".join(parts) + "。"


def _normalize_value(value: Any) -> str:
    """将不同类型的值转换为可读字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = [str(v).strip() for v in value if v]
        return "、".join(filter(None, parts))
    if isinstance(value, dict):
        parts = [f"{k}:{v}" for k, v in value.items() if v]
        return "、".join(parts)
    return str(value).strip()


def extract_text_info_from_json(json_data: Dict[str, Any]) -> str:
    """从JSON数据中提取有意义的文字信息。"""

    def push(label: str, value: Any) -> None:
        normalized = _normalize_value(value)
        if normalized:
            parts.append(f"{label}{normalized}")

    parts: List[str] = []

    title_block = json_data.get('title', {})
    push("标题: ", title_block.get('title_text'))

    author_editors = json_data.get('author_and_editors', {})
    push("作者: ", author_editors.get('author'))
    push("编者: ", author_editors.get('editor_commentator'))

    edition_info = json_data.get('edition_information', {})
    push("版本: ", [edition_info.get('edition_type'),
                  edition_info.get('edition_style'),
                  edition_info.get('publication_period')])
    push("出版者: ", edition_info.get('publisher'))

    doc_metadata = json_data.get('document_metadata', {})
    push("分类: ", doc_metadata.get('classification', {}).get('value'))
    push("语言: ", doc_metadata.get('language', {}).get('value'))
    push("文献类型: ", doc_metadata.get('document_type', {}).get('value'))

    collection = json_data.get('collection_and_provenance', {})
    current_location = collection.get('current_location')
    if current_location:
        location_match = re.search(
            r'([^/]+(?:图书馆|博物院|书店))', str(current_location))
        push("现藏: ", location_match.group(1)
             if location_match else current_location)
    push("旧藏: ", collection.get('previous_owner'))

    path_info = json_data.get('path_info', {})
    push("类别: ", path_info.get('category'))
    push("编号: ", path_info.get('catalog_number'))
    push("卷册: ", path_info.get('volume'))

    notes = json_data.get('notes') or json_data.get('note')
    push("备注: ", notes)

    filtered = [segment for segment in parts if segment]
    if filtered:
        return "。".join(filtered) + "。"
    return json.dumps(json_data, ensure_ascii=False)


def find_image_text_pairs(
    image_dir: str,
    text_info_dir: Optional[str] = None,
    text_file_extensions: tuple = ('.json', '.txt', '.md')
) -> List[Dict[str, Any]]:
    """
    查找图片和对应的文字信息对

    Args:
        image_dir: 图片目录
        text_info_dir: 文字信息目录（如果为None，则在图片同目录查找）
        text_file_extensions: 文字信息文件扩展名

    Returns:
        包含图片路径和文字信息的字典列表
    """
    pairs = []
    image_dir_path = Path(image_dir)

    # 支持的图片格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

    for image_path in image_dir_path.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            # 查找对应的文字信息文件
            text_info = None
            text_path = None
            json_data = None

            # 策略1: 在同目录查找同名但不同扩展名的文件
            for ext in text_file_extensions:
                candidate = image_path.with_suffix(ext)
                if candidate.exists():
                    text_path = candidate
                    break

            # 策略2: 如果指定了text_info_dir，在那里查找
            if text_path is None and text_info_dir:
                text_info_path = Path(text_info_dir) / \
                    image_path.relative_to(image_dir_path)
                for ext in text_file_extensions:
                    candidate = text_info_path.with_suffix(ext)
                    if candidate.exists():
                        text_path = candidate
                        break

            # 读取文字信息
            if text_path and text_path.exists():
                try:
                    if text_path.suffix == '.json':
                        with open(text_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            # 从JSON中提取有意义的文字信息
                            if isinstance(json_data, dict):
                                text_info = extract_text_info_from_json(
                                    json_data)
                            else:
                                text_info = str(json_data)
                    else:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            text_info = f.read()
                except Exception as e:
                    print(f"警告: 无法读取文字信息文件 {text_path}: {e}")
                    text_info = None
                    json_data = None

            # 构建元数据
            metadata = {
                'filename': image_path.name,
                'directory': str(image_path.parent.relative_to(image_dir_path)),
                'has_text_info': text_path is not None
            }

            # 如果读取了JSON数据，添加更多元数据
            if json_data:
                path_info = json_data.get('path_info', {})
                doc_metadata = json_data.get('document_metadata', {})
                title_info = json_data.get('title', {})
                edition_info = json_data.get('edition_information', {})
                collection_info = json_data.get(
                    'collection_and_provenance', {})

                # 安全地提取嵌套字典中的值
                document_type_obj = doc_metadata.get('document_type', {})
                document_type_value = document_type_obj.get('value', '') if isinstance(
                    document_type_obj, dict) else str(document_type_obj) if document_type_obj else ''

            json_lookup: Dict[str, Dict[str, Any]] = {}
            json_lookup_base = Path(
                text_info_dir) if text_info_dir else image_dir_path
            if json_lookup_base.exists():
                json_lookup = build_json_lookup(json_lookup_base)
                language_obj = doc_metadata.get('language', {})
                language_value = language_obj.get('value', '') if isinstance(
                    language_obj, dict) else str(language_obj) if language_obj else ''

                metadata.update({
                    'category': path_info.get('category', ''),
                    'catalog_number': path_info.get('catalog_number', ''),
                    'title': title_info.get('title_text', ''),
                    'edition_type': edition_info.get('edition_type', ''),
                    'current_location': collection_info.get('current_location', ''),
                    'document_type': document_type_value,
                    'language': language_value
                })

            if not text_info or not text_info.strip():
                text_info = build_default_text_info(image_path, metadata)

            pairs.append({
                'image_path': str(image_path),
                'text_info': text_info,
                'id': str(image_path.relative_to(image_dir_path)),
                'metadata': metadata,
                'json_data': json_data  # 保存完整的JSON数据，以便后续使用
            })

    return pairs


def build_txtai_index(
    pairs: List[Dict[str, Any]],
    index_path: str,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
    separate_text_docs: bool = False
) -> None:
    """
    使用txtai构建图片+文字信息的联合索引

    Args:
        pairs: 图片和文字信息对列表
        index_path: 索引保存路径
        model_name: 嵌入模型名称
        batch_size: 批处理大小
        separate_text_docs: 是否为文字信息创建单独文档（默认False，保证每对仅生成一个文档）
    """
    if not TXTAI_AVAILABLE:
        print("错误: txtai未安装，无法构建索引")
        return

    if not pairs:
        print("错误: 没有找到图片和文字信息对")
        return

    print(f"开始构建索引，共 {len(pairs)} 个图片-文字对...")

    # 初始化txtai Embeddings
    # 使用CLIP模型，支持图片和文本的统一向量空间
    # 注意：如果使用CLIP模型，需要安装sentence-transformers: pip install sentence-transformers
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("错误: sentence-transformers未安装")
        print("提示: 要使用CLIP模型处理图片，请先安装: pip install sentence-transformers")
        raise ImportError(
            "sentence-transformers is required for CLIP models. Install it with: pip install sentence-transformers")

    # 检查模型路径
    # 如果是本地路径，检查是否存在
    actual_model_name = model_name
    if os.path.exists(model_name) and os.path.isdir(model_name):
        # 本地路径存在，使用它
        print(f"使用本地模型: {model_name}")
        ensure_model_assets(Path(model_name))
        actual_model_name = str(
            ensure_sentence_transformer_dir(Path(model_name)))
    elif not os.path.exists(model_name) and (model_name.startswith('./') or model_name.startswith('../') or '/' in model_name):
        raise FileNotFoundError(f"指定的模型路径不存在: {model_name}")
    else:
        # Hugging Face模型名称，直接使用
        print(f"使用模型: {actual_model_name}")

    def create_embeddings(model_path: str) -> Embeddings:
        """构建Embeddings实例，失败时向上传播异常。"""
        return Embeddings({
            "method": "sentence-transformers",
            "path": model_path,
            "content": True,
            "gpu": True,
            "format": "numpy"
        })

    try:
        # 使用sentence-transformers方法（支持CLIP模型处理图片）
        embeddings = create_embeddings(actual_model_name)
    except Exception as exc:
        raise RuntimeError(f"无法加载嵌入模型 {actual_model_name}: {exc}") from exc

    # 准备索引数据
    # txtai的索引格式: (id, data, tags)
    # - id: 唯一记录ID
    # - data: 可以是文本、字典或对象（这里用字典以便同时上传图片对象和文本）
    # - tags: 任意可序列化的元数据字典
    documents = []
    metadata_store: Dict[str, Dict[str, Any]] = {}

    for pair in pairs:
        image_path = pair['image_path']
        text_info = pair['text_info']
        doc_id = pair['id']
        metadata = pair['metadata']

        try:
            # 加载图片为PIL.Image对象
            image_obj = Image.open(image_path)

            # 使用字典格式的data，将图片对象和metadata都放在字典中
            # 确保metadata中的所有值都是基本类型（SQLite不支持字典/列表）
            clean_metadata = flatten_metadata({
                **metadata,
                'text_info': text_info,  # 文字信息
                'image_path': str(image_path),  # 保存原始路径
                'type': 'image'
            })

            # 构建data字典：包含object（图片）和所有metadata字段
            data_dict = {
                "object": image_obj,
                "text": text_info
            }

            documents.append((
                doc_id,  # id
                data_dict,  # data: 字典格式，包含object和metadata
                None
            ))
            metadata_store[doc_id] = clean_metadata
        except Exception as e:
            print(f"警告: 无法加载图片 {image_path}: {e}")
            continue

        # 可选：为文字信息创建单独文档（默认关闭）
        if separate_text_docs and text_info and len(text_info.strip()) > 0:
            text_doc_id = f"{doc_id}_text"
            # 确保metadata中的所有值都是基本类型
            clean_text_metadata = flatten_metadata({
                **metadata,
                'image_path': str(image_path),
                'type': 'text',
                'related_image_id': doc_id
            })

            documents.append((
                text_doc_id,
                {
                    "text": text_info
                },
                None
            ))
            metadata_store[text_doc_id] = clean_text_metadata

    print(f"准备索引 {len(documents)} 个文档...")

    # 批量索引
    try:
        # txtai会自动批处理
        embeddings.index(documents)
        print(f"索引构建完成，共索引 {len(documents)} 个文档")

        # 保存索引
        os.makedirs(index_path, exist_ok=True)
        embeddings.save(index_path)
        print(f"索引已保存到: {index_path}")

        try:
            metadata_path = Path(index_path) / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata_store, meta_file,
                          ensure_ascii=False, indent=2)
            print(f"索引元数据已保存到: {metadata_path}")
        except Exception as meta_error:
            print(f"警告: 无法保存元数据文件: {meta_error}")

    except Exception as e:
        print(f"错误: 构建索引时出错: {e}")
        raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用txtai构建图片+文字信息的RAG知识库索引')
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data',
        help='图片目录路径（默认: data）'
    )
    parser.add_argument(
        '--text-info-dir',
        type=str,
        default=None,
        help='文字信息目录路径（可选，如果为None则在图片同目录查找）'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='rag/index',
        help='索引保存路径（默认: rag/index）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_NAME,
        help=f'嵌入模型名称（默认: {MODEL_NAME}）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批处理大小（默认: 32）'
    )
    parser.add_argument(
        '--separate-text-docs',
        action='store_true',
        help='为文字信息创建单独文档（默认关闭：每对仅生成一个文档）'
    )

    args = parser.parse_args()

    # 检查图片目录
    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录不存在: {args.image_dir}")
        return

    # 查找图片和文字信息对
    print(f"扫描图片目录: {args.image_dir}")
    pairs = find_image_text_pairs(args.image_dir, args.text_info_dir)

    if not pairs:
        print("警告: 没有找到任何图片文件")
        return

    print(f"找到 {len(pairs)} 个图片文件")

    # 构建索引
    build_txtai_index(
        pairs=pairs,
        index_path=args.index_path,
        model_name=args.model,
        batch_size=args.batch_size,
        separate_text_docs=args.separate_text_docs
    )

    print("完成！")


if __name__ == "__main__":
    main()
