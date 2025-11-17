#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# 修复macOS上的OpenMP库冲突问题
# 如果多个库都链接了OpenMP，需要设置这个环境变量
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 允许txtai加载pickle数据（用于加载索引配置）
# 这是安全的，因为我们处理的是本地数据
if 'ALLOW_PICKLE' not in os.environ:
    os.environ['ALLOW_PICKLE'] = 'True'


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
