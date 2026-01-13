# -*- coding: utf-8 -*-
"""
路径工具模块 - 支持环境变量替换

功能：
    - 支持在 YAML 配置文件中使用环境变量占位符
    - 兼容 Docker 容器（DATA_ROOT=/data）和 Windows 本地环境

支持格式：
    - ${VAR}           - 使用环境变量 VAR，未设置时使用空字符串
    - ${VAR:default}   - 使用环境变量 VAR，未设置时使用 default

示例：
    data_root: "${DATA_ROOT:/mnt/d/DL_LEARN/Dataset/Processdata_HIL&REAL}"
    
    - Docker 容器中（DATA_ROOT=/data）：解析为 "/data"
    - Windows 未设置环境变量时：解析为 "/mnt/d/DL_LEARN/Dataset/Processdata_HIL&REAL"

作者：Auto-generated
日期：2026-01-10
"""
import os
import re
from typing import Any, Dict, List, Union


# 环境变量占位符正则表达式
# 匹配格式：${VAR} 或 ${VAR:default_value}
ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')


def expand_env_vars(value: str) -> str:
    """
    展开字符串中的环境变量占位符
    
    参数:
        value: 包含 ${VAR} 或 ${VAR:default} 格式占位符的字符串
        
    返回:
        str: 替换后的字符串
        
    示例:
        >>> os.environ['DATA_ROOT'] = '/data'
        >>> expand_env_vars('${DATA_ROOT}/processed')
        '/data/processed'
        >>> expand_env_vars('${NOTSET:/default/path}')
        '/default/path'
    """
    def replace_match(match):
        var_name = match.group(1)      # 环境变量名称
        default = match.group(2)        # 默认值（可能为 None）
        
        # 优先使用环境变量值
        env_value = os.getenv(var_name)
        if env_value is not None:
            return env_value
        
        # 如果有默认值则使用默认值，否则返回空字符串
        if default is not None:
            return default
        
        return ''
    
    return ENV_VAR_PATTERN.sub(replace_match, value)


def process_config_paths(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    递归处理配置字典/列表中的所有字符串，展开环境变量占位符
    
    参数:
        config: 配置字典、列表、字符串或其他类型的值
        
    返回:
        处理后的配置（与输入类型相同）
        
    示例:
        >>> config = {'data': {'root': '${DATA_ROOT:/default}'}}
        >>> process_config_paths(config)
        {'data': {'root': '/default'}}  # 如果 DATA_ROOT 未设置
    """
    if isinstance(config, dict):
        # 递归处理字典的每个值
        return {key: process_config_paths(value) for key, value in config.items()}
    elif isinstance(config, list):
        # 递归处理列表的每个元素
        return [process_config_paths(item) for item in config]
    elif isinstance(config, str):
        # 字符串类型：展开环境变量
        return expand_env_vars(config)
    else:
        # 其他类型（int, float, bool, None 等）：原样返回
        return config


def get_data_root(default: str = '/data') -> str:
    """
    获取数据根目录，优先使用环境变量
    
    参数:
        default: 默认路径（当 DATA_ROOT 环境变量未设置时使用）
        
    返回:
        str: 数据根目录路径
        
    说明:
        - Docker 容器中：设置 DATA_ROOT=/data，返回 "/data"
        - Windows/WSL：未设置环境变量时返回默认路径
    """
    return os.getenv('DATA_ROOT', default)


# ============================================================
# 单元测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("路径工具模块测试")
    print("=" * 60)
    
    # 测试 1: 基本环境变量替换
    print("\n测试 1: 基本环境变量替换")
    os.environ['TEST_VAR'] = '/custom/test/path'
    result = expand_env_vars('${TEST_VAR}')
    assert result == '/custom/test/path', f"期望 '/custom/test/path'，实际 '{result}'"
    print(f"  ${'{TEST_VAR}'} -> '{result}' ✓")
    
    # 测试 2: 带子路径的替换
    result = expand_env_vars('${TEST_VAR}/subdir/file.csv')
    assert result == '/custom/test/path/subdir/file.csv', f"替换失败: {result}"
    print(f"  ${'{TEST_VAR}'}/subdir/file.csv -> '{result}' ✓")
    
    # 测试 3: 使用默认值（环境变量未设置）
    print("\n测试 2: 默认值回退")
    result = expand_env_vars('${UNDEFINED_VAR:/fallback/path}')
    assert result == '/fallback/path', f"期望 '/fallback/path'，实际 '{result}'"
    print(f"  ${'{UNDEFINED_VAR:/fallback/path}'} -> '{result}' ✓")
    
    # 测试 4: 环境变量覆盖默认值
    print("\n测试 3: 环境变量覆盖默认值")
    os.environ['DATA_ROOT'] = '/data'
    result = expand_env_vars('${DATA_ROOT:/mnt/d/default}')
    assert result == '/data', f"期望 '/data'，实际 '{result}'"
    print(f"  DATA_ROOT=/data 设置后: ${'{DATA_ROOT:/mnt/d/default}'} -> '{result}' ✓")
    
    # 测试 5: 递归配置处理
    print("\n测试 4: 递归配置处理")
    test_config = {
        'data': {
            'data_root': '${DATA_ROOT:/default}',
            'processed_dir': '${DATA_ROOT:/default}/processed',
            'num_value': 42,
            'bool_value': True,
            'list_value': ['${DATA_ROOT}/a', '${DATA_ROOT}/b']
        }
    }
    processed = process_config_paths(test_config)
    assert processed['data']['data_root'] == '/data', "data_root 解析失败"
    assert processed['data']['processed_dir'] == '/data/processed', "processed_dir 解析失败"
    assert processed['data']['num_value'] == 42, "数值类型被修改"
    assert processed['data']['bool_value'] == True, "布尔类型被修改"
    assert processed['data']['list_value'] == ['/data/a', '/data/b'], "列表处理失败"
    print(f"  配置字典递归处理 ✓")
    
    # 清理测试环境变量
    del os.environ['TEST_VAR']
    del os.environ['DATA_ROOT']
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
