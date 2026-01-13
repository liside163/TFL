# -*- coding: utf-8 -*-
"""
配置加载器 - 统一的 YAML 配置加载模块

功能：
    - 加载 YAML 配置文件
    - 自动处理环境变量占位符 (${VAR:default} 格式)
    - 兼容 Docker 容器（DATA_ROOT=/data）和 Windows 本地环境

使用方法：
    from config_loader import load_config
    
    config = load_config('config/config.yaml')
    print(config['data']['data_root'])  # 自动替换为环境变量值

作者：Auto-generated
日期：2026-01-10
"""
import yaml
from typing import Any, Dict

# 导入路径工具模块
from path_utils import process_config_paths


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件并处理环境变量占位符
    
    参数:
        config_path: YAML 配置文件路径
        
    返回:
        Dict[str, Any]: 处理后的配置字典
        
    示例:
        >>> config = load_config('config/config.yaml')
        >>> print(config['data']['data_root'])
        '/data'  # 如果 DATA_ROOT=/data
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理配置中的环境变量占位符
    # 例如：${DATA_ROOT:/mnt/d/DL_LEARN/Dataset} 会被替换为实际路径
    return process_config_paths(config)


def load_config_with_override(config_path: str, override_config_path: str = None) -> Dict[str, Any]:
    """
    加载主配置文件，可选地用另一个配置文件覆盖部分参数
    
    参数:
        config_path: 主配置文件路径
        override_config_path: 覆盖配置文件路径（可选）
        
    返回:
        Dict[str, Any]: 合并后的配置字典
    """
    config = load_config(config_path)
    
    if override_config_path:
        override_config = load_config(override_config_path)
        config = deep_merge(config, override_config)
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典，override 中的值覆盖 base 中的同名键
    
    参数:
        base: 基础配置字典
        override: 覆盖配置字典
        
    返回:
        Dict[str, Any]: 合并后的配置字典
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    
    return result


# ============================================================
# 单元测试
# ============================================================
if __name__ == '__main__':
    import os
    import tempfile
    
    print("=" * 60)
    print("配置加载器模块测试")
    print("=" * 60)
    
    # 创建临时测试配置文件
    test_yaml_content = '''
data:
  data_root: "${DATA_ROOT:/default/path}"
  processed_dir: "${DATA_ROOT:/default/path}/processed"
  batch_size: 32

model:
  name: "dann"
  hidden_dim: 128
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(test_yaml_content)
        temp_path = f.name
    
    try:
        # 测试 1: 使用默认值（环境变量未设置）
        print("\n测试 1: 使用默认值")
        if 'DATA_ROOT' in os.environ:
            del os.environ['DATA_ROOT']
        config = load_config(temp_path)
        assert config['data']['data_root'] == '/default/path', f"期望 '/default/path'，实际 '{config['data']['data_root']}'"
        assert config['data']['processed_dir'] == '/default/path/processed'
        print(f"  data_root: '{config['data']['data_root']}' ✓")
        print(f"  processed_dir: '{config['data']['processed_dir']}' ✓")
        
        # 测试 2: 环境变量覆盖
        print("\n测试 2: 环境变量覆盖")
        os.environ['DATA_ROOT'] = '/data'
        config = load_config(temp_path)
        assert config['data']['data_root'] == '/data', f"期望 '/data'，实际 '{config['data']['data_root']}'"
        assert config['data']['processed_dir'] == '/data/processed'
        print(f"  DATA_ROOT=/data 设置后:")
        print(f"  data_root: '{config['data']['data_root']}' ✓")
        print(f"  processed_dir: '{config['data']['processed_dir']}' ✓")
        
        # 测试 3: 非字符串类型保持不变
        print("\n测试 3: 非字符串类型保持不变")
        assert config['data']['batch_size'] == 32, "batch_size 应保持为整数"
        assert config['model']['hidden_dim'] == 128, "hidden_dim 应保持为整数"
        print(f"  batch_size: {config['data']['batch_size']} (int) ✓")
        print(f"  hidden_dim: {config['model']['hidden_dim']} (int) ✓")
        
        del os.environ['DATA_ROOT']
        
    finally:
        os.unlink(temp_path)
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
