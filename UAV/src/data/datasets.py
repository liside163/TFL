"""
数据集抽象与注册表模块。

本模块定义了无人机故障诊断项目的通用数据集接口 `BaseUAVDataset`，
并实现了数据集注册机制，支持通过字符串名称动态创建数据集实例。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from torch.utils.data import Dataset

__all__ = [
    "DATASET_REGISTRY",
    "register_dataset",
    "create_dataset",
    "BaseUAVDataset",
    "SignalUAVDataset",
    "ImageUAVDataset",
]

# -----------------------------------------------------------------------------
# Registry Mechanism
# -----------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Type["BaseUAVDataset"]] = {}


def register_dataset(name: str) -> Callable:
    """
    类装饰器，用于将数据集类注册到全局注册表中。

    Args:
        name (str): 数据集的注册名称。

    Returns:
        Callable: 装饰器函数。
    """
    def decorator(cls: Type["BaseUAVDataset"]) -> Type["BaseUAVDataset"]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"数据集名称 '{name}' 已被注册: {DATASET_REGISTRY[name]}")
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def create_dataset(name: str, **kwargs) -> "BaseUAVDataset":
    """
    根据名称创建数据集实例。

    Args:
        name (str): 数据集名称（需先通过 @register_dataset 注册）。
        **kwargs: 传递给数据集构造函数的参数。

    Returns:
        BaseUAVDataset: 数据集实例。

    Raises:
        ValueError: 如果名称未注册。
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"数据集 '{name}' 未找到。可用数据集: {available}")
    return DATASET_REGISTRY[name](**kwargs)


# -----------------------------------------------------------------------------
# Base Dataset Abstraction
# -----------------------------------------------------------------------------

class BaseUAVDataset(Dataset):
    """
    无人机故障诊断数据集的基类。

    所有自定义数据集都应继承此类，并实现 `_load_raw` 方法。
    """

    def __init__(
        self,
        data_index: List[Tuple[str, int, Dict[str, Any]]],
        transform: Optional[Callable] = None,
        feature_extractor: Optional[Callable] = None,
    ):
        """
        初始化数据集。

        Args:
            data_index (List[Tuple[str, int, Dict[str, Any]]]):
                数据索引列表。每个元素包含：
                - path (str): 数据文件路径。
                - label (int): 故障类别标签。
                - meta_dict (Dict[str, Any]): 元数据（如采样率、工况、SNR等）。
            transform (Optional[Callable]):
                数据变换/增强函数。输入为 raw_data，输出为变换后的 data。
            feature_extractor (Optional[Callable]):
                特征提取器。输入为 (transformed) data，输出为特征 Tensor。
                如果为 None，则直接返回 data。
        """
        self.data_index = data_index
        self.transform = transform
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        """返回数据集样本数量。"""
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定索引的样本。

        Args:
            idx (int): 样本索引。

        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - "input": 模型输入 (Tensor 或 ndarray)。
                - "label": 标签 (int)。
                - "meta": 元数据 (dict)。
        """
        path, label, meta = self.data_index[idx]

        # 1. 加载原始数据 (由子类实现)
        raw_data = self._load_raw(path, meta)

        # 2. 数据增强 / 预处理
        if self.transform:
            raw_data = self.transform(raw_data)

        # 3. 特征提取 (可选)
        if self.feature_extractor:
            input_data = self.feature_extractor(raw_data)
        else:
            input_data = raw_data

        return {
            "input": input_data,
            "label": label,
            "meta": meta,
        }

    def _load_raw(self, path: str, meta: Dict[str, Any]) -> Any:
        """
        加载原始数据。

        Args:
            path (str): 文件路径。
            meta (Dict[str, Any]): 元数据。

        Returns:
            Any: 原始数据对象（如 numpy array, PIL Image 等）。

        Raises:
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError("子类必须实现 _load_raw 方法")


# -----------------------------------------------------------------------------
# Concrete Examples
# -----------------------------------------------------------------------------

@register_dataset("SignalUAVDataset")
class SignalUAVDataset(BaseUAVDataset):
    """
    用于 1D 信号（振动、电流、声音等）的数据集。
    """

    def _load_raw(self, path: str, meta: Dict[str, Any]) -> Any:
        """
        加载 1D 信号数据。

        说明:
            通常使用 numpy 或 scipy.io 读取 .npy, .mat, .csv 文件。

        示例代码 (伪代码):
            import numpy as np
            import scipy.io as sio

            if path.endswith('.npy'):
                return np.load(path)
            elif path.endswith('.mat'):
                mat = sio.loadmat(path)
                return mat['DE_time']  # 假设键名为 DE_time
            # ...
        """
        # 实际项目中在此处实现 I/O 逻辑
        # return np.load(path)
        pass


@register_dataset("ImageUAVDataset")
class ImageUAVDataset(BaseUAVDataset):
    """
    用于图像（可见光、红外热像图等）的数据集。
    """

    def _load_raw(self, path: str, meta: Dict[str, Any]) -> Any:
        """
        加载图像数据。

        说明:
            通常使用 PIL.Image 或 cv2 读取图片。

        示例代码 (伪代码):
            from PIL import Image
            
            # 使用 PIL 读取并转为 RGB
            image = Image.open(path).convert('RGB')
            return image
        """
        # 实际项目中在此处实现 I/O 逻辑
        pass
