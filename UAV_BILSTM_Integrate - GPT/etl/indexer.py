from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import DataConfig, ExperimentConfig


_CASE_RE = re.compile(r"^Case_([A-Za-z0-9])([A-Za-z0-9])([A-Za-z0-9]{2})([A-Za-z0-9]{6})\.csv$")


def parse_rflymad_case(path: str) -> Dict[str, object]:
    """
    RflyMAD 文件名解析器：

    文件名格式：Case_[A][B][CD][EFGHIJ].csv
    A: 1=SIL,2=HIL,3=Real
    B: 0 hover /1 waypoint /2 velocity /3 circling /4 acce /5 dece
    CD: 两位故障类型，10=No fault

    输出字典：
    {
      "domain": "SIL|HIL|Real",
      "flight_mode": int,
      "fault_code": "00"~"10",
      "case_id": int,
      "path": str
    }
    """
    p = Path(path)
    name = p.name
    m = _CASE_RE.match(name)
    if not m:
        raise ValueError(f"文件名不符合格式 Case_[A][B][CD][EFGHIJ].csv：{name}")

    a, b, cd, efg = m.groups()

    domain_map = {"1": "SIL", "2": "HIL", "3": "Real"}
    if a not in domain_map:
        raise ValueError(f"domain 码 A 不合法：{a}（期望 1/2/3） in {name}")

    try:
        flight_mode = int(b)
    except Exception as e:
        raise ValueError(f"flight_mode 码 B 不可转 int：{b} in {name}") from e

    if not re.fullmatch(r"\d{2}", cd):
        raise ValueError(f"fault_code 码 CD 不合法（期望两位数字）：{cd} in {name}")

    if not efg.isdigit():
        raise ValueError(f"case_id 段 EFGHIJ 不是纯数字：{efg} in {name}")
    case_id = int(efg)

    return {
        "domain": domain_map[a],
        "flight_mode": int(flight_mode),
        "fault_code": cd,
        "case_id": int(case_id),
        "path": str(p.resolve()),
    }


def build_index(data_cfg: DataConfig | ExperimentConfig) -> pd.DataFrame:
    """
    扫描 dataset_root 下所有 CSV，生成索引 DataFrame。

    关键点（中文注释）：
    - 只接收命名符合 Case_*.csv 的文件（可避免把无关 CSV 混进来）
    - 输出按路径排序，确保可复现
    """
    # 兼容误用：很多人会把 ExperimentConfig 直接传进来
    # 这里做一个“温和的自动纠正”，等价于 build_index(cfg.data)
    if isinstance(data_cfg, ExperimentConfig):
        data_cfg = data_cfg.data
    elif not isinstance(data_cfg, DataConfig) and hasattr(data_cfg, "data"):
        # 进一步兼容鸭子类型（例如你自己包了一层 config 对象）
        maybe = getattr(data_cfg, "data", None)
        if isinstance(maybe, DataConfig):
            data_cfg = maybe

    if not isinstance(data_cfg, DataConfig):
        raise TypeError("build_index 期望 DataConfig（或 ExperimentConfig），例如：build_index(cfg.data)")

    root = Path(data_cfg.dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            "找不到 dataset_root："
            f"{root.resolve()}\n"
            f"- 当前读取到的 data.dataset_root = {data_cfg.dataset_root!r}\n"
            "- 请确认你修改的是正在加载的那个 YAML（例如 user_run.py 里用的 configs/rflymad_etl.yaml）\n"
            "- Windows 绝对路径推荐用正斜杠：D:/path/to/dataset，或用单引号：'D:\\path\\to\\dataset'（避免 \\U 等转义）"
        )

    rows = []
    for p in root.glob(data_cfg.file_glob):
        if not p.is_file():
            continue
        # 只接收符合命名规则的文件；不符合则跳过（避免把无关 CSV 混进来）
        if not _CASE_RE.match(p.name):
            continue
        info = parse_rflymad_case(str(p))
        rows.append(
            {
                "filename": p.name,
                **info,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"在 {root.resolve()} 下未找到符合 `Case_[A][B][CD][EFGHIJ].csv` 的文件；当前 data 配置：{asdict(data_cfg)}"
        )

    df = pd.DataFrame(rows).sort_values(["domain", "flight_mode", "fault_code", "case_id", "path"]).reset_index(drop=True)
    df["file_id"] = df.index.astype(int)
    return df
