#!/usr/bin/env python
"""
tools/setup_env.py
==================
扫描硬件配置，并使用 Conda 自动创建/配置“无人机故障诊断”项目运行环境。

设计目标（研究复现友好）：
- 自动生成硬件报告（便于论文/实验记录可追溯）
- 自动判断 GPU 是否“可用且与当前 PyTorch 预编译包兼容”，否则回退 CPU，避免出现 no kernel image
- 全程使用 conda 安装（minepy 等在 Windows 上更稳）

用法示例：
  python tools/setup_env.py --env-name DTL_Learn
  python tools/setup_env.py --env-name DTL_Learn --device cpu
  python tools/setup_env.py --env-name DTL_Learn --device cuda --cuda 12.4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


def run(cmd: List[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        shell=False,
    )


def run_shell(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True, shell=True)


def which_conda() -> Optional[str]:
    # 优先用 CONDA_EXE（conda activate 后通常存在）
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # 尝试 PATH
    exe = shutil.which("conda")
    if exe:
        return exe

    # Windows 下 conda 常是 conda.bat
    exe = shutil.which("conda.bat")
    if exe:
        return exe
    return None


def conda_cmd(conda_exe: str, args: List[str]) -> List[str]:
    """
    Windows 下 conda.bat 需要通过 cmd /c 执行；其它平台直接执行 conda。
    """
    p = Path(conda_exe).name.lower()
    if p.endswith(".bat") or p.endswith(".cmd"):
        return ["cmd", "/c", conda_exe, *args]
    return [conda_exe, *args]


def scan_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "cpu": {},
        "memory": {},
        "gpu": {"nvidia": []},
    }

    # CPU / RAM (Windows: wmic; Linux: /proc)
    if platform.system().lower() == "windows":
        try:
            cpu_name = run_shell("wmic cpu get Name /value").stdout
            m = re.search(r"Name=(.+)", cpu_name)
            if m:
                info["cpu"]["name"] = m.group(1).strip()
        except Exception:
            pass
        try:
            mem = run_shell("wmic computersystem get TotalPhysicalMemory /value").stdout
            m = re.search(r"TotalPhysicalMemory=(\d+)", mem)
            if m:
                info["memory"]["total_bytes"] = int(m.group(1))
        except Exception:
            pass
    else:
        try:
            if Path("/proc/cpuinfo").exists():
                txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"model name\s*:\s*(.+)", txt)
                if m:
                    info["cpu"]["name"] = m.group(1).strip()
        except Exception:
            pass
        try:
            if Path("/proc/meminfo").exists():
                txt = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"MemTotal:\s*(\d+)\s*kB", txt)
                if m:
                    info["memory"]["total_bytes"] = int(m.group(1)) * 1024
        except Exception:
            pass

    # NVIDIA GPU
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            # compute_cap 不是所有驱动都支持查询；失败就只记录 name
            cp = run([nvsmi, "--query-gpu=name,compute_cap", "--format=csv,noheader"], check=False)
            lines = (cp.stdout or "").strip().splitlines()
            if cp.returncode == 0 and lines:
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    item = {"name": parts[0] if parts else line.strip()}
                    if len(parts) > 1 and parts[1]:
                        item["compute_capability"] = parts[1]
                    info["gpu"]["nvidia"].append(item)
            else:
                cp2 = run([nvsmi, "--query-gpu=name", "--format=csv,noheader"], check=False)
                for line in (cp2.stdout or "").strip().splitlines():
                    if line.strip():
                        info["gpu"]["nvidia"].append({"name": line.strip()})
        except Exception:
            pass
    return info


def decide_device(
    hw: Dict[str, Any],
    requested: Literal["auto", "cpu", "cuda"],
) -> Literal["cpu", "cuda"]:
    """
    自动决策策略（务实优先）：
    - requested=cpu：强制 CPU
    - requested=cuda：强制 CUDA（若未检测到 NVIDIA 则报错）
    - requested=auto：有 NVIDIA 则尝试 CUDA，否则 CPU

    关键：对“超新架构”（如 sm_120）很多旧 torch wheel 不含 kernel，会直接崩溃。
    因此 auto 模式下，如果检测到 compute capability >= 12.0，则默认回退 CPU（更稳）。
    """
    if requested == "cpu":
        return "cpu"

    gpus = hw.get("gpu", {}).get("nvidia", []) or []
    if not gpus:
        if requested == "cuda":
            raise RuntimeError("请求使用 CUDA，但未检测到 NVIDIA GPU（或 nvidia-smi 不可用）")
        return "cpu"

    # 若能读到 compute_capability，做一个保守的兼容性判断
    cap = None
    for g in gpus:
        cc = g.get("compute_capability")
        if isinstance(cc, str) and re.match(r"^\d+(\.\d+)?$", cc):
            cap = float(cc)
            break

    if requested == "cuda":
        return "cuda"

    # auto：对非常新的 compute capability（>=12）先默认 CPU，避免“装了 CUDA 但跑不了”的坑
    if cap is not None and cap >= 12.0:
        return "cpu"
    return "cuda"


def conda_env_exists(conda_exe: str, env_name: str) -> bool:
    cp = run(conda_cmd(conda_exe, ["env", "list"]), check=True)
    return any(line.split() and line.split()[0] == env_name for line in cp.stdout.splitlines())


def conda_install(conda_exe: str, env_name: str, pkgs: List[str], channels: List[str]) -> None:
    args = ["install", "-n", env_name, "-y"]
    for c in channels:
        args.extend(["-c", c])
    args.extend(pkgs)
    run(conda_cmd(conda_exe, args), check=True, capture=False)


def conda_create(conda_exe: str, env_name: str, python_version: str) -> None:
    run(conda_cmd(conda_exe, ["create", "-n", env_name, f"python={python_version}", "-y"]), check=True, capture=False)


def conda_run_python(conda_exe: str, env_name: str, code: str) -> None:
    run(conda_cmd(conda_exe, ["run", "-n", env_name, "python", "-c", code]), check=True, capture=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="DTL_Learn", help="Conda 环境名")
    ap.add_argument("--python", default="3.10", help="Python 版本（建议 3.10）")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="设备策略：auto/cpu/cuda")
    ap.add_argument("--cuda", default="12.4", help="pytorch-cuda 版本（device=cuda 时使用）")
    ap.add_argument("--report", default="env/hardware_report.json", help="硬件报告输出路径")
    args = ap.parse_args()

    conda_exe = which_conda()
    if not conda_exe:
        print("未找到 conda（请先安装 Miniconda/Anaconda 并确保 conda 在 PATH 中）", file=sys.stderr)
        return 2

    hw = scan_hardware()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(hw, ensure_ascii=False, indent=2), encoding="utf-8")

    device = decide_device(hw, args.device)  # cpu/cuda
    print(f"[hardware] report={report_path}")
    print(f"[env] conda={conda_exe}")
    print(f"[env] name={args.env_name} python={args.python} device={device}")

    if not conda_env_exists(conda_exe, args.env_name):
        conda_create(conda_exe, args.env_name, args.python)

    # 基础科研包（数据、可视化、日志）
    conda_install(
        conda_exe,
        args.env_name,
        [
            "numpy=1.26.*",
            "pandas",
            "pyyaml",
            "matplotlib",
            "tensorboard",
            "scikit-learn",
            "minepy",
        ],
        channels=["conda-forge"],
    )

    # PyTorch：CPU 或 CUDA
    if device == "cpu":
        conda_install(conda_exe, args.env_name, ["pytorch", "cpuonly"], channels=["pytorch"])
    else:
        conda_install(
            conda_exe,
            args.env_name,
            ["pytorch", f"pytorch-cuda={args.cuda}"],
            channels=["pytorch", "nvidia"],
        )

    # 快速自检：import + MIC + torch device
    conda_run_python(
        conda_exe,
        args.env_name,
        "import torch, numpy, pandas, yaml; "
        "from minepy import MINE; import numpy as np; "
        "m=MINE(); m.compute_score(np.arange(10), np.arange(10)); "
        "print('verify_ok', 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mic', m.mic());",
    )

    print("\nNext:")
    if platform.system().lower() == "windows":
        print(f"  conda activate {args.env_name}")
    else:
        print(f"  conda activate {args.env_name}")
    print("  python train_evaluate.py --config config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

