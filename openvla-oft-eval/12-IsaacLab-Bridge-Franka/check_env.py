"""Preflight check: is this shell ready to run the Isaac Lab x OpenVLA experiment?

检查三类东西, 按"会不会让实验静默跑错"排序:

1. 环境变量是否配置齐全 (本脚本的主要目的)
2. 路径与代码资产是否真实存在 (环境变量可能设了, 但指向不存在的目录)
3. 依赖 / 硬件是否达到门槛 (A10 能不能跑 Isaac Sim, 驱动是否够新)

用法 (在工作区根目录执行):

    python 12-IsaacLab-Bridge-Franka/check_env.py
    python 12-IsaacLab-Bridge-Franka/check_env.py --json
    python 12-IsaacLab-Bridge-Franka/check_env.py --policy heuristic --ignore-hardware
    python 12-IsaacLab-Bridge-Franka/check_env.py --checkpoint /mnt/ckpts/openvla-bridge
    python 12-IsaacLab-Bridge-Franka/check_env.py \\
        --simulate-argv "./isaaclab.sh -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py --policy openvla"

退出码:
    0 = 没有 FAIL (WARN/OK/SKIP/NOTE 都可以)
    1 = 存在 FAIL
    2 = 加了 --strict 且存在 WARN

只依赖标准库, 可以在三个地方跑: 本机 / OFT 的 venv_oft / Isaac Lab 的 venv_isaaclab。
在非 Isaac 环境里, Isaac 相关检查标 SKIP, 硬件门槛降级为 WARN。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BRIDGE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BRIDGE_DIR.parent

#: run_domain_gap_eval.sh / .ps1 的默认值, 必须与脚本保持一致
DEFAULT_LAUNCHER = "./isaaclab.sh"
DEFAULT_UNNORM_KEY = "bridge_orig"
DEFAULT_SEED = "7"

#: 上游 prismatic/vla/constants.py 的 BRIDGE_CONSTANTS
BRIDGE_CONSTANTS = {"NUM_ACTIONS_CHUNK": 5, "ACTION_DIM": 7, "PROPRIO_DIM": 7}
#: 猜不中平台时的兜底, 与上游 detect_robot_platform() 的默认分支一致
FALLBACK_PLATFORM = "LIBERO"

#: Isaac Sim 5.1.0 官方验证过的 Linux 驱动版本
MIN_DRIVER = (580, 65, 6)
#: Isaac Sim 官方要求的最低显存 (MiB)
MIN_VRAM_MIB = 16 * 1024
#: pip 安装方式要求的 GLIBC 下限
MIN_GLIBC = (2, 35)

#: 官方明确不支持: 无 RT Core
NO_RT_CORE_MARKERS = (
    "a100", "a800", "h100", "h200", "b100", "b200", "gb200",
    "v100", "p100", "p40", "p4", "k80", "k40", "k20", "m40", "m60",
)
#: 已知带 RT Core 的型号; 未命中任何一侧时报未知
RT_CORE_MARKERS = (
    "a10", "a40", "a16", "a30", "l4", "l40", "t4",
    "rtx", "a6000", "a5000", "a4000", "a2000", "a4500", "a500",
)

#: 本目录必须在 PYTHONPATH 里才能直接 import (脚本自身也会 insert, 所以只算 WARN)
BRIDGE_FILES = (
    "bridge_action_mapper.py",
    "domain_randomization.py",
    "observation.py",
    "openvla_policy.py",
    "bridge_pick_place_env_cfg.py",
    "register_env.py",
    "run_closed_loop.py",
    "analyze_results.py",
    "record_video.py",
    "camera_sweep.py",
    "run_domain_gap_eval.sh",
    "run_domain_gap_eval.ps1",
)

#: 上游 openvla_utils 里 get_vla / get_processor / get_action_head / get_proprio_projector / get_vla_action
OPENVLA_ROOT_MARKER = Path("experiments") / "robot" / "openvla_utils.py"
OPENVLA_CONSTANTS_MARKER = Path("prismatic") / "vla" / "constants.py"

STATUS_MARKS = {
    "OK": "[ OK ]",
    "FAIL": "[FAIL]",
    "WARN": "[WARN]",
    "SKIP": "[SKIP]",
    "NOTE": "[NOTE]",
}
STATUS_ORDER = {"FAIL": 0, "WARN": 1, "OK": 2, "NOTE": 3, "SKIP": 4}


# --------------------------------------------------------------------------- #
# 纯函数 (被 tests/test_env_preflight.py 单独覆盖)
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str = ""
    hint: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "group": self.group,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


def norm_path(value: Any, *, casefold: Optional[bool] = None) -> str:
    """规范化路径用于比较。Windows 下大小写不敏感。"""

    text = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"')))
    text = os.path.normpath(text)
    casefold = (os.name == "nt") if casefold is None else casefold
    return text.casefold() if casefold else text


def split_path_list(value: Any, *, sep: Optional[str] = None) -> List[str]:
    """按 os.pathsep 切分 PYTHONPATH 这类列表变量。

    必须用 os.pathsep 而不是硬编码 ':' —— Windows 的盘符 (C:\\) 会被切碎。
    """

    if value is None:
        return []
    sep = os.pathsep if sep is None else sep
    return [part for part in str(value).split(sep) if part.strip()]


def path_in_list(entries: Sequence[str], target: Any, *, allow_ancestor: bool = True) -> Tuple[bool, str]:
    """target 是否已被 entries 覆盖。返回 (是否覆盖, 覆盖方式)。"""

    wanted = norm_path(target)
    for entry in entries:
        got = norm_path(entry)
        if got == wanted:
            return True, "exact"
        if allow_ancestor:
            try:
                if os.path.commonpath([got, wanted]) == got:
                    return True, "ancestor"
            except ValueError:
                # 不同盘符 / 混合绝对相对路径
                continue
    return False, "missing"


def parse_version(text: Any) -> Tuple[int, ...]:
    """从 '580.65.06' / '1.2' 这类字符串里取数字段。"""

    parts = re.findall(r"\d+", str(text))
    return tuple(int(p) for p in parts[:3]) or (0,)


def version_ge(got: Sequence[int], want: Sequence[int]) -> bool:
    length = max(len(got), len(want))
    a = tuple(got) + (0,) * (length - len(got))
    b = tuple(want) + (0,) * (length - len(want))
    return a >= b


def has_rt_cores(gpu_name: Any) -> Optional[bool]:
    """Isaac Sim 要求 GPU 有 RT Core。返回 True/False, 认不出返回 None。"""

    name = str(gpu_name).lower()
    for marker in NO_RT_CORE_MARKERS:
        if marker in name:
            return False
    for marker in RT_CORE_MARKERS:
        if marker in name:
            return True
    return None


def detect_robot_platform(argv: Sequence[str]) -> str:
    """镜像上游 prismatic/vla/constants.py 的 detect_robot_platform()。"""

    joined = " ".join(str(a) for a in argv).lower()
    if "libero" in joined:
        return "LIBERO"
    if "aloha" in joined:
        return "ALOHA"
    if "bridge" in joined:
        return "BRIDGE"
    return FALLBACK_PLATFORM


def expected_constants(platform_name: str) -> Dict[str, int]:
    if platform_name == "BRIDGE":
        return dict(BRIDGE_CONSTANTS)
    if platform_name == "ALOHA":
        return {"NUM_ACTIONS_CHUNK": 25, "ACTION_DIM": 14, "PROPRIO_DIM": 14}
    return {"NUM_ACTIONS_CHUNK": 8, "ACTION_DIM": 7, "PROPRIO_DIM": 8}


def check_unnorm_key(statistics: Mapping[str, Any], unnorm_key: str) -> Tuple[str, str]:
    """校验 dataset_statistics.json 里能不能找到 unnorm_key 及其 action/proprio 子项。"""

    if not isinstance(statistics, Mapping) or not statistics:
        return "FAIL", "dataset_statistics.json 不是非空对象"
    if unnorm_key not in statistics:
        available = sorted(str(k) for k in statistics.keys())
        return "FAIL", f"缺少 key '{unnorm_key}'；文件里可用的 key: {available}"
    entry = statistics[unnorm_key]
    if not isinstance(entry, Mapping):
        return "FAIL", f"'{unnorm_key}' 不是对象"
    missing = [k for k in ("action", "proprio") if k not in entry]
    if missing:
        return "FAIL", f"'{unnorm_key}' 缺少子项 {missing}"
    return "OK", f"'{unnorm_key}' 同时含 action / proprio"


def driver_status(driver: Any) -> Tuple[str, str]:
    got = parse_version(driver)
    if not got or got == (0,):
        return "SKIP", "驱动版本解析失败"
    if version_ge(got, MIN_DRIVER):
        return "OK", f"{'.'.join(str(x) for x in got)} ≥ {'.'.join(str(x) for x in MIN_DRIVER)}"
    return "WARN", (
        f"{'.'.join(str(x) for x in got)} < {'.'.join(str(x) for x in MIN_DRIVER)}"
        "（Isaac Sim 5.1.0 验证版本）"
    )


# --------------------------------------------------------------------------- #
# 外部探测
# --------------------------------------------------------------------------- #


def _run(cmd: Sequence[str], timeout: int = 20) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def probe_gpu() -> Optional[List[Dict[str, Any]]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        for candidate in (r"C:\Windows\System32\nvidia-smi.exe", "/usr/bin/nvidia-smi"):
            if Path(candidate).is_file():
                exe = candidate
                break
    if exe is None:
        return None
    out = _run(
        [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None
    rows: List[Dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory = int(float(parts[2]))
        except ValueError:
            memory = -1
        rows.append({"name": parts[0], "driver": parts[1], "memory_mib": memory})
    return rows or None


def probe_glibc() -> Optional[Tuple[int, ...]]:
    """只在 Linux 上可信。

    Windows / Git Bash 下 ldd 是 cygwin 版, 输出版本号与 GLIBC 无关,
    照抄会得出"GLIBC 通过"的假结论。
    """

    if not sys.platform.startswith("linux"):
        return None
    out = _run(["ldd", "--version"])
    if not out:
        return None
    match = re.search(r"(\d+)\.(\d+)", out)
    if not match:
        return None
    version = (int(match.group(1)), int(match.group(2)))
    if version[0] < 2:
        # 明显不是 GLIBC 的版本号, 宁可报未知
        return None
    return version


def probe_memory_gib() -> Optional[float]:
    if os.name == "nt":
        try:
            import ctypes  # noqa: PLC0415

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            return None
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return round(pages * size / (1024 ** 3), 1)


def find_openvla_roots(root: Path, max_depth: int = 3) -> List[Path]:
    """深度受限地找 'experiments/robot/openvla_utils.py' 所在目录。"""

    root = Path(root).resolve()
    if not root.is_dir():
        return []
    hits: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        try:
            depth = len(Path(dirpath).resolve().relative_to(root).parts)
        except ValueError:
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in {"__pycache__", "node_modules", "site-packages"}
        ]
        if (Path(dirpath) / OPENVLA_ROOT_MARKER).is_file():
            hits.append(Path(dirpath))
    return hits


_PATCH_CHECK_RE = re.compile(r"^patch-check(\d*)$", re.IGNORECASE)


def pick_preferred_openvla_root(hits: Sequence[Path]) -> Optional[Path]:
    """多个候选时选一个最可能对的。

    工作区里同时存在 patch-check / patch-check2 / patch-check3, 直接取"第一个带 patch 的"
    会选中最早那一个。这里按 patch-checkN 的 N 取最大, 与本项目 README 的约定一致。
    """

    if not hits:
        return None
    numbered: List[Tuple[int, str, Path]] = []
    for hit in hits:
        match = _PATCH_CHECK_RE.match(hit.name)
        if match:
            numbered.append((int(match.group(1) or 0), str(hit), hit))
    if numbered:
        numbered.sort(key=lambda item: (item[0], item[1]))
        return numbered[-1][2]
    patched = [hit for hit in hits if "patch" in hit.name.lower()]
    if patched:
        return sorted(patched, key=lambda p: str(p))[-1]
    return sorted(hits, key=lambda p: len(str(p)))[0]


# --------------------------------------------------------------------------- #
# 检查器
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == "WARN"]

    def counts(self) -> Dict[str, int]:
        counts = {key: 0 for key in STATUS_MARKS}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return counts

    def groups(self) -> List[str]:
        seen: List[str] = []
        for check in self.checks:
            if check.group not in seen:
                seen.append(check.group)
        return seen


class EnvChecker:
    """按"环境变量 → 路径资产 → 依赖 → 硬件"四段跑一遍自检。"""

    ENV_GROUP = "环境变量"
    PATH_GROUP = "路径与代码资产"
    DEP_GROUP = "Python 与依赖"
    HW_GROUP = "目标机硬件门槛"
    ISAAC_GROUP = "Isaac Lab 运行态"

    def __init__(
        self,
        *,
        bridge_dir: Path = BRIDGE_DIR,
        workspace_root: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        policy: str = "openvla",
        checkpoint: str = "",
        openvla_root: str = "",
        launcher: str = "",
        unnorm_key: str = "",
        simulate_argv: str = "",
        ignore_hardware: bool = False,
        require_isaac: bool = False,
        strict_hardware: bool = False,
    ) -> None:
        self.bridge_dir = Path(bridge_dir).resolve()
        self.workspace_root = Path(workspace_root or self.bridge_dir.parent).resolve()
        self.env = dict(os.environ if env is None else env)
        self.policy = policy
        self.cli_checkpoint = checkpoint
        self.cli_openvla_root = openvla_root
        self.cli_launcher = launcher
        self.cli_unnorm_key = unnorm_key
        self.simulate_argv = simulate_argv
        self.ignore_hardware = ignore_hardware
        self.require_isaac = require_isaac
        self.strict_hardware = strict_hardware

        self.report = Report()
        self.isaac_available = self._try_import("isaaclab")
        self.checkpoint = ""
        self.openvla_root = ""
        self.launcher = ""
        self.unnorm_key = ""

    # -- 小工具 ------------------------------------------------------------ #

    def add(self, group: str, name: str, status: str, detail: str = "", hint: str = "") -> Check:
        check = Check(group=group, name=name, status=status, detail=detail, hint=hint)
        self.report.checks.append(check)
        return check

    def _try_import(self, module: str) -> bool:
        try:
            __import__(module)
        except Exception:
            return False
        return True

    def _module_version(self, module: str) -> str:
        try:
            mod = __import__(module)
        except Exception:
            return ""
        return str(getattr(mod, "__version__", "?"))

    def env_get(self, key: str) -> str:
        return str(self.env.get(key, "") or "").strip()

    def _fix_lines(self) -> List[str]:
        """按平台生成可直接粘贴的配置命令。"""

        bash = os.name != "nt"
        lines: List[str] = []
        root = str(self.workspace_root)
        ovla = self.openvla_root or str(Path("02-实验代码") / "patch-check3")
        if bash:
            lines.append(f'export OFT_ROOT="{root}"')
            lines.append(f'export OPENVLA_OFT_ROOT="{ovla}"')
            lines.append(
                f'export PYTHONPATH="$OFT_ROOT/12-IsaacLab-Bridge-Franka:$OPENVLA_OFT_ROOT:$PYTHONPATH"'
            )
            lines.append(f'export ISAACLAB_LAUNCHER="{self.launcher or DEFAULT_LAUNCHER}"')
        else:
            lines.append(f"$env:OFT_ROOT = '{root}'")
            lines.append(f"$env:OPENVLA_OFT_ROOT = '{ovla}'")
            lines.append(
                '$env:PYTHONPATH = "$env:OFT_ROOT\\12-IsaacLab-Bridge-Franka;'
                '$env:OPENVLA_OFT_ROOT;$env:PYTHONPATH"'
            )
            lines.append(f"$env:ISAACLAB_LAUNCHER = '{self.launcher or 'isaaclab.bat'}'")
        return lines

    # -- 1. 环境变量 -------------------------------------------------------- #

    def check_env_vars(self) -> None:
        group = self.ENV_GROUP
        needs_openvla = self.policy == "openvla"

        # OFT_ROOT
        oft_root = self.env_get("OFT_ROOT")
        if not oft_root:
            self.add(
                group, "OFT_ROOT", "WARN",
                f"未设置（本脚本自动推断为 {self.workspace_root}）",
                "run_domain_gap_eval.* 不使用该变量，只有 README 的手动流程用；建议设上以便统一路径",
            )
        elif not Path(oft_root).is_dir():
            self.add(group, "OFT_ROOT", "FAIL", f"指向不存在的目录: {oft_root}", "改成工作区根目录")
        elif norm_path(oft_root) != norm_path(self.workspace_root):
            self.add(
                group, "OFT_ROOT", "WARN",
                f"{oft_root} ≠ 本脚本推断的 {self.workspace_root}",
                "确认你确实在用另一个工作区副本；否则两边混用会出现找不到文件的问题",
            )
        else:
            self.add(group, "OFT_ROOT", "OK", oft_root)

        # OPENVLA_OFT_ROOT —— 被 run_domain_gap_eval.* 用 ${VAR:?} 强制要求
        declared = self.cli_openvla_root or self.env_get("OPENVLA_OFT_ROOT")
        discovered = find_openvla_roots(self.workspace_root)
        preferred = pick_preferred_openvla_root(discovered)
        self.openvla_root = str(Path(declared).expanduser()) if declared else ""

        if declared and (Path(declared) / OPENVLA_ROOT_MARKER).is_file():
            if (Path(declared) / OPENVLA_CONSTANTS_MARKER).is_file():
                self.add(group, "OPENVLA_OFT_ROOT", "OK", self.openvla_root)
            else:
                self.add(
                    group, "OPENVLA_OFT_ROOT", "WARN",
                    f"{self.openvla_root} 有 openvla_utils.py，但缺少 prismatic/vla/constants.py",
                    "可能指到了不完整的 checkout；ACTION_DIM/PROPRIO_DIM 会取不到",
                )
        elif declared:
            hint = f"该目录下找不到 {OPENVLA_ROOT_MARKER}"
            if preferred:
                hint += f"；自动定位到 {preferred}"
            self.add(group, "OPENVLA_OFT_ROOT", "FAIL", f"{self.openvla_root} 不像是 openvla-oft 根目录", hint)
        else:
            status = "FAIL" if needs_openvla else "WARN"
            if preferred:
                detail = f"未设置；自动定位到 {preferred}"
                hint = f"run_domain_gap_eval.* 用 ${{OPENVLA_OFT_ROOT:?}} 强制要求该变量，必须显式 export 为 {preferred}"
            else:
                detail = "未设置，且工作区内未找到 openvla-oft checkout"
                hint = "确认 02-实验代码/patch-check3 是否存在；paths 里需要含 experiments/robot/openvla_utils.py"
            if not needs_openvla:
                hint = "--policy heuristic 不需要该变量，可忽略"
            self.add(group, "OPENVLA_OFT_ROOT", status, detail, hint)
            if preferred and not self.openvla_root:
                self.openvla_root = str(preferred)

        # 工作区里同时存在 patch-check / patch-check2 / patch-check3 时, 必须让人知道选了哪个
        if len(discovered) > 1:
            chosen = Path(self.openvla_root) if self.openvla_root else preferred
            others = [d.name for d in discovered if chosen is None or d != chosen]
            detail = (
                f"发现 {len(discovered)} 个候选，已选用 {chosen.name if chosen else '（未确定）'}"
                f"；另有 {others}"
            )
            self.add(
                group, "openvla-oft 候选目录", "NOTE" if declared else "WARN", detail,
                "用 --openvla-root / OPENVLA_OFT_ROOT 显式指定本次实验真正用的那一份；"
                "不同补丁版本的 ACTION_DIM 与 dataset_statistics 可能不一致",
            )

        # PYTHONPATH
        entries = split_path_list(self.env_get("PYTHONPATH"))
        if not entries:
            self.add(
                group, "PYTHONPATH", "WARN", "未设置",
                "可选：run_closed_loop.py 与 OpenVLAAdapter 都会自行 insert sys.path，所以 PYTHONPATH 只是冗余保险",
            )
        else:
            problems: List[str] = []
            covered, how = path_in_list(entries, self.bridge_dir)
            if not covered:
                problems.append(f"缺少 {self.bridge_dir}")
            if self.openvla_root:
                covered2, _ = path_in_list(entries, self.openvla_root)
                if not covered2:
                    problems.append(f"缺少 {self.openvla_root}")
            if problems:
                self.add(
                    group, "PYTHONPATH", "WARN", "; ".join(problems),
                    "两个目录都可用 --openvla_root 与脚本内 sys.path.insert 兜住，所以只算 WARN",
                )
            else:
                self.add(group, "PYTHONPATH", "OK", f"{len(entries)} 项，已覆盖本项目所需两个目录")

        # ISAACLAB_LAUNCHER
        declared_launcher = self.cli_launcher or self.env_get("ISAACLAB_LAUNCHER")
        resolved = ""
        if declared_launcher:
            if Path(declared_launcher).is_file():
                resolved = declared_launcher
            else:
                found = shutil.which(declared_launcher)
                if found:
                    resolved = found
        else:
            for candidate in ("./isaaclab.sh", "isaaclab.sh", "isaaclab.bat"):
                if shutil.which(candidate):
                    resolved = candidate
                    break
                if Path(candidate).is_file():
                    resolved = candidate
                    break
        self.launcher = resolved
        if resolved:
            self.add(group, "ISAACLAB_LAUNCHER", "OK", resolved)
        else:
            self.add(
                group, "ISAACLAB_LAUNCHER", "WARN",
                f"未设置且找不到 {DEFAULT_LAUNCHER}",
                "run_domain_gap_eval.sh 默认用 ./isaaclab.sh（即 IsaacLab checkout 根目录）；"
                "从别处执行时必须 export 成绝对路径",
            )

        # UNNORM_KEY
        self.unnorm_key = self.cli_unnorm_key or self.env_get("UNNORM_KEY") or DEFAULT_UNNORM_KEY
        source = "命令行/环境变量" if (self.cli_unnorm_key or self.env_get("UNNORM_KEY")) else "默认值"
        self.add(group, "UNNORM_KEY", "OK", f"{self.unnorm_key}（{source}）")

        # SEED / OUT_ROOT
        seed = self.env_get("SEED")
        if not seed:
            self.add(group, "SEED", "OK", f"默认 {DEFAULT_SEED}", "成对评测必须同 seed，别在两组之间改")
        elif not re.fullmatch(r"-?\d+", seed):
            self.add(group, "SEED", "FAIL", f"不是整数: {seed!r}", "run_domain_gap_eval.sh 会把 SEED 直接传给 --seed")
        else:
            self.add(group, "SEED", "OK", seed)

        out_root = self.env_get("OUT_ROOT")
        self.add(
            group, "OUT_ROOT", "OK" if out_root else "NOTE",
            out_root or f"未设置，脚本默认 12-IsaacLab-Bridge-Franka/outputs/domain-gap",
        )

        # checkpoint
        self.checkpoint = self.cli_checkpoint or self.env_get("CHECKPOINT")
        if not self.checkpoint:
            status = "FAIL" if needs_openvla else "WARN"
            hint = "--policy openvla 必须给 --checkpoint（run_domain_gap_eval.* 的第 1 个位置参数）"
            if not needs_openvla:
                hint = "--policy heuristic 不需要 checkpoint"
            self.add(group, "CHECKPOINT", status, "未提供", hint)
        else:
            self.add(group, "CHECKPOINT", "OK", self.checkpoint)

        # Isaac 运行态环境变量
        for key, default, meaning in (
            ("HEADLESS", "未设置", "等价于 --headless"),
            ("ENABLE_CAMERAS", "未设置", "等价于 --enable_cameras；本实验读 RGB，必须为真"),
            ("LIVESTREAM", "-1", "0=关, 1=WebRTC 公网, 2=私网"),
            ("PUBLIC_IP", "未设置", "仅公网串流需要"),
        ):
            value = self.env_get(key)
            self.add(group, f"(可选) {key}", "NOTE", f"{value or default}  ·  {meaning}")

        eula_keys = [k for k in ("OMNI_KIT_ACCEPT_EULA", "ACCEPT_EULA", "PRIVACY_CONSENT") if self.env_get(k)]
        if eula_keys:
            self.add(
                group, "(注意) EULA 相关变量", "WARN",
                f"已设置: {', '.join(eula_keys)}",
                "这会绕过 Omniverse 许可的交互确认；无人值守脚本里不要代用户接受许可条款",
            )
        else:
            self.add(group, "EULA 相关变量", "OK", "未设置（首次启动需交互确认许可，属正常）")

    # -- 2. 路径与代码资产 -------------------------------------------------- #

    def check_paths(self) -> None:
        group = self.PATH_GROUP

        missing = [name for name in BRIDGE_FILES if not (self.bridge_dir / name).is_file()]
        if missing:
            self.add(group, "本目录代码文件", "FAIL", f"缺少 {missing}", "工作区不完整")
        else:
            self.add(group, "本目录代码文件", "OK", f"{len(BRIDGE_FILES)} 个文件齐全")

        tests_dir = self.bridge_dir / "tests"
        test_files = sorted(p.name for p in tests_dir.glob("test_*.py")) if tests_dir.is_dir() else []
        if test_files:
            self.add(group, "无 Isaac 依赖单测", "OK", ", ".join(test_files))
        else:
            self.add(group, "无 Isaac 依赖单测", "WARN", "tests/ 下没有 test_*.py", "改动映射器后没有回归网")

        if not self.checkpoint:
            self.add(group, "checkpoint 目录", "SKIP", "未提供 --checkpoint")
        else:
            ckpt = Path(self.checkpoint).expanduser()
            if not ckpt.exists():
                if "/" in self.checkpoint and not self.checkpoint.startswith((".", "/", "~")):
                    self.add(
                        group, "checkpoint 目录", "SKIP",
                        f"{self.checkpoint} 不像本地路径，按 HF Hub repo id 处理",
                        "dataset_statistics.json 会从 Hub 下载",
                    )
                else:
                    self.add(group, "checkpoint 目录", "FAIL", f"不存在: {ckpt}", "核对挂载路径")
            else:
                self.add(group, "checkpoint 目录", "OK", str(ckpt))
                stats = ckpt / "dataset_statistics.json"
                if not stats.is_file():
                    self.add(
                        group, "dataset_statistics.json", "WARN", f"{stats} 不存在",
                        "基座 checkpoint 可以没有；微调 checkpoint 缺它会导致 predict_action 报 unnorm_key 缺失",
                    )
                else:
                    try:
                        data = json.loads(stats.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        self.add(group, "dataset_statistics.json", "FAIL", f"解析失败: {exc!r}", "文件损坏或不是 JSON")
                    else:
                        status, detail = check_unnorm_key(data, self.unnorm_key)
                        self.add(
                            group, "dataset_statistics.json", status, detail,
                            "run_domain_gap_eval.* 的 UNNORM_KEY 必须与文件里的 key 完全一致",
                        )

    # -- 3. Python 与依赖 --------------------------------------------------- #

    def check_deps(self) -> None:
        group = self.DEP_GROUP

        py_version = ".".join(str(v) for v in sys.version_info[:3])
        self.add(group, "Python 解释器", "OK", f"{py_version} @ {sys.executable}")
        self.add(group, "虚拟环境", "NOTE", sys.prefix)

        if sys.version_info[:2] == (3, 11):
            self.add(group, "Python 版本线", "OK", "3.11 —— Isaac Sim 5.x 要求的版本")
        elif sys.version_info[:2] == (3, 10):
            self.add(
                group, "Python 版本线", "NOTE",
                "3.10 —— OFT 侧版本线；Isaac Sim 5.1 需要 3.11，两者不能共用同一个环境",
            )
        else:
            self.add(
                group, "Python 版本线", "WARN", f"{py_version} 既不是 3.11（Isaac）也不是 3.10（OFT）",
                "本机跑静态检查可以，但不要用它跑 Isaac Lab 或 OFT 推理",
            )

        for module in ("isaaclab", "isaacsim"):
            if self._try_import(module):
                self.add(group, module, "OK", self._module_version(module))
            else:
                status = "FAIL" if self.require_isaac else "SKIP"
                self.add(
                    group, module, status, f"无法 import（当前不在 Isaac Lab 环境）",
                    "在 venv_isaaclab 里用 ./isaaclab.sh -p <script> 运行即可；加 --require-isaac 可把它当失败",
                )

        if self._try_import("torch"):
            version = self._module_version("torch")
            try:
                import torch  # noqa: PLC0415

                cuda_ok = bool(torch.cuda.is_available())
                name = torch.cuda.get_device_name(0) if cuda_ok else "不可用"
            except Exception as exc:  # pragma: no cover - 取决于本机 torch
                cuda_ok, name = False, f"查询失败: {exc!r}"
            self.add(group, "torch", "OK" if cuda_ok else "WARN", f"{version} · CUDA={cuda_ok} · {name}")
            if version.startswith("2.7"):
                self.add(group, "torch 版本线", "OK", "2.7.x —— 与 Isaac Sim 5.1 匹配")
            elif version.startswith("2.2"):
                self.add(group, "torch 版本线", "NOTE", "2.2.x —— OFT 侧版本线，不能用于 Isaac Lab")
            else:
                self.add(
                    group, "torch 版本线", "WARN", version,
                    "Isaac Lab 侧期望 2.7.0+cu128；OFT 侧是 2.2.0+cu121。升级时务必带 --no-deps",
                )
        else:
            self.add(group, "torch", "SKIP", "无法 import", "在本机做静态检查时属正常")

        if self._try_import("numpy"):
            self.add(group, "numpy", "OK", self._module_version("numpy"))
        else:
            self.add(group, "numpy", "WARN", "无法 import", "tests/ 与动作映射需要 numpy")

        for module, why in (("imageio", "--video 录制需要"), ("PIL", "camera_sweep / record_video 需要")):
            if self._try_import(module):
                self.add(group, module, "OK", f"{self._module_version(module)} · {why}")
            else:
                self.add(group, module, "WARN", f"无法 import · {why}", "不录制视频时可以忽略")

        if self._try_import("pytest"):
            self.add(group, "pytest", "OK", self._module_version("pytest"))
        else:
            self.add(group, "pytest", "WARN", "无法 import", "阶段 0 的静态检查需要 pytest")

        # 平台常量探测：猜错会静默换掉 ACTION_DIM / PROPRIO_DIM
        argv: Sequence[str]
        if self.simulate_argv:
            try:
                argv = shlex.split(self.simulate_argv)
            except ValueError as exc:
                self.add(group, "平台常量探测", "FAIL", f"--simulate-argv 解析失败: {exc!r}")
                argv = []
        else:
            argv = sys.argv
        if argv:
            platform_name = detect_robot_platform(argv)
            consts = expected_constants(platform_name)
            detail = (
                f"argv 判定为 {platform_name} · ACTION_DIM={consts['ACTION_DIM']} "
                f"· PROPRIO_DIM={consts['PROPRIO_DIM']} · chunk={consts['NUM_ACTIONS_CHUNK']}"
            )
            if platform_name == "BRIDGE":
                self.add(group, "平台常量探测", "OK", detail)
            else:
                self.add(
                    group, "平台常量探测", "FAIL", detail,
                    "上游靠 'bridge' 字样匹配 argv，匹配不到会默认 LIBERO（PROPRIO_DIM=8，chunk=8），"
                    "而本项目观测只有 7 维 proprio、闭环默认 open_loop_steps=1。"
                    "命令里必须出现 bridge 字样（--unnorm_key bridge_orig 或 --domain_mode bridge_aligned 即可）；"
                    "OpenVLAAdapter 会自动补 --bridge，但用 --simulate-argv 可以先验证真实命令",
                )

    # -- 4. 目标机硬件门槛 -------------------------------------------------- #

    def check_hardware(self) -> None:
        group = self.HW_GROUP
        if self.ignore_hardware:
            self.add(group, "硬件门槛检查", "SKIP", "--ignore-hardware")
            return

        # 非 Isaac 环境里, 硬件门槛降级为 WARN, 免得在本机刷出一堆无关 FAIL
        downgrade = (not self.isaac_available) and (not self.strict_hardware)

        def emit(name: str, status: str, detail: str, hint: str = "") -> None:
            if downgrade and status == "FAIL":
                status = "WARN"
                hint = (hint + " ").strip() + "（本机不是 Isaac 运行环境，仅作参考；目标机请加 --strict-hardware）"
            self.add(group, name, status, detail, hint)

        gpus = probe_gpu()
        if not gpus:
            emit("GPU", "WARN", "nvidia-smi 不可用或没有输出", "云实例上应能直接运行 nvidia-smi")
        else:
            for index, gpu in enumerate(gpus):
                tag = f"GPU{index}"
                rt = has_rt_cores(gpu["name"])
                if rt is False:
                    emit(
                        f"{tag} RT Core", "FAIL", f"{gpu['name']} 无 RT Core",
                        "Isaac Sim 官方明确不支持无 RT Core 的卡（A100 / H100 等），换卡才能解决",
                    )
                elif rt is None:
                    emit(f"{tag} RT Core", "WARN", f"{gpu['name']} 未能判定", "对照 Isaac Sim 系统要求自行确认")
                else:
                    emit(f"{tag} RT Core", "OK", f"{gpu['name']} 带 RT Core")

                memory = int(gpu.get("memory_mib") or -1)
                if memory < 0:
                    emit(f"{tag} 显存", "SKIP", "读取失败")
                elif memory < MIN_VRAM_MIB:
                    emit(
                        f"{tag} 显存", "FAIL", f"{memory} MiB < {MIN_VRAM_MIB} MiB",
                        "低于 Isaac Sim 官方下限；A10 的 24 GB 也只能让 Isaac 单独跑，见文档 §6",
                    )
                else:
                    emit(f"{tag} 显存", "OK", f"{memory} MiB ≥ {MIN_VRAM_MIB} MiB")
                    if memory <= 24 * 1024 + 512:
                        emit(
                            f"{tag} 显存预算", "NOTE",
                            f"{memory} MiB：与 OpenVLA 共存前必须实测 Isaac 峰值"
                            "（BF16 实测 15839 MiB / INT8 9315 MiB）",
                        )

                status, detail = driver_status(gpu["driver"])
                emit(f"{tag} 驱动", status, detail, "装 Isaac Sim 5.1 前先升级到 580.65.06 或更高")

        glibc = probe_glibc()
        if glibc is None:
            emit("GLIBC", "SKIP", f"非 Linux（{platform.system()}），该项不适用",
                 "pip 方式安装 Isaac Sim 要求在 Linux 上核对 GLIBC ≥ 2.35")
        elif version_ge(glibc, MIN_GLIBC):
            emit("GLIBC", "OK", f"{glibc[0]}.{glibc[1]} ≥ {MIN_GLIBC[0]}.{MIN_GLIBC[1]}")
        else:
            emit(
                "GLIBC", "FAIL", f"{glibc[0]}.{glibc[1]} < {MIN_GLIBC[0]}.{MIN_GLIBC[1]}",
                "不要升级系统 GLIBC；改用 Isaac Sim 二进制包或容器方案",
            )

        if sys.platform.startswith("linux"):
            emit("架构", "OK", f"{platform.machine()} · {platform.system()}")
        else:
            emit(
                "架构", "WARN", f"{platform.system()} {platform.machine()}",
                "生产运行建议放 Ubuntu 22.04/24.04；Windows 仅适合练 GUI",
            )

        memory_gib = probe_memory_gib()
        if memory_gib is None:
            emit("主机内存", "SKIP", "无法读取")
        elif memory_gib >= 32:
            emit("主机内存", "OK", f"{memory_gib} GiB ≥ 32 GiB")
        else:
            emit("主机内存", "WARN", f"{memory_gib} GiB < 32 GiB", "Isaac Sim 官方要求 32 GB 以上")

        try:
            free_gib = round(shutil.disk_usage(self.bridge_dir).free / (1024 ** 3), 1)
        except OSError:
            emit("磁盘可用", "SKIP", "无法读取")
        else:
            if free_gib >= 50:
                emit("磁盘可用", "OK", f"{free_gib} GiB ≥ 50 GiB")
            else:
                emit("磁盘可用", "WARN", f"{free_gib} GiB < 50 GiB", "还要给 Isaac 资产缓存留空间")

    # -- 5. Isaac 运行态 ---------------------------------------------------- #

    def check_isaac_runtime(self) -> None:
        group = self.ISAAC_GROUP
        if not self.isaac_available:
            self.add(
                group, "Isaac 运行态", "SKIP",
                "当前不在 Isaac Lab 环境，以下检查需要 ./isaaclab.sh -p 运行",
            )
            return

        if self._try_import("gymnasium"):
            try:
                import gymnasium as gym  # noqa: PLC0415

                sys.path.insert(0, str(self.bridge_dir))
                import register_env  # noqa: F401,PLC0415

                spec = gym.spec("Isaac-Bridge-PickPlace-Franka-IK-Rel-v0")
                self.add(group, "任务注册", "OK" if spec else "FAIL", "Isaac-Bridge-PickPlace-Franka-IK-Rel-v0")
            except Exception as exc:
                self.add(group, "任务注册", "FAIL", f"{type(exc).__name__}: {exc}", "先确认 register_env 能 import")
        else:
            self.add(group, "任务注册", "SKIP", "gymnasium 不可用")

        # 相机与渲染是否被打开：--enable_cameras 决定 experience 文件
        enable = self.env_get("ENABLE_CAMERAS").lower() in {"1", "true", "yes"}
        cli_hint = "--enable_cameras" in " ".join(sys.argv)
        if enable or cli_hint:
            self.add(group, "相机渲染开关", "OK", "ENABLE_CAMERAS/--enable_cameras 已打开")
        else:
            self.add(
                group, "相机渲染开关", "FAIL", "未检测到 --enable_cameras / ENABLE_CAMERAS",
                "环境里有 CameraCfg，不加这个参数渲染器不启动，相机输出是空张量且不报错",
            )

    # -- 驱动 -------------------------------------------------------------- #

    def run(self) -> Report:
        self.report.meta = {
            "host": f"{platform.system()} {platform.release()} / {platform.machine()}",
            "python": f"{'.'.join(str(v) for v in sys.version_info[:3])} @ {sys.executable}",
            "workspace_root": str(self.workspace_root),
            "bridge_dir": str(self.bridge_dir),
            "policy": self.policy,
            "in_isaac_env": "yes" if self.isaac_available else "no",
        }
        self.check_env_vars()
        self.check_paths()
        self.check_deps()
        self.check_hardware()
        self.check_isaac_runtime()
        self.report.hints = self._fix_lines()
        return self.report


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def render_text(report: Report) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Isaac Lab × OpenVLA 环境自检")
    lines.append("=" * 78)
    for key, value in report.meta.items():
        lines.append(f"  {key:<16} {value}")

    for group in report.groups():
        lines.append("")
        lines.append(f"── {group} " + "─" * max(0, 70 - len(group)))
        for check in report.checks:
            if check.group != group:
                continue
            lines.append(f"  {STATUS_MARKS.get(check.status, '[????]')} {check.name}")
            if check.detail:
                lines.append(f"         {check.detail}")
            if check.hint and check.status in {"FAIL", "WARN"}:
                lines.append(f"         ↳ {check.hint}")

    counts = report.counts()
    total = sum(counts.values())
    lines.append("")
    lines.append("-" * 78)
    lines.append(
        "  统计: " + " · ".join(f"{key} {counts[key]}" for key in ("OK", "FAIL", "WARN", "SKIP", "NOTE"))
        + f"  (共 {total} 项)"
    )
    if report.failures:
        lines.append(f"  结论: 不通过 —— {len(report.failures)} 项 FAIL")
    elif report.warnings:
        lines.append(f"  结论: 通过 —— 但有 {len(report.warnings)} 项 WARN 需要确认")
    else:
        lines.append("  结论: 通过 —— 未发现需要处理的问题")

    if report.failures:
        lines.append("")
        lines.append("  必须先修复:")
        for index, check in enumerate(report.failures, 1):
            lines.append(f"    {index}. [{check.group}] {check.name}: {check.detail}")
    if report.warnings:
        lines.append("")
        lines.append("  建议确认:")
        for index, check in enumerate(report.warnings, 1):
            lines.append(f"    {index}. [{check.group}] {check.name}: {check.detail}")

    lines.append("")
    lines.append("  可直接粘贴的配置命令:")
    for hint in report.hints:
        lines.append(f"    {hint}")
    lines.append("-" * 78)
    return "\n".join(lines)


def to_json(report: Report) -> str:
    return json.dumps(
        {
            "meta": report.meta,
            "counts": report.counts(),
            "passed": not report.failures,
            "checks": [check.as_dict() for check in report.checks],
            "fix_hints": report.hints,
        },
        indent=2,
        ensure_ascii=False,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 Isaac Lab × OpenVLA 实验所需的环境变量、路径与依赖是否齐全",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", choices=["openvla", "heuristic"], default="openvla",
                        help="决定 OPENVLA_OFT_ROOT / CHECKPOINT 是否必需（默认 openvla）")
    parser.add_argument("--checkpoint", default="", help="Bridge checkpoint 路径，或 HF repo id")
    parser.add_argument("--openvla-root", default="", help="覆盖 OPENVLA_OFT_ROOT")
    parser.add_argument("--launcher", default="", help="覆盖 ISAACLAB_LAUNCHER")
    parser.add_argument("--unnorm-key", default="", help="覆盖 UNNORM_KEY（默认 bridge_orig）")
    parser.add_argument("--simulate-argv", default="",
                        help="用真实命令行验证平台常量探测，例如 \"./isaaclab.sh -p .../run_closed_loop.py --policy openvla\"")
    parser.add_argument("--ignore-hardware", action="store_true", help="跳过硬件门槛检查（本机自检时用）")
    parser.add_argument("--strict-hardware", action="store_true",
                        help="非 Isaac 环境里也把硬件 FAIL 当 FAIL")
    parser.add_argument("--require-isaac", action="store_true",
                        help="把 isaaclab/isaacsim 无法 import 当作 FAIL")
    parser.add_argument("--strict", action="store_true", help="存在 WARN 时也返回退出码 2")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checker = EnvChecker(
        policy=args.policy,
        checkpoint=args.checkpoint,
        openvla_root=args.openvla_root,
        launcher=args.launcher,
        unnorm_key=args.unnorm_key,
        simulate_argv=args.simulate_argv,
        ignore_hardware=args.ignore_hardware,
        require_isaac=args.require_isaac,
        strict_hardware=args.strict_hardware,
    )
    report = checker.run()
    print(to_json(report) if args.json else render_text(report))
    if report.failures:
        return 1
    if args.strict and report.warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
