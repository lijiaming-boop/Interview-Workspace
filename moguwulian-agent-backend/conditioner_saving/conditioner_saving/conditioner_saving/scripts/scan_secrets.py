"""密钥/敏感信息扫描（P0 安全改造）

P1 修复（审查报告）：扫描范围改为 git 受版本控制的文件（`git ls-files`），
不再无条件跳过 `.env`——如果有人把 `.env` 强制加入版本控制，CI 必须能
查出来；未跟踪的本地 `.env`（正常情况）自然不在扫描范围。
git 不可用时退化为目录遍历（排除依赖/产物目录）。

历史中已知的 .env 泄露必须靠"轮换密钥 + 清理历史"解决
（见 docs/密钥轮换与历史清理.md），扫描历史会让 CI 永远失败；
本脚本的目标是防止"新的"密钥进入版本控制。

用法:
    python scripts/scan_secrets.py [paths...]   # 默认扫整个仓库
退出码: 发现疑似密钥 → 1（CI 门禁失败），否则 0
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Windows 默认 GBK 控制台无法输出 Emoji（报告验证项）：
# 统一切换 UTF-8，避免扫描器自身在 CI/终端崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 退化目录遍历时的排除：二进制/依赖/产物目录
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "tmp", "data", ".workbuddy", ".trae",
    ".qoder", "output", "outputs", ".next",
}
# 自身与文档允许引用模式示例（否则脚本自己也会被扫出来）
ALLOWLIST_FILES = {
    "scripts/scan_secrets.py",
}

# 常见密钥模式（正则）
PATTERNS = {
    "anthropic_key": re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    "openai_key": re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "private_key_block": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_password_assignment": re.compile(
        r"(?i)\b(PASSWORD|PASSWD|SECRET)\s*[:=]\s*['\"](?!\s*['\"])[^'\"\s]{8,}['\"]"),
}

TEXT_SUFFIXES = {
    ".py", ".go", ".ts", ".tsx", ".js", ".json", ".yml", ".yaml", ".toml",
    ".md", ".env.example", ".ini", ".cfg", ".sh", ".bat", ".ps1", ".txt",
}


def _git_tracked_files() -> list[Path] | None:
    """返回 git 受版本控制的文件列表；git 不可用/非仓库返回 None"""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            return None
        out = []
        for name in proc.stdout.decode("utf-8", errors="replace").split("\0"):
            if not name:
                continue
            p = PROJECT_ROOT / name
            if p.is_file():
                out.append(p)
        return out
    except (OSError, subprocess.SubprocessError):
        return None


def iter_files() -> list[Path]:
    """扫描范围：git 受版本控制的文件（含被误提交的 .env）；无 git 时目录遍历"""
    tracked = _git_tracked_files()
    if tracked is not None:
        return tracked
    out = []
    for p in PROJECT_ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(PROJECT_ROOT)
        parts = set(rel.parts)
        if parts & EXCLUDED_DIRS:
            continue
        if p.suffix.lower() in TEXT_SUFFIXES or p.name in (".env", ".env.example", "Dockerfile"):
            out.append(p)
    return out


def scan() -> int:
    findings = 0
    for path in iter_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in ALLOWLIST_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    print(f"❌ 疑似密钥 [{name}] {rel}:{lineno}: {line.strip()[:80]}")
                    findings += 1
    if findings:
        print(f"\n共发现 {findings} 处疑似密钥——请立即轮换并从源码中移除。")
        return 1
    print("✅ 密钥扫描通过：未发现疑似密钥")
    return 0


if __name__ == "__main__":
    sys.exit(scan())
