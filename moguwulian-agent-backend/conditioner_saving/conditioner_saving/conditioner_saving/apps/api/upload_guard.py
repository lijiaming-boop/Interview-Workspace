"""上传文件安全守卫（P0 安全改造）

统一收敛所有上传/下载文件路径的安全处理，修复改进方案 P0-1：
- 文件名净化：剥离路径成分，拒绝危险名（..、控制字符、Windows 保留名）
- 扩展名白名单：.xls 缺少安全读取依赖（xlrd 未装），统一拒绝，只放行 .xlsx
- 文件签名校验：xlsx 必须是 ZIP 容器（PK\\x03\\x04），防止改扩展名的伪装文件
- xlsx 结构校验：验证 ZIP 目录、[Content_Types].xml、工作簿入口、成员数量、
  单成员与总解压大小上限（审查报告 P1：仅查 4 字节魔数挡不住伪 ZIP 与
  ZIP 炸弹）
- 流式写盘 + 字节上限：不再把整个文件读进内存后再判断大小
- 最终路径边界检查：resolve 后必须仍位于允许目录内，杜绝符号链接/.. 穿越
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, Optional

# 允许上传的扩展名（.xls 暂拒：运行环境无 xlrd，且旧格式存在解析器安全风险）
ALLOWED_SPREADSHEET_EXTS = {".xlsx"}
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
# 允许下载的结果文件扩展名
ALLOWED_DOWNLOAD_EXTS = {".xlsx", ".docx", ".csv", ".json", ".md", ".png", ".jpg", ".jpeg"}

# ZIP 本地文件头签名（xlsx/docx 都是 OPC 容器）
_ZIP_MAGIC = b"PK\x03\x04"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

# xlsx（OPC 容器）结构校验上限——防伪 ZIP 与 ZIP 炸弹
_XLSX_REQUIRED_MEMBERS = ("[Content_Types].xml",)   # OPC 必备；工作簿入口另行检查
_XLSX_WORKBOOK_ENTRIES = ("xl/workbook.xml",)
_XLSX_MAX_MEMBERS = 1000                            # 正常 xlsx 成员数远小于此
_XLSX_MAX_MEMBER_BYTES = 64 * 1024 * 1024           # 单成员解压上限 64MB
_XLSX_MAX_TOTAL_BYTES = 256 * 1024 * 1024           # 总解压上限 256MB（压缩比>256x 拒绝）

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_CHUNK_SIZE = 1024 * 1024  # 1MB


class UploadRejected(Exception):
    """上传被安全守卫拒绝（由路由层转成 4xx）"""


def sanitize_filename(name: str) -> str:
    """剥离路径成分并拒绝危险文件名，返回安全文件名

    - Path(name).name 剥离目录部分（含 .. 穿越、绝对路径、反斜杠）
    - 拒绝控制字符、纯点名、Windows 保留名、尾部点/空格（Windows 陷阱）
    """
    if not name or not name.strip():
        raise UploadRejected("文件名不能为空")
    base = Path(name).name
    if base in ("", ".", ".."):
        raise UploadRejected("非法文件名")
    if _CONTROL_RE.search(base):
        raise UploadRejected("文件名包含非法控制字符")
    stem = Path(base).stem.upper().split(".")[0]
    if stem in _WINDOWS_RESERVED:
        raise UploadRejected(f"文件名使用了 Windows 保留名: {stem}")
    base = base.rstrip(". ")
    if not base:
        raise UploadRejected("非法文件名")
    return base


def check_extension(filename: str, allowed: Iterable[str]) -> str:
    """校验扩展名白名单，返回小写扩展名"""
    suffix = Path(filename).suffix.lower()
    if suffix not in set(allowed):
        raise UploadRejected(f"仅支持 {' / '.join(sorted(allowed))}，收到 {suffix!r}")
    return suffix


def verify_xlsx_structure(path: Path) -> None:
    """校验 xlsx 的 ZIP/OPC 容器结构与解压规模

    P1 修复（审查报告）：原实现只检查 `PK\\x03\\x04` 4 字节——伪 ZIP
    （4 字节头 + 随意内容）与 ZIP 炸弹（小压缩比爆炸解压）都能通过。
    现在校验：
    - 可完整解析 ZIP 中央目录（zipfile.ZipFile）
    - 必含 OPC 必备成员 `[Content_Types].xml` 与工作簿入口 `xl/workbook.xml`
    - 成员数量、单成员声明解压大小、总解压大小均有上限
    - 逐成员限流解压读取：CRC 完整性校验（zipfile 在尾部校验失败抛
      BadZipFile），同时防止中央目录谎报 size 的"小头大身"文件
    """
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if not infos:
                raise UploadRejected("xlsx 容器为空（无有效成员），已拒绝")

            names = {i.filename for i in infos}
            for required in _XLSX_REQUIRED_MEMBERS:
                if required not in names:
                    raise UploadRejected(
                        f"缺少 OPC 必备成员 {required}，不是有效的 xlsx，已拒绝")
            if not any(n in names for n in _XLSX_WORKBOOK_ENTRIES):
                raise UploadRejected(
                    "缺少工作簿入口 xl/workbook.xml，不是有效的 xlsx，已拒绝")

            if len(infos) > _XLSX_MAX_MEMBERS:
                raise UploadRejected(
                    f"xlsx 成员数量 {len(infos)} 超过上限 {_XLSX_MAX_MEMBERS}，已拒绝")

            total_declared = 0
            for info in infos:
                if info.file_size > _XLSX_MAX_MEMBER_BYTES:
                    raise UploadRejected(
                        f"xlsx 成员 {info.filename} 解压大小超过 "
                        f"{_XLSX_MAX_MEMBER_BYTES // (1024 * 1024)}MB 上限，已拒绝")
                total_declared += info.file_size
                if total_declared > _XLSX_MAX_TOTAL_BYTES:
                    raise UploadRejected(
                        f"xlsx 总解压大小超过 {_XLSX_MAX_TOTAL_BYTES // (1024 * 1024)}MB "
                        "上限（疑似 ZIP 炸弹），已拒绝")

            # 限流解压读取：CRC 校验 + 防中央目录谎报
            total_read = 0
            for info in infos:
                member_read = 0
                try:
                    with zf.open(info) as f:
                        while True:
                            chunk = f.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            member_read += len(chunk)
                            total_read += len(chunk)
                            if member_read > _XLSX_MAX_MEMBER_BYTES:
                                raise UploadRejected(
                                    f"xlsx 成员 {info.filename} 实际解压超出上限"
                                    "（目录信息不可信），已拒绝")
                            if total_read > _XLSX_MAX_TOTAL_BYTES:
                                raise UploadRejected(
                                    "xlsx 实际总解压大小超出上限（疑似 ZIP 炸弹），已拒绝")
                except UploadRejected:
                    raise
                except (zipfile.BadZipFile, RuntimeError, NotImplementedError,
                        OSError, ValueError, EOFError) as e:
                    raise UploadRejected(
                        f"xlsx 成员 {info.filename} 读取/校验失败（{e}），已拒绝") from e
    except zipfile.BadZipFile as e:
        raise UploadRejected(f"文件不是有效的 ZIP/xlsx 容器（{e}），已拒绝") from e
    except UploadRejected:
        raise
    except (OSError, RuntimeError, NotImplementedError, ValueError) as e:
        raise UploadRejected(f"xlsx 容器校验失败（{e}），已拒绝") from e


def verify_signature(path: Path, suffix: str) -> None:
    """按扩展名校验文件魔数，防止伪装扩展名的恶意文件"""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as e:
        raise UploadRejected(f"无法读取上传文件: {e}") from e
    if suffix == ".xlsx":
        if not head.startswith(_ZIP_MAGIC):
            raise UploadRejected("文件内容不是有效的 xlsx（ZIP 容器），已拒绝")
        verify_xlsx_structure(path)
    if suffix in (".png",) and not head.startswith(_PNG_MAGIC):
        raise UploadRejected("文件内容不是有效的 PNG 图片，已拒绝")
    if suffix in (".jpg", ".jpeg") and not head.startswith(_JPEG_MAGIC):
        raise UploadRejected("文件内容不是有效的 JPEG 图片，已拒绝")


def ensure_inside(directory: Path, path: Path) -> Path:
    """路径边界检查：path 解析后必须仍在 directory 内，返回解析后的安全路径"""
    base = directory.resolve()
    resolved = path.resolve()
    if base != resolved and base not in resolved.parents:
        raise UploadRejected("文件路径越出允许目录，已拒绝")
    return resolved


def save_upload(
    fileobj,
    dest_dir: Path,
    filename: str,
    *,
    allowed_exts: Iterable[str] = ALLOWED_SPREADSHEET_EXTS,
    max_bytes: int = 100 * 1024 * 1024,
    forced_name: Optional[str] = None,
    verify_magic: bool = True,
) -> Path:
    """安全保存上传文件，返回最终落盘路径

    流程：文件名净化 → 扩展名白名单 → 流式写盘（超限即断） → 魔数校验 →
    路径边界检查。任一步失败都会清理半成品文件并抛 UploadRejected。

    Args:
        fileobj: 具有 read(n) 的文件对象（FastAPI UploadFile.file / SpooledTemporaryFile）
        dest_dir: 目标目录（通常为会话目录）
        filename: 客户端提交的原始文件名（先经 _recover_multipart_filename 解码）
        forced_name: 强制使用的服务端文件名（忽略客户端文件名，如 upload_survey.xlsx）
        verify_magic: 是否校验文件魔数
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if forced_name is not None:
        final_name = sanitize_filename(forced_name)
    else:
        final_name = sanitize_filename(filename)
    suffix = check_extension(final_name, allowed_exts)

    target = dest_dir / final_name
    target = ensure_inside(dest_dir, target)

    written = 0
    try:
        with open(target, "wb") as out:
            while True:
                chunk = fileobj.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise UploadRejected(f"文件超过 {max_bytes // (1024 * 1024)}MB 上限")
                out.write(chunk)
    except UploadRejected:
        _cleanup(target)
        raise
    except OSError as e:
        _cleanup(target)
        raise UploadRejected(f"写入上传文件失败: {e}") from e

    if written == 0:
        _cleanup(target)
        raise UploadRejected("上传文件为空")

    if verify_magic:
        try:
            verify_signature(target, suffix)
        except UploadRejected:
            _cleanup(target)
            raise

    os.chmod(target, 0o644)
    return target


def safe_download_path(directory: Path, filename: str,
                       allowed_exts: Optional[Iterable[str]] = None) -> Path:
    """安全解析下载路径：文件名净化 + 扩展名白名单 + 目录边界检查

    下载端点必须使用本函数，不允许 output_dir / filename 直拼。
    """
    final_name = sanitize_filename(filename)
    suffix = Path(final_name).suffix.lower()
    if allowed_exts is not None and suffix not in set(allowed_exts):
        raise UploadRejected(f"不允许下载的文件类型: {suffix!r}")
    target = ensure_inside(directory, directory / final_name)
    return target


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
