"""Skill 调用封装(使用 Claude Agent SDK)

提供统一的 Skill 调用接口,使用异步方式调用 Claude Agent SDK

P0 安全改造（改进方案第三节）：
1. 确定性技能（纯 Python 脚本）不再经过通用 Agent——直接 subprocess 执行
   受控脚本，不消耗 token、不依赖 API Key、天然没有 Bash/Write 攻击面。
2. LLM 型技能按"最小工具集"原则收敛 Agent 权限：
   - extract-air-compressor-data-llm：彻底不走 Agent——文本预提取由本地
     脚本完成（进程内），结构化提取由一次纯文本 LLM 调用完成（allowed_tools
     为空、单轮），模型只返回 JSON 文本；输出经 Schema 校验后由服务端写盘，
     模型全程没有 Read/Write 文件系统权限（审查报告 P0-2）。
   - plant-simulation：生成并执行仿真代码必须保留 Bash/Write（已知边界），
     但每次调用在独立沙箱目录中执行：cwd 指向临时沙箱（不含宿主项目文件，
     输入/输出经 io/ 中转），不加载 user/project 设置（setting_sources=[]，
     隔离模式），Linux/macOS 上启用 SDK bash 沙箱并默认断网（仿真脚本只需
     本地 Python；密钥无法从沙箱外传——网络被沙箱阻断）。
3. 校准/验证分离（改进方案第六节）：old_station 仿真按输入 JSON 的
   simulation_intent 区分——verify（默认）禁止调参追目标；calibrate 才允许对齐。

已知边界（如实记录）：SDK 的 env 参数只能增改、不能删除环境变量，
ANTHROPIC_API_KEY 必须保留给 CLI 自身认证；沙箱代码可读但无法外传
（网络被沙箱阻断，文件写入限制在沙箱目录内）。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import asyncio
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock


class SkillError(Exception):
    """Skill 调用错误"""
    pass


# 确定性技能注册表：skill_name -> (skill 目录, 脚本文件名)
# 这些 Skill 的脚本是纯 Python 规则实现，直接 subprocess 执行即可，
# 不需要也不应该起一个拥有 Bash/Write 权限的 Agent 会话（P0 安全改造）。
_DETERMINISTIC_SKILLS = {
    "extract-and-check-data": ("extract-and-check-data", "extract_and_check_data.py"),
    "extract-survey-data": ("extract-survey-data", "extract_survey_data.py"),
    "check-data-quality": ("check-data-quality", "check_data_quality.py"),
    "suggest-data-filling": ("suggest-data-filling", "suggest_data_filling.py"),
}

# LLM 型技能的工具白名单（最小工具集原则）
# - 空压提取：不再走 Agent（纯文本 LLM 调用 + 服务端 Schema 校验与写盘），
#   因此不在本表中
# - 冷站仿真：需生成并执行仿真代码，保留 Bash；在独立沙箱目录执行（见下）
_SKILL_TOOL_POLICY = {
    "plant-simulation": ["Skill", "Read", "Write", "Bash"],
}

# 需要沙箱化的 LLM 技能：cwd 指向临时沙箱目录，不加载宿主项目设置，
# Linux/macOS 上启用 bash 沙箱并默认断网（Windows 上 SDK 沙箱不可用，
# 退化为 cwd 隔离——生产部署目标为 Linux 容器）。
_SANDBOXED_SKILLS = {"plant-simulation"}

# 沙箱内 io 中转目录名（输入拷入 / 输出拷出）
_SANDBOX_IO = "io"

# 空压 LLM 提取输出的 JSON Schema（模型必须按此结构返回，服务端校验后写盘）
_AIR_COMPRESSOR_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["rawData", "rejectionCheck"],
    "properties": {
        "rawData": {
            "type": "object",
            "required": ["projectLevel", "equipmentLevel"],
            "properties": {
                "projectLevel": {"type": "object"},
                "equipmentLevel": {"type": "array"},
            },
        },
        "rejectionCheck": {
            "type": "object",
            "required": ["shouldReject"],
            "properties": {
                "shouldReject": {"type": "boolean"},
                "criticalIssues": {"type": "array"},
                "highSeverityIssues": {"type": "array"},
            },
        },
    },
}


def _coerce_issue_list(value: Any) -> List[Any]:
    """把模型返回的 issues 字段规范化为列表（非列表/无效项丢弃）"""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [item for item in value
            if (isinstance(item, str) and item.strip()) or isinstance(item, dict)]


def _as_number(value: Any) -> Optional[float]:
    """数值字段宽容解析：缺失/不可解析返回 None；bool 不算数值"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _as_text(value: Any) -> str:
    """文本字段规范化：None/空白视为缺失，返回去空白字符串"""
    return str(value).strip() if value is not None else ""


def _enforce_air_compressor_rejection_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
    """服务端独立复评 SKILL.md 打回条件（审查报告2 P1-1）

    模型自报的 shouldReject 不可信——纯文本路径曾出现"空项目、空设备且
    shouldReject=false"绕过打回继续测算。这里按 SKILL.md 打回条件清单
    在服务端重新逐条判定：与模型报告合并去重后，只要存在致命问题即
    强制 shouldReject=true，打回由数据事实决定而非模型自觉。
    """
    raw = payload["rawData"]
    project = raw["projectLevel"]
    equipment = raw["equipmentLevel"]
    check = payload["rejectionCheck"]

    critical = _coerce_issue_list(check.get("criticalIssues"))
    high = _coerce_issue_list(check.get("highSeverityIssues"))

    def add_issue(bucket: List[Any], msg: str) -> None:
        if msg not in bucket:
            bucket.append(msg)

    # 致命 1-3：报警压力 / 平均压力 / 年度总用量缺失或非正
    for field, msg in (
        ("alarmPressureLowerLimit",
         "现有系统报警压力缺失：请查看空压机控制系统中设置的\"低压报警\"值，"
         "或在站房压力表附近标注的报警阈值。"),
        ("averagePressure",
         "近14天系统平均压力缺失：请从监控系统或流量计/压力变送器导出至少"
         "连续7天的站房出口总管压力数据（CSV文件，包含时间戳和压力值）。"),
        ("totalAnnualEnergyConsumption",
         "空压系统年度总用量缺失：请提供空压站全年总耗电量（kWh），"
         "可根据月度电费单累加。"),
    ):
        v = _as_number(project.get(field))
        if v is None or v <= 0:
            add_issue(critical, msg)

    # 致命 4：无运行设备（设备清单为空同样视为无运行设备）
    running_like = [eq for eq in equipment
                    if _as_text(eq.get("runningStatus")) in ("运行", "备用")]
    if not equipment or not running_like:
        add_issue(critical,
                  "无运行设备：请在空压机清单中逐台标注当前状态：运行、备用、"
                  "故障、停用。至少有一台设备处于\"运行\"或\"备用\"状态。")

    # 致命 5-6：站房编码为空；设备编码为空或同站房内重复
    seen_ids: Dict[str, set] = {}
    for i, eq in enumerate(equipment):
        station = _as_text(eq.get("stationId"))
        if not station:
            add_issue(critical,
                      "站房编码为空：请确保每台设备都有站房编码，"
                      "允许多台设备属于同一站房。")
        eq_id = _as_text(eq.get("equipmentId"))
        if not eq_id:
            add_issue(critical, "设备编码为空：请确保每台设备有唯一编码。")
        elif eq_id in seen_ids.setdefault(station, set()):
            add_issue(critical,
                      f"设备编码重复：同一站房内设备编码 {eq_id!r} 重复，"
                      "请确保每台设备有唯一编码。")
        else:
            seen_ids[station].add(eq_id)

        # 高严重性 1-3：运行设备功率/流量缺失；设备类型缺失
        if _as_text(eq.get("runningStatus")) == "运行":
            if (_as_number(eq.get("ratedPower")) is None
                    or _as_number(eq.get("ratedPower")) <= 0):
                add_issue(high, "运行设备铭牌功率缺失：请补充运行空压机的铭牌功率（kW）。")
            if (_as_number(eq.get("ratedFlow")) is None
                    or _as_number(eq.get("ratedFlow")) <= 0):
                add_issue(high, "运行设备铭牌流量缺失：请补充运行空压机的铭牌流量（m³/min）。")
        if not _as_text(eq.get("equipmentType")):
            add_issue(high, "设备类型缺失：请填写设备类型（螺杆/离心）。")

    check["criticalIssues"] = critical
    check["highSeverityIssues"] = high
    check["shouldReject"] = bool(critical)
    return payload


def _validate_air_compressor_payload(payload: Any) -> Dict[str, Any]:
    """对空压 LLM 返回做轻量 Schema 校验，返回规范化的 dict

    校验失败抛 SkillError——模型输出不可信时绝不写盘。
    结构校验通过后，再由 _enforce_air_compressor_rejection_rules 按数据
    事实独立复评打回条件（不信任模型自报的 shouldReject）。
    """
    if not isinstance(payload, dict):
        raise SkillError("空压提取 LLM 输出不是 JSON 对象")

    def _fail(msg: str) -> None:
        raise SkillError(f"空压提取 LLM 输出未通过 Schema 校验: {msg}")

    raw = payload.get("rawData")
    if not isinstance(raw, dict):
        _fail("缺少 rawData 对象")
    if not isinstance(raw.get("projectLevel"), dict):
        _fail("rawData.projectLevel 必须是对象")
    if not isinstance(raw.get("equipmentLevel"), list):
        _fail("rawData.equipmentLevel 必须是数组")
    check = payload.get("rejectionCheck")
    if not isinstance(check, dict):
        _fail("缺少 rejectionCheck 对象")
    if not isinstance(check.get("shouldReject"), bool):
        _fail("rejectionCheck.shouldReject 必须是布尔值")
    # 设备行结构抽查：equipmentLevel 必须是设备对象数组
    for i, eq in enumerate(raw["equipmentLevel"]):
        if not isinstance(eq, dict):
            _fail(f"rawData.equipmentLevel[{i}] 不是对象")
        if not isinstance(eq.get("equipmentId"), (str, type(None))):
            _fail(f"rawData.equipmentLevel[{i}].equipmentId 类型非法")
    return _enforce_air_compressor_rejection_rules(payload)


class SkillCaller:
    """Skill 调用器(基于 Claude Agent SDK)"""

    def __init__(
        self,
        timeout: int = 300,
        api_key: Optional[str] = None,
        project_root: Optional[Path] = None
    ):
        """初始化 Skill 调用器

        Args:
            timeout: 调用超时时间(秒)
            api_key: Anthropic API Key,如果不提供则从环境变量读取
            project_root: 项目根目录,包含 .claude/skills/ 目录

        Raises:
            SkillError: API key 未配置（仅调用 LLM 型技能时才必需；
                        P0 改造后确定性技能可在无 Key 环境运行）
        """
        self.timeout = timeout
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

        # 注意：不再在初始化时强制要求 API Key——确定性技能（纯脚本）
        # 完全不需要 LLM。缺 Key 只在真正调用 LLM 型技能时报错。

        # 确定项目根目录
        if project_root:
            self.project_root = Path(project_root).resolve()
        else:
            # 默认:从当前文件向上两级(core/ -> 项目根)
            self.project_root = Path(__file__).parent.parent.resolve()

        # 验证 skills 目录存在
        self.skills_dir = self.project_root / ".claude" / "skills"
        if not self.skills_dir.exists():
            raise SkillError(
                f"Skills 目录不存在: {self.skills_dir}\n"
                f"项目根目录: {self.project_root}"
            )

    def call_skill(
        self,
        skill_name: str,
        *args,
        check_output: bool = True,
        callback: Optional[Callable[[str, str], None]] = None,
        max_retries: int = 0
    ) -> Dict[str, Any]:
        """调用 Claude Skill(同步包装器)

        P0 安全改造：确定性技能（纯 Python 规则脚本）直接 subprocess 执行，
        不再起 Agent 会话；仅 LLM 型技能走 Claude Agent SDK。

        Args:
            skill_name: Skill 名称 (如 "extract-and-check-data")
            *args: Skill 参数(文件路径等)
            check_output: 是否检查输出文件
            callback: 回调函数 callback(timestamp, message)，每次 AI 回复时调用
            max_retries: 最大重试次数，用于代码生成类任务(如仿真)，失败后自动修复重试

        Returns:
            包含执行结果的字典

        Raises:
            SkillError: Skill 调用失败
        """
        # 确定性技能：受控脚本直调（无 LLM、无 Agent、无 Shell 授权问题）
        if skill_name in _DETERMINISTIC_SKILLS:
            return self._run_deterministic_script(
                skill_name, *args, check_output=check_output, callback=callback)

        # LLM 型技能：必须配置 API Key
        if not self.api_key:
            raise SkillError(
                f"Skill '{skill_name}' 需要 LLM，但 ANTHROPIC_API_KEY 环境变量未设置。"
            )

        # 在新的事件循环中运行异步调用
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._call_skill_async(skill_name, *args, check_output=check_output, callback=callback, max_retries=max_retries)
        )

    def _run_deterministic_script(
        self,
        skill_name: str,
        *args,
        check_output: bool = True,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """直接执行确定性 Skill 的受控脚本（subprocess，不经 Agent）

        与旧 Agent 路径相比：零 token、无 Bash/Write 权限面、
        无 API Key 依赖；脚本失败/缺输出直接显式报错。
        """
        skill_dir, script_name = _DETERMINISTIC_SKILLS[skill_name]
        script_path = self.project_root / ".claude" / "skills" / skill_dir / "scripts" / script_name
        if not script_path.exists():
            raise SkillError(f"Skill 脚本不存在: {script_path}")

        def cb(message: str) -> None:
            if callback:
                callback(datetime.now().strftime("%H:%M:%S"), message)

        cb(f"⚙️ 确定性脚本直调（不经 Agent）：{script_name} {' '.join(str(a) for a in args)}")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        proc = subprocess.Popen(
            [sys.executable, str(script_path), *[str(a) for a in args]],
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        tail: List[str] = []

        # P1 修复（审查报告）：原先在主线程阻塞遍历 stdout（读到 EOF 为止），
        # 之后才 wait(timeout)——脚本挂死时超时永远不生效，线程/任务锁被永久
        # 占用。改为后台线程排空 stdout，主线程带超时等待进程退出。
        def _drain_stdout() -> None:
            try:
                for line in proc.stdout or []:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    tail.append(line)
                    if len(tail) > 50:
                        tail.pop(0)
                    cb(line)
            except (ValueError, OSError):
                # 进程被 kill 后管道关闭，读取报错属预期
                pass

        reader = threading.Thread(target=_drain_stdout, daemon=True,
                                  name=f"skill-out-{script_name}")
        reader.start()
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise SkillError(f"Skill '{skill_name}' 脚本执行超时(>{self.timeout}s)") from e
        reader.join(timeout=5)

        if proc.returncode != 0:
            raise SkillError(
                f"Skill '{skill_name}' 脚本退出码 {proc.returncode}:\n" + "\n".join(tail[-10:])
            )

        # 输出文件检查（与旧逻辑一致的语义：第一个文件参数之后是输出）
        if check_output and len(args) >= 2:
            output_file = Path(args[2]) if skill_name == "suggest-data-filling" else Path(args[1])
            if not output_file.exists():
                raise SkillError(
                    f"Skill '{skill_name}' 执行完成，但未生成预期的输出文件: {output_file}"
                )

        cb(f"✅ {skill_name} 完成")
        return {
            "success": True,
            "ai_responses": [],
            "all_messages": list(tail),
            "result_text": "\n".join(tail),
            "attempts": 1,
        }

    async def _call_skill_async(
        self,
        skill_name: str,
        *args,
        check_output: bool = True,
        callback: Optional[Callable[[str, str], None]] = None,
        max_retries: int = 0
    ) -> Dict[str, Any]:
        """调用 Claude Skill(异步实现)

        Args:
            skill_name: Skill 名称
            *args: Skill 参数
            check_output: 是否检查输出文件
            callback: 回调函数，每次收到 AI 回复时调用
            max_retries: 最大重试次数(用于仿真代码生成失败后自动修复)

        Returns:
            包含执行结果的字典

        Raises:
            SkillError: Skill 调用失败
        """
        # LLM 型技能必须配置 API Key（确定性技能已在 call_skill 直调，不会到这里）
        if not self.api_key:
            raise SkillError(
                f"Skill '{skill_name}' 需要 LLM，但 ANTHROPIC_API_KEY 环境变量未设置。"
            )

        # P0 安全改造（审查报告 P0-2）：空压提取不走 Agent——模型只返回
        # JSON 文本，Schema 校验与写盘由服务端完成，模型没有文件系统权限。
        if skill_name == "extract-air-compressor-data-llm":
            return await self._extract_air_compressor_llm(
                *args, callback=callback)

        # 沙箱技能：在临时沙箱目录运行 Agent（cwd 隔离 + 不加载宿主设置 +
        # Linux/macOS bash 沙箱默认断网），输入/输出经 io 中转拷入拷出。
        if skill_name in _SANDBOXED_SKILLS:
            return await self._call_sandboxed_skill(
                skill_name, *args, check_output=check_output,
                callback=callback, max_retries=max_retries)

        options = self._build_agent_options(skill_name, cwd=self.project_root)
        return await self._run_agent(
            skill_name, args, options,
            check_output=check_output, callback=callback,
            max_retries=max_retries,
        )

    async def _extract_air_compressor_llm(
        self,
        *args,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """空压提取（纯文本 LLM 路径，不走 Agent）

        P0 安全改造（审查报告 P0-2）：原实现给 Agent 配 Read+Write 文件
        权限，上传内容可经提示词注入让模型读密钥/覆盖项目文件。现改为：
          1. 本地脚本预提取文本（subprocess，与原逻辑一致）
          2. 一次纯文本 LLM 调用（allowed_tools=[]，单轮），模型只返回 JSON
          3. 服务端按 Schema 校验，校验通过后由服务端写输出文件
        模型全程没有 Read/Write/Bash 权限。
        """
        if len(args) < 2:
            raise SkillError(
                "extract-air-compressor-data-llm 需要至少2个参数: 输入文件路径, 输出JSON路径, [原始文件名]"
            )
        input_path, output_path = Path(args[0]), Path(args[1])
        original = args[2] if len(args) >= 3 else input_path.name
        text_tmp = output_path.with_suffix(".text.json")

        def cb(message: str) -> None:
            if callback:
                callback(datetime.now().strftime("%H:%M:%S"), message)

        # 1. 文本预提取（本地受控脚本，进程内 subprocess）
        cb(f"⚙️ 文本预提取（本地脚本，不经 Agent）: {input_path.name}")
        extract_script = (
            self.project_root / ".claude" / "skills"
            / "extract-air-compressor-data-llm"
            / "scripts" / "extract_air_compressor_data_llm.py"
        )
        if extract_script.exists():
            proc = await asyncio.to_thread(
                lambda: subprocess.run(
                    [sys.executable, str(extract_script),
                     str(input_path), str(text_tmp), str(original)],
                    cwd=str(self.project_root),
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=300,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                )
            )
            if proc.returncode != 0:
                raise SkillError(
                    f"文本预提取脚本失败(退出码 {proc.returncode}): "
                    f"{(proc.stdout or '')[-400:]}{(proc.stderr or '')[-400:]}"
                )
        try:
            text_content = json.loads(text_tmp.read_text(encoding="utf-8")).get("text_content", "")
        except (OSError, json.JSONDecodeError) as e:
            raise SkillError(f"预提取文本不可读: {e}") from e

        # 2. 纯文本 LLM 调用（无工具、单轮；输出必须是 JSON 文本）
        cb("🧠 结构化提取（纯文本 LLM 调用，模型无文件系统权限）...")
        # P1 修复（审查报告2）：提示词要求模型"按 SKILL.md 规范"执行打回
        # 检查，却没有提供 SKILL.md 内容——模型并不知道打回条件清单。
        # 这里把 SKILL.md 全文注入提示词；缺失视为部署错误直接失败
        # （与 _run_agent 对 SKILL.md 缺失的处理一致）。
        skill_md_path = (self.skills_dir / "extract-air-compressor-data-llm"
                         / "SKILL.md")
        if not skill_md_path.exists():
            raise SkillError(f"Skill not found: extract-air-compressor-data-llm")
        skill_md = skill_md_path.read_text(encoding="utf-8")
        prompt = (
            "你是空压机调研表数据提取引擎。请严格按下方 SKILL.md 规范，"
            "从调研表文本中提取结构化数据，并逐条执行打回条件检查。\n\n"
            f"## SKILL.md 规范\n{skill_md}\n\n"
            "## 输出要求\n"
            "- 只输出一个 JSON 对象，不要输出任何解释、Markdown 代码块围栏或其他文本\n"
            "- 顶层结构: {\"rawData\": {\"projectLevel\": {...}, \"equipmentLevel\": [...]}, "
            "\"rejectionCheck\": {\"shouldReject\": 布尔, \"criticalIssues\": [], \"highSeverityIssues\": []}}\n"
            "- 数值字段只保留纯数字（去除单位），缺失设为 null\n"
            "- equipmentLevel 为每台设备一个对象（stationId/equipmentId/equipmentType/brand/model/"
            "manufactureYear/runningStatus/ratedPower/ratedFlow 等）\n"
            "- industryType 取值必须为：电池、整车及电子、钢铁有色、其他\n"
            "- 打回条件判定必须以 SKILL.md 的打回条件清单为准；任一致命错误满足时 "
            "shouldReject 必须为 true 并在 criticalIssues 中记录问题\n\n"
            f"## 原始文件名\n{original}\n\n"
            "## 调研表文本\n"
            f"{text_content}"
        )
        raw_text = await self._create_message_async(prompt)

        # 3. 解析 + Schema 校验 + 服务端写盘
        payload = self._parse_json_response(raw_text)
        payload = _validate_air_compressor_payload(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        cb(f"✅ 提取完成，输出已由服务端写入: {output_path.name}")
        return {
            "success": True,
            "ai_responses": [raw_text],
            "all_messages": [raw_text],
            "result_text": raw_text,
            "attempts": 1,
        }

    @staticmethod
    def _parse_json_response(raw_text: str) -> Any:
        """从 LLM 文本回复中解析 JSON（容忍代码围栏与前后杂文本）"""
        text = raw_text.strip()
        if text.startswith("```"):
            # 剥离 ```json ... ``` 围栏
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 兜底：截取首个 { 到末个 } 之间
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError as e:
                raise SkillError(f"空压提取 LLM 返回不是有效 JSON: {e}") from e
        raise SkillError("空压提取 LLM 返回中未找到 JSON")

    def _build_agent_options(self, skill_name: str, *, cwd: Path,
                             extra_env: Optional[Dict[str, str]] = None):
        """构造 Agent SDK 选项（权限最小化 + 隔离设置）"""
        # 隔离模式：setting_sources=[] 不加载 user/project/local 设置
        #（避免外部设置扩大权限面）；skills= 显式启用唯一需要的技能，
        # SDK 自动配置 Skill 工具与技能加载。
        options = ClaudeAgentOptions(
            cwd=str(cwd),
            setting_sources=[],
            skills=[skill_name],
            allowed_tools=_SKILL_TOOL_POLICY.get(skill_name, ["Skill", "Read"]),
            permission_mode="bypassPermissions",
        )
        if extra_env:
            options.env = dict(extra_env)
        return options

    # ==================== 沙箱执行（plant-simulation） ====================

    def _stage_sandbox(self, skill_name: str, args: tuple,
                       sandbox: Path) -> Tuple[tuple, list]:
        """按技能参数语义把输入拷入沙箱、把输出映射到沙箱内

        plant-simulation 参数约定（见 _build_prompt）：
          args[0] 输入文件（generate_operating 时为内联 JSON 数据）
          args[1] 输出路径（new_station/old_station 为目录，其余为 .json 文件）
          args[2] 任务类型标记
          args[3] 可选第二输入文件（extract_topology 的文本转储）

        返回 (映射后的参数, 回拷计划)；回拷计划元素为 (沙箱路径, 原路径)。
        """
        io_dir = sandbox / _SANDBOX_IO
        io_dir.mkdir(parents=True, exist_ok=True)
        mapped: List[Any] = list(args)
        backplan: list = []  # (sandbox_path, original_path)

        if skill_name == "plant-simulation":
            task_type = str(args[2]) if len(args) >= 3 else ""

            # 输入文件：存在即拷入沙箱（generate_operating 兼容内联数据——
            # 仅当 args[0] 不是已存在的文件时保持原样）
            if Path(args[0]).exists():
                src = Path(args[0])
                dest = io_dir / src.name
                shutil.copy2(src, dest)
                mapped[0] = str(dest)
            elif task_type != "generate_operating":
                raise SkillError(f"沙箱输入不存在: {args[0]}")

            # 输出路径（目录型 / 文件型）
            if len(args) >= 2:
                out = Path(args[1])
                dest_out = io_dir / out.name
                if task_type in ("new_station", "old_station"):
                    dest_out.mkdir(parents=True, exist_ok=True)
                mapped[1] = str(dest_out)
                backplan.append((dest_out, out))

            # 第二输入文件
            if len(args) >= 4 and task_type == "extract_topology":
                src2 = Path(args[3])
                if src2.exists():
                    dest2 = io_dir / src2.name
                    shutil.copy2(src2, dest2)
                    mapped[3] = str(dest2)

        return tuple(mapped), backplan

    def _collect_sandbox_outputs(self, backplan: list) -> None:
        """把沙箱内产出拷回原位置（文件/目录均支持）"""
        for sandbox_path, original in backplan:
            try:
                if sandbox_path.is_dir():
                    original.mkdir(parents=True, exist_ok=True)
                    for item in sandbox_path.rglob("*"):
                        if item.is_file():
                            rel = item.relative_to(sandbox_path)
                            target = original / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, target)
                elif sandbox_path.is_file():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sandbox_path, original)
            except OSError as e:
                raise SkillError(f"沙箱输出回拷失败: {e}") from e

    async def _call_sandboxed_skill(
        self,
        skill_name: str,
        *args,
        check_output: bool = True,
        callback: Optional[Callable[[str, str], None]] = None,
        max_retries: int = 0,
    ) -> Dict[str, Any]:
        """在独立沙箱中运行 LLM 技能（当前用于 plant-simulation）

        沙箱边界：
        - cwd = 临时沙箱目录：宿主项目文件不在 Agent 视野内（无宿主目录）
        - setting_sources=[]：不加载 user/project/local 设置（隔离模式）
        - Linux/macOS：启用 SDK bash 沙箱，网络默认全拒（仿真脚本只需
          本地 Python）；Windows 上 SDK 沙箱不可用，退化为 cwd 隔离
        - 输入参数拷入沙箱，输出运行后拷回原路径
        """
        sandbox = Path(tempfile.mkdtemp(prefix="sim_sandbox_"))
        try:
            # 技能资产拷入沙箱（cwd 变更后技能发现依赖 <cwd>/.claude/skills）
            skill_src = self.skills_dir / skill_name
            if skill_src.exists():
                shutil.copytree(skill_src, sandbox / ".claude" / "skills" / skill_name)

            mapped_args, backplan = self._stage_sandbox(skill_name, args, sandbox)
            cb = callback
            if cb:
                cb(datetime.now().strftime("%H:%M:%S"),
                   f"🛡️ 沙箱执行：{skill_name}（宿主目录隔离，默认断网）")

            options = self._build_agent_options(
                skill_name, cwd=sandbox)
            if sys.platform != "win32":  # bash 沙箱仅 macOS/Linux 可用
                options.sandbox = {
                    "enabled": True,
                    "autoAllowBashIfSandboxed": True,
                    "allowUnsandboxedCommands": False,
                    "network": {"allowedDomains": []},
                }

            result = await self._run_agent(
                skill_name, mapped_args, options,
                check_output=check_output, callback=cb,
                max_retries=max_retries,
            )
            self._collect_sandbox_outputs(backplan)
            return result
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    async def _run_agent(
        self,
        skill_name: str,
        args: tuple,
        options,
        *,
        check_output: bool = True,
        callback: Optional[Callable[[str, str], None]] = None,
        max_retries: int = 0,
    ) -> Dict[str, Any]:
        """Agent 重试循环（原 _call_skill_async 主体，与沙箱/路径配置解耦）"""
        # 检查 Skill 是否存在（沙箱调用已在 staging 阶段拷贝，跳过亦可）
        skill_md = self.skills_dir / skill_name / "SKILL.md"
        if not skill_md.exists():
            skill_md_old = self.skills_dir / f"{skill_name}.md"
            if not skill_md_old.exists():
                raise SkillError(f"Skill not found: {skill_name}")

        # 重试循环：支持 Agent 多轮修复代码
        last_error = None
        for attempt in range(max_retries + 1):
            all_messages = []
            ai_responses = []
            
            # 构建 prompt（首次用原始 prompt，重试用修正 prompt）
            if attempt == 0:
                prompt = self._build_prompt(skill_name, *args)
            else:
                prompt = self._build_retry_prompt(skill_name, args, last_error, attempt)
                if callback:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    callback(timestamp, f"🔄 第 {attempt} 次重试修复...")

            try:
                async with asyncio.timeout(self.timeout):
                    async for message in query(prompt=prompt, options=options):
                        all_messages.append(str(message))
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    try:
                                        print(f"[AI Response] {block.text}", flush=True)
                                    except (UnicodeEncodeError, OSError):
                                        pass
                                    ai_responses.append(block.text)
                                    if callback:
                                        timestamp = datetime.now().strftime("%H:%M:%S")
                                        callback(timestamp, block.text)
                                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError as e:
                last_error = f"执行超时(>{self.timeout}秒)"
                if callback:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    callback(timestamp, f"❌ {last_error}")
                if attempt < max_retries:
                    continue
                raise SkillError(f"Skill '{skill_name}' {last_error}") from e
            except Exception as e:
                import traceback
                last_error = f"{e}\n{traceback.format_exc()}"
                if callback:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    callback(timestamp, f"❌ 执行出错: {last_error}")
                if attempt < max_retries:
                    continue
                raise SkillError(f"Skill '{skill_name}' 执行出错: {last_error}") from e

            # 检查输出文件（如果需要）
            output_file = None
            if check_output and len(args) >= 2:
                if skill_name == "suggest-data-filling":
                    if len(args) < 3:
                        raise SkillError(f"Skill '{skill_name}' 需要3个参数")
                    output_file = Path(args[2])
                else:
                    output_file = Path(args[1])

                if output_file and not output_file.exists():
                    # 检查是否是仿真脚本执行失败（需要查找错误信息）
                    last_error = self._extract_execution_error(all_messages, output_file)
                    if attempt < max_retries:
                        if callback:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            callback(timestamp, f"⚠️ 输出不完整: {last_error[:200]}...")
                        continue
                    raise SkillError(
                        f"Skill 执行完成，但未生成预期的输出文件: {output_file}"
                    )
                
                # 仿真任务特殊检查：args[1] 是目录，检查内部是否有实际输出文件
                if skill_name == "plant-simulation" and len(args) >= 3:
                    task_type = args[2]
                    if task_type in ("new_station", "old_station"):
                        output_dir = Path(args[1])
                        # 检查是否有实际仿真输出文件
                        csv_files = list(output_dir.glob("*.csv"))
                        xlsx_files = list(output_dir.glob("*.xlsx"))
                        if not csv_files and not xlsx_files:
                            # 目录存在但没有输出文件 -> 仿真脚本执行失败
                            last_error = self._extract_execution_error(all_messages, output_dir)
                            if attempt < max_retries:
                                if callback:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    callback(timestamp, f"⚠️ 仿真输出文件缺失，触发重试: {last_error[:200]}...")
                                continue
                            raise SkillError(
                                f"仿真执行失败，输出目录 {output_dir} 中没有生成 CSV/Excel 文件"
                            )

            # 成功！返回结果
            result_text = "\n".join(all_messages)
            return {
                "success": True,
                "ai_responses": ai_responses,
                "all_messages": all_messages,
                "result_text": result_text,
                "attempts": attempt + 1
            }

        # 不应到达这里
        raise SkillError(f"Skill '{skill_name}' 重试 {max_retries} 次后仍然失败")

    def create_message(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """使用 Claude Agent SDK 做一次纯文本 LLM 调用（同步包装器）

        禁用工具与技能加载，只要 AssistantMessage 的文本输出。
        接口与 DirectLLMClient.create_message 兼容：prompt -> 文本。

        Args:
            prompt: 用户提示词
            system: 系统提示词（可选）
            max_tokens: 保留参数（SDK 不直接使用，仅为接口兼容）
            model: 模型名（可选，默认读 ANTHROPIC_MODEL 环境变量）

        Returns:
            LLM 返回的文本

        Raises:
            SkillError: 调用失败或返回空响应
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._create_message_async(prompt, system=system, model=model)
        )

    async def _create_message_async(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """纯文本 LLM 调用的异步实现"""
        options = ClaudeAgentOptions(
            cwd=str(self.project_root),
            setting_sources=["user"],     # 加载 user 设置（代理 API 需要），不加载项目设置
            allowed_tools=[],            # 禁用工具，只要文本回复
            max_turns=1,                 # 单轮
            system_prompt=system,
            model=model or os.getenv("ANTHROPIC_MODEL") or None,
            permission_mode="bypassPermissions",
        )

        texts: List[str] = []
        try:
            async with asyncio.timeout(self.timeout):
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                texts.append(block.text)
        except asyncio.TimeoutError as e:
            raise SkillError(f"LLM 调用超时(>{self.timeout}秒)") from e
        except Exception as e:
            raise SkillError(f"LLM 调用出错: {e}") from e

        result = "".join(texts).strip()
        if not result:
            raise SkillError("LLM 返回空响应")
        return result

    def _build_prompt(self, skill_name: str, *args) -> str:
        """构建 Skill 调用提示词

        根据不同的 skill 构建合适的提示词
        """
        if skill_name == "extract-and-check-data":
            if len(args) < 2:
                raise SkillError(
                    "extract-and-check-data 需要至少2个参数: 输入Excel路径, 输出JSON路径, [原始文件名]"
                )
            prompt = (
                f"请使用 extract-and-check-data skill 从调研表中提取数据并进行质量检查。\n"
                f"请使用以下命令执行脚本：\n"
                f"python .claude/skills/extract-and-check-data/scripts/extract_and_check_data.py {args[0]} {args[1]}"
            )
            # 如果提供了原始文件名,添加到提示词中
            if len(args) >= 3:
                prompt += f" {args[2]}"
                prompt += (
                    f"\n命令的第三个参数 {args[2]} 是用户上传时的原始文件名，"
                    f"请确保它被原样传入，脚本会用它做项目地区推断。"
                )
            prompt += "\n"
            prompt += "请确保输出包含 extracted_data 和 quality_report 两部分。"
            return prompt

        elif skill_name == "extract-survey-data":
            if len(args) < 2:
                raise SkillError(
                    "extract-survey-data 需要2个参数: 输入Excel路径, 输出JSON路径"
                )
            return (
                f"请使用 extract-survey-data skill 从调研表中提取数据。\n"
                f"请使用以下命令执行脚本：\n"
                f"python .claude/skills/extract-survey-data/scripts/extract_survey_data.py {args[0]} {args[1]}\n"
                f"请确保输出为有效的 JSON 格式。"
            )

        elif skill_name == "check-data-quality":
            if len(args) < 2:
                raise SkillError(
                    "check-data-quality 需要2个参数: 提取数据JSON路径, 输出报告路径"
                )
            return (
                f"请使用 check-data-quality skill 检查数据质量。\n"
                f"请使用以下命令执行脚本：\n"
                f"python .claude/skills/check-data-quality/scripts/check_data_quality.py {args[0]} {args[1]}\n"
                f"请生成包含完整率、置信度和问题列表的质量报告。"
            )

        elif skill_name == "suggest-data-filling":
            if len(args) < 3:
                raise SkillError(
                    "suggest-data-filling 需要3个参数: "
                    "提取数据路径, 质量报告路径, 输出建议路径"
                )
            return (
                f"请使用 suggest-data-filling skill 生成数据填充建议。\n"
                f"请使用以下命令执行脚本：\n"
                f"python .claude/skills/suggest-data-filling/scripts/suggest_data_filling.py {args[0]} {args[1]} {args[2]}\n"
                f"请基于季节模式和插值法提供智能填充建议。"
            )

        # extract-air-compressor-data-llm 不再走 Agent/提示词路径：
        # P0 改造后由 _extract_air_compressor_llm 在进入 Agent 循环前直调，
        # 模型只返回 JSON 文本，无 Read/Write/Bash 权限面。

        elif skill_name == "plant-simulation":
            if len(args) < 3:
                raise SkillError(
                    "plant-simulation 需要3个参数: "
                    "输入数据路径, 输出路径/目录, 任务类型"
                )
            task_type = args[2]

            if task_type == "extract_topology":
                text_dump_hint = ""
                if len(args) >= 4:
                    text_dump_hint = (
                        f"\n\n已预先读取调研表全部内容，文本转储文件: {args[3]}\n"
                        f"请先用 Read 工具读取 {args[3]} 查看完整内容，"
                        f"从中识别设备规格表（包含类型、品牌、型号、台数、制冷量、功率、COP等列），"
                        f"提取真实的设备数据行，按 SKILL.md 规范输出结构化 JSON。"
                    )
                return (
                    f"请使用 plant-simulation skill 从调研表中提取冷站设备拓扑信息。\n"
                    f"调研表路径: {args[0]}\n"
                    f"输出JSON路径: {args[1]}\n"
                    f"请读取Excel文件，识别冷水主机、冷却泵、冷冻泵、冷却塔和特殊结构信息，"
                    f"按型号汇总台数，输出结构化JSON。"
                    f"{text_dump_hint}"
                )
            elif task_type == "parse_topology":
                return (
                    f"请使用 plant-simulation skill 解析用户输入的设备拓扑描述。\n"
                    f"用户输入文件: {args[0]}\n"
                    f"输出JSON路径: {args[1]}\n"
                    f"请将自然语言描述解析为结构化设备拓扑JSON。"
                )
            elif task_type == "parse_cop_curve":
                return (
                    f"请使用 plant-simulation skill 识别性能曲线图片中的COP数据。\n"
                    f"图片路径: {args[0]}\n"
                    f"输出JSON路径: {args[1]}\n"
                    f"请读取图片，提取冷却水温度×负荷率的二维COP查找表。"
                )
            elif task_type == "generate_operating":
                return (
                    f"请生成月度运行组合表格并写入 JSON 文件。\n\n"
                    f"## 输入数据\n{args[0]}\n\n"
                    f"## 输出路径\n{args[1]}\n\n"
                    f"## 任务\n"
                    f"1. 根据输入数据中的运行规律描述，整理为 12 个月的运行组合\n"
                    f"2. 每个月包含：\n"
                    f"   - 工作日运行: on_units(开机台数), load_ratio(负载率), description(描述)\n"
                    f"   - 休息日运行: on_units(开机台数), load_ratio(负载率), description(描述)\n"
                    f"   - 季节: winter/transition/summer\n"
                    f"   - representative_load_ratio: 代表性负载率\n"
                    f"3. 必须使用 Write 工具将结果写入 {args[1]}\n\n"
                    f"## 输出 JSON 格式\n"
                    f'{{"monthly_table": [{{"month": 1, "season": "winter", '
                    f'"weekday_operation": {{"on_units": 0, "load_ratio": 0, "description": "不开机"}}, '
                    f'"weekend_operation": {{"on_units": 0, "load_ratio": 0, "description": "不开机"}}, '
                    f'"representative_load_ratio": 0, "cooling_demand_kw": 0, "note": "冬季"}}]}}\n\n'
                    f"## 重要\n"
                    f"- 必须立即使用 Write 工具写入文件，不要只输出 JSON 文本\n"
                    f"- cooling_demand_kw 可以填 0（后续由代码自动计算）"
                )
            elif task_type in ("new_station", "old_station"):
                mode_desc = "新站房仿真（需求冷量+COP计算）" if task_type == "new_station" else "旧站房仿真（基准+优化对比节能率）"

                # P1 改造：校准/验证分离（改进方案第六节——"按目标调参"不是独立验证）
                # 输入 JSON 的 simulation_intent 决定本仿真是校准还是验证
                sim_intent = "verify"
                try:
                    with open(args[0], "r", encoding="utf-8") as _f:
                        sim_intent = (json.load(_f).get("simulation_intent") or "verify").lower()
                except Exception:
                    pass

                if task_type == "old_station":
                    if sim_intent == "calibrate":
                        intent_hint = (
                            "\n## 仿真模式：校准（calibrate）\n"
                            "- 本次允许按 target_energy_saving_rate 调整优化参数（如冷冻泵最低频率、目标温差）"
                            "使仿真节能率与目标误差<1%\n"
                            "- 报告必须标注：本仿真为校准模式，节能率按评估模块目标对齐\n"
                        )
                    else:
                        intent_hint = (
                            "\n## 仿真模式：独立验证（verify，默认）\n"
                            "- 优化参数必须依据拓扑/设备/工况数据冻结设定，"
                            "严禁为了对齐 target_energy_saving_rate 而调整参数\n"
                            "- 报告必须如实标注：`仿真节能率 X.X%，评估模块节能率 Y.Y%，偏差 Z.Z%`，"
                            "偏差大就是大，不得通过调参缩小\n"
                            "- 这是独立验证：仿真结论的价值恰恰在于与评估结果相互印证或暴露分歧\n"
                        )
                    report_hint = (
                        "\n## 旧站房特别要求\n"
                        "- 报告标题格式: `# {项目名称}中央空调冷站 控制策略优化仿真报告`\n"
                        "- 报告文件名: `{项目名称}_COP性能仿真分析报告.md`\n"
                        "- 报告内容由 AI 根据仿真数据动态生成，参考七章节结构\n"
                        f"{intent_hint}"
                    )
                else:
                    report_hint = ""
                return (
                    f"请使用 plant-simulation skill 生成并执行冷站仿真代码。\n"
                    f"输入数据(汇总JSON): {args[0]}\n"
                    f"输出目录: {args[1]}\n"
                    f"仿真模式: {mode_desc}\n\n"
                    f"请根据 SKILL.md 规范：\n"
                    f"1. 读取输入JSON中的拓扑、COP数据和工况信息\n"
                    f"2. 生成完整的Python仿真脚本到输出目录\n"
                    f"3. **执行前先用 ast.parse 验证脚本语法**\n"
                    f"4. 语法无误后执行脚本\n"
                    f"5. 确保输出8760h CSV、月度Excel、3张分析图表和Markdown报告\n\n"
                    f"## 代码质量要求\n"
                    f"- 多行文本必须用三引号，禁止在单引号中嵌入未转义的换行\n"
                    f"- 报告内容用列表拼接 + '\\n'.join(lines) 写入\n"
                    f"- 生成脚本后先运行: python -c \"import ast; ast.parse(open('{args[1]}/simulate_*.py').read())\"\n"
                    f"- 语法检查通过后再执行脚本"
                    f"{report_hint}"
                )
            else:
                return (
                    f"请使用 plant-simulation skill 完成任务。\n"
                    f"输入: {args[0]}\n"
                    f"输出: {args[1]}\n"
                    f"任务类型: {task_type}"
                )

        else:
            # 通用提示词
            params_str = ', '.join(str(arg) for arg in args)
            return (
                f"请使用 {skill_name} skill 完成任务。\n"
                f"参数: {params_str}"
            )

    def _build_retry_prompt(self, skill_name: str, args: tuple, error_info: str, attempt: int) -> str:
        """构建重试修正 prompt，让 Agent 根据错误信息修复代码"""
        output_dir = str(args[1]) if len(args) >= 2 else "未知"
        
        # 截断错误信息，避免过长
        if len(error_info) > 2000:
            error_info = error_info[:2000] + "\n...(截断)"

        return (
            f"## ❌ 上次仿真执行失败（第 {attempt} 次重试）\n\n"
            f"### 错误信息\n```\n{error_info}\n```\n\n"
            f"### 修复步骤（必须严格执行）\n\n"
            f"1. **用 Read 工具**读取输出目录中的仿真脚本 (`{output_dir}/simulate_*.py`)\n"
            f"2. **定位错误位置**（通常是字符串换行、变量未定义、import缺失）\n"
            f"3. **用 Write 工具直接覆盖修复后的完整脚本**（不要请求权限，直接写入）\n"
            f"4. 用 Bash 执行: `python -c \"import ast; ast.parse(open('{output_dir}/simulate_*.py').read())\"`\n"
            f"5. 语法通过后，用 Bash 执行: `python {output_dir}/simulate_*.py`\n"
            f"6. 确认 {output_dir}/ 目录下生成了 *.csv 和 *.xlsx 文件\n\n"
            f"### 常见错误修复示例\n"
            f"- `SyntaxError: unterminated string literal` → 把 `'...'` 改为 `\"...\"` 或 `'''...'''`\n"
            f"- `NameError` → 检查变量是否在使用前定义\n"
            f"- `ModuleNotFoundError` → 用 try/except 包裹 import，缺失时降级\n\n"
            f"### 重要\n"
            f"- 你有 Write 工具权限，直接修复脚本，不要等待用户授权\n"
            f"- 报告写入必须用: `Path(path).write_text(chr(10).join(lines), encoding='utf-8')`\n"
            f"- 输出目录: {output_dir}"
        )

    def _extract_execution_error(self, all_messages: list, output_file: Path) -> str:
        """从 Agent 消息中提取执行错误信息"""
        error_lines = []
        for msg in all_messages:
            msg_str = str(msg)
            # 查找常见错误模式
            if any(kw in msg_str.lower() for kw in ['error', 'traceback', 'exception', 'failed', 'syntaxerror', 'nameerror', 'typeerror']):
                error_lines.append(msg_str)
        
        if error_lines:
            return "\n".join(error_lines[-5:])  # 取最后 5 条错误相关消息
        
        # 查找输出目录中已生成的脚本文件
        output_dir = output_file.parent if output_file.is_file() else output_file
        script_files = list(output_dir.glob("simulate_*.py")) if output_dir.exists() else []
        
        if script_files:
            return f"脚本已生成但未执行成功。脚本文件: {script_files[0].name}\n输出目录: {output_dir}"
        
        return f"输出文件不存在: {output_file}"