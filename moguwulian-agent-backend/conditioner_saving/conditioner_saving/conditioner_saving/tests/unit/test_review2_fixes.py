"""审查报告2 专项修复测试

对应审查报告2 的四个问题：
- [P1] 空压机纯文本 LLM 路径没有向模型提供 SKILL.md 内容，且浅层校验接受
  空项目/空设备且 shouldReject=false，绕过必填字段和打回规则
- [P1] 身份仅含 id 不含 sub 时会话列表过滤失效（存储层收到 None 不过滤，
  普通用户可枚举全部会话）
- [P1] 心跳续租失败（锁已被其他实例接管）后任务仍继续执行
- [P2] 抢锁后 create_run/update_run/线程创建失败会遗留一小时任务锁
"""
import asyncio
import json
import shutil
import threading
from pathlib import Path

import pytest

from core.session_manager import SessionManager
from core.skill_caller import SkillError, SkillCaller, _validate_air_compressor_payload
from apps.api.lock_lease import LockLostError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def fresh_store(tmp_path, monkeypatch):
    """独立 SQLite 库 + 独立会话目录的存储实例（模拟全新进程）"""
    from apps.api import config, db as db_mod
    from apps.api.storage import PersistentSessionStore

    monkeypatch.setattr(config.settings, "database_url",
                        f"sqlite:///{tmp_path / 'locks.db'}")
    monkeypatch.setattr(db_mod, "_engine", None)
    store = PersistentSessionStore(manager=SessionManager(base_dir=tmp_path / "tmp"))
    yield store
    monkeypatch.setattr(db_mod, "_engine", None)


# ==================== P1-1 空压 LLM 提取：SKILL.md 注入 + 打回规则服务端执行 ====================

def _valid_payload() -> dict:
    """满足 SKILL.md 全部打回条件的最小合法载荷"""
    return {
        "rawData": {
            "projectLevel": {
                "companyName": "示例公司",
                "projectLocation": "广东省惠州市",
                "industryType": "其他",
                "totalAnnualEnergyConsumption": 1200000,
                "electricityPrice": 0.69,
                "alarmPressureLowerLimit": 6.5,
                "averagePressure": 7.2,
                "dailyRunningHours": 10,
                "annualRunningDays": 330,
            },
            "equipmentLevel": [
                {
                    "stationId": "站房A",
                    "equipmentId": "AC-01",
                    "equipmentType": "螺杆",
                    "brand": "阿特拉斯",
                    "model": "GA55",
                    "manufactureYear": 2020,
                    "runningStatus": "运行",
                    "ratedPower": 55,
                    "ratedFlow": 10.5,
                },
                {
                    "stationId": "站房A",
                    "equipmentId": "AC-02",
                    "equipmentType": "螺杆",
                    "brand": "寿力",
                    "model": "LS75",
                    "manufactureYear": 2018,
                    "runningStatus": "备用",
                    "ratedPower": 75,
                    "ratedFlow": 13,
                },
            ],
        },
        "rejectionCheck": {"shouldReject": False, "criticalIssues": [], "highSeverityIssues": []},
    }


def test_empty_project_and_equipment_forced_reject():
    """审查报告2 点名的绕过场景：空项目 + 空设备 + shouldReject=false

    服务端按数据事实复评致命规则后必须强制打回，不允许继续错误测算。
    """
    payload = _valid_payload()
    payload["rawData"]["projectLevel"] = {}
    payload["rawData"]["equipmentLevel"] = []
    payload["rejectionCheck"] = {"shouldReject": False, "criticalIssues": [], "highSeverityIssues": []}

    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is True
    assert len(out["rejectionCheck"]["criticalIssues"]) >= 4  # 报警压力/平均压力/年用量/无运行设备


def test_missing_critical_fields_forced_reject():
    """三个关键运行指标（报警压力/平均压力/年用量）任一缺失即致命打回"""
    for field in ("alarmPressureLowerLimit", "averagePressure", "totalAnnualEnergyConsumption"):
        payload = _valid_payload()
        payload["rawData"]["projectLevel"][field] = None
        out = _validate_air_compressor_payload(payload)
        assert out["rejectionCheck"]["shouldReject"] is True, f"{field} 缺失必须打回"


def test_zero_or_nonpositive_critical_values_rejected():
    """关键指标为 0/负数同样视为缺失（SKILL.md: 为空或 ≤0）"""
    payload = _valid_payload()
    payload["rawData"]["projectLevel"]["averagePressure"] = 0
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is True


def test_valid_payload_not_rejected():
    """数据完整时保持 shouldReject=false（不能误伤正常测算）"""
    out = _validate_air_compressor_payload(_valid_payload())
    assert out["rejectionCheck"]["shouldReject"] is False
    assert out["rejectionCheck"]["criticalIssues"] == []


def test_no_running_device_rejected():
    """全部设备都不是运行/备用状态 → 致命打回"""
    payload = _valid_payload()
    for eq in payload["rawData"]["equipmentLevel"]:
        eq["runningStatus"] = "故障"
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is True


def test_missing_station_id_rejected():
    """任意设备站房编码为空 → 致命打回"""
    payload = _valid_payload()
    payload["rawData"]["equipmentLevel"][1]["stationId"] = ""
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is True


def test_duplicate_equipment_id_in_same_station_rejected():
    """同一站房内设备编码重复 → 致命打回；不同站房允许同编码"""
    payload = _valid_payload()
    payload["rawData"]["equipmentLevel"][1]["equipmentId"] = "AC-01"  # 同站房重复
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is True

    payload = _valid_payload()
    payload["rawData"]["equipmentLevel"][1]["stationId"] = "站房B"
    payload["rawData"]["equipmentLevel"][1]["equipmentId"] = "AC-01"  # 跨站房同编码允许
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is False


def test_model_reported_issues_merged_not_trusted():
    """模型自报 criticalIssues 但 shouldReject=false → 服务端强制打回并保留报告"""
    payload = _valid_payload()
    payload["rejectionCheck"] = {
        "shouldReject": False,
        "criticalIssues": ["模型自定义的致命问题"],
        "highSeverityIssues": [],
    }
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is True
    assert "模型自定义的致命问题" in out["rejectionCheck"]["criticalIssues"]


def test_high_severity_recorded_without_fatal_reject():
    """高严重性问题（运行设备缺功率/流量/类型）记录 issues 但不触发致命打回"""
    payload = _valid_payload()
    payload["rawData"]["equipmentLevel"][0]["ratedFlow"] = None
    payload["rawData"]["equipmentLevel"][1]["equipmentType"] = None
    out = _validate_air_compressor_payload(payload)
    assert out["rejectionCheck"]["shouldReject"] is False
    high = out["rejectionCheck"]["highSeverityIssues"]
    assert len(high) == 2


def test_nonlist_issues_normalized():
    """模型返回的 issues 不是数组时规范化，不让异常类型炸掉校验"""
    payload = _valid_payload()
    payload["rejectionCheck"]["criticalIssues"] = "单个字符串问题"
    payload["rejectionCheck"]["highSeverityIssues"] = {"a": 1}
    out = _validate_air_compressor_payload(payload)
    # 字符串型 issue 被保留成列表
    assert out["rejectionCheck"]["criticalIssues"] == ["单个字符串问题"]


@pytest.fixture()
def skill_root(tmp_path):
    """带空压技能 SKILL.md 的临时项目根（无预提取脚本 → 跳过脚本步骤）"""
    root = tmp_path / "proj"
    skill_dir = root / ".claude" / "skills" / "extract-air-compressor-data-llm"
    skill_dir.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / ".claude" / "skills" / "extract-air-compressor-data-llm"
                / "SKILL.md", skill_dir / "SKILL.md")
    return root


def test_llm_prompt_contains_skill_md(skill_root, tmp_path, monkeypatch):
    """纯文本 LLM 路径必须把 SKILL.md 内容注入提示词（审查报告2 P1-1）"""
    caller = SkillCaller(project_root=skill_root)
    out_path = tmp_path / "extracted_air_compressor_data.json"
    text_tmp = out_path.with_suffix(".text.json")
    text_tmp.write_text(json.dumps({"text_content": "站房A 设备1 铭牌功率 55kW"}),
                        encoding="utf-8")

    captured = {}

    async def fake_create_message(prompt, **kw):
        captured["prompt"] = prompt
        return json.dumps(_valid_payload(), ensure_ascii=False)

    monkeypatch.setattr(caller, "_create_message_async", fake_create_message)
    result = asyncio.run(caller._extract_air_compressor_llm(
        str(tmp_path / "in.xlsx"), str(out_path), "调研表.xlsx"))

    prompt = captured["prompt"]
    # SKILL.md 的打回条件清单正文必须出现在提示词里
    assert "打回条件检查" in prompt
    assert "shouldReject = true" in prompt
    assert "压缩机功率因子查询表" in prompt
    # 调研表文本与原始文件名也在
    assert "站房A 设备1 铭牌功率 55kW" in prompt
    assert "调研表.xlsx" in prompt
    # 输出经服务端校验后写盘
    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8"))["rawData"]["equipmentLevel"]
    assert result["success"] is True


def test_llm_path_fails_without_skill_md(tmp_path):
    """SKILL.md 缺失视为部署错误直接失败，不再让模型空谈"按 SKILL.md 规范\""""
    root = tmp_path / "empty_proj"
    (root / ".claude" / "skills").mkdir(parents=True)
    caller = SkillCaller(project_root=root)
    with pytest.raises(SkillError):
        asyncio.run(caller._extract_air_compressor_llm(
            str(tmp_path / "in.xlsx"), str(tmp_path / "out.json"), "f.xlsx"))


# ==================== P1-2 身份归一化：仅含 id 的身份过滤不失效 ====================

def test_user_key_prefers_sub():
    from apps.api.deps import user_key

    assert user_key({"sub": "sub-1", "id": "id-1"}) == "sub-1"
    assert user_key({"id": "id-1"}) == "id-1"
    assert user_key({"sub": "sub-1"}) == "sub-1"
    assert user_key({}) is None


def test_id_only_identity_cannot_enumerate_others_sessions(fresh_store, monkeypatch):
    """审查报告2 复现场景：身份只有 id 时，/sessions 只能看到自己的会话"""
    from apps.api import routers

    monkeypatch.setattr(routers, "session_store", fresh_store)

    mine = fresh_store.register(session_dir=tmp_dir(fresh_store), original_filename="mine.xlsx",
                                user_id="id-only-user", module_type="air_conditioner")
    other = fresh_store.register(session_dir=tmp_dir(fresh_store), original_filename="other.xlsx",
                                 user_id="someone-else", module_type="air_conditioner")

    # 仅含 id 的身份（无 sub）：列表必须按同一口径过滤
    sessions = routers.list_sessions(user={"id": "id-only-user"})
    ids = {s["session_id"] for s in sessions}
    assert mine in ids
    assert other not in ids, "普通用户不能枚举其他用户的会话"

    # sub 身份口径不受影响
    sessions = routers.list_sessions(user={"sub": "someone-else"})
    ids = {s["session_id"] for s in sessions}
    assert other in ids
    assert mine not in ids


def test_ensure_session_accepts_id_only_owner(fresh_store, monkeypatch):
    """ACL 归属校验同口径：仅含 id 的归属人访问自己的会话不再 403"""
    from apps.api import config
    from apps.api.deps import ensure_session_access

    monkeypatch.setattr(config.settings, "require_auth", True)
    sid = fresh_store.register(session_dir=tmp_dir(fresh_store), original_filename="a.xlsx",
                               user_id="id-only-user", module_type="air_conditioner")
    s = fresh_store.get(sid)
    # 归属人本人（只有 id）放行
    ensure_session_access(s, {"id": "id-only-user"}, action="test")
    # 他人被拒
    with pytest.raises(Exception):
        ensure_session_access(s, {"id": "intruder"}, action="test")


def tmp_dir(store):
    return store.manager.create_session_directory()


# ==================== P1-3 锁易主：续租失败后任务必须停止 ====================

def test_lock_lease_marks_lost_on_failed_renew():
    """renew_task 返回 False（锁被新实例接管）→ lock_lost 置位、check 抛错"""
    from apps.api.lock_lease import LockLease, LockLostError

    lost_messages = []
    lease = LockLease("extract", "s1", "tok", ttl_seconds=60,
                      renew=lambda *a, **k: False,
                      on_lost=lost_messages.append, interval=0.02)
    lease.start()
    assert lease.lock_lost.wait(timeout=2), "续租失败必须置位 lock_lost"
    with pytest.raises(LockLostError):
        lease.check()
    assert any("接管" in m for m in lost_messages)
    lease.stop.set()


def test_lock_lease_tolerates_transient_renew_errors():
    """瞬时 DB 异常不算易主：任务继续、锁不丢"""
    from apps.api.lock_lease import LockLease, LockLostError

    def flaky_renew(*a, **k):
        raise RuntimeError("db down")

    lease = LockLease("extract", "s1", "tok", ttl_seconds=60,
                      renew=flaky_renew, interval=0.02)
    lease.start()
    import time
    time.sleep(0.2)
    assert not lease.lock_lost.is_set()
    lease.check()  # 不抛
    lease.stop.set()


def test_lock_lease_renews_while_alive():
    """正常续租路径：renew 周期性被调用、锁不判易主"""
    from apps.api.lock_lease import LockLease

    calls = []
    lease = LockLease("extract", "s1", "tok", ttl_seconds=60,
                      renew=lambda *a, **k: calls.append(1) or True, interval=0.02)
    lease.start()
    import time
    time.sleep(0.15)
    lease.stop.set()
    assert len(calls) >= 2, f"心跳应多次续租，实际 {len(calls)} 次"
    assert not lease.lock_lost.is_set()


def test_orchestrator_aborts_when_lock_lost():
    """锁易主后编排器在步骤边界停止：不再执行工具、不再写状态"""
    from agent.brain import BrainDecision
    from agent.orchestrator import AgentOrchestrator
    from agent.tools import Tool, ToolContext, ToolRegistry

    executed = []

    def handler(ctx, args):
        executed.append(1)
        return {}

    registry = ToolRegistry()
    registry.register(Tool(name="extract_survey", description="",
                           input_schema={"type": "object"}, handler=handler))

    class OneStepBrain:
        def decide(self, state, available_tools, registry_):
            return BrainDecision(tool="extract_survey", args={}, reason="")

    def abort_check():
        raise LockLostError("任务锁已被其他实例接管")

    orch = AgentOrchestrator(
        session_id="s1", session_dir=Path("."), survey_path=Path("x.xlsx"),
        original_filename="x.xlsx", brain=OneStepBrain(), registry=registry,
        abort_check=abort_check,
    )
    with pytest.raises(LockLostError):
        orch.run()
    assert not executed, "锁易主后不应执行任何工具"


def test_agent_worker_interrupted_on_lock_lost(fresh_store, monkeypatch):
    """_run_worker 捕获 LockLostError：run 记录置 interrupted、不动 stage"""
    from apps.api import agent_service

    monkeypatch.setattr(agent_service, "session_store", fresh_store)
    session_dir = tmp_dir(fresh_store)
    (session_dir / "survey.xlsx").write_bytes(b"stub")
    sid = fresh_store.register(session_dir=session_dir, original_filename="survey.xlsx",
                               user_id="u", module_type="air_conditioner")
    run_id = fresh_store.create_run(sid, "deterministic")

    def boom(*a, **k):
        from apps.api.lock_lease import LockLostError
        raise LockLostError("任务锁已被其他实例接管，停止执行: agent:s1")

    # 编排器构造成功后 run() 抛 LockLostError
    class FakeOrch:
        def __init__(self, **kw):
            assert kw.get("abort_check") is not None, "必须传入 abort_check"

        def hydrate_from_session(self):
            pass

        def run(self):
            boom()

    monkeypatch.setattr(agent_service, "AgentOrchestrator", FakeOrch)
    agent_service._run_worker(
        sid, run_id, brain=None, registry=None, resume_run=None,
        approval_granted=False, require_fill_approval=False,
        max_steps=10, start_step=1, main_loop=None,
        survey_path=session_dir / "survey.xlsx", original_filename="survey.xlsx",
        trace_path=session_dir / "trace.json", abort_check=boom,
    )
    run = fresh_store.get_latest_run(sid)
    assert run["status"] == "interrupted", "锁易主的中止应标记 interrupted（可恢复口径）"
    assert "接管" in (run["summary"] or "")
    assert fresh_store.get(sid)["stage"] != "error", "锁易主不应把会话 stage 置为 error"


# ==================== P2-4 抢锁后失败路径必须释放锁 ====================

def test_start_agent_run_releases_lock_when_create_run_fails(fresh_store, monkeypatch):
    """start_agent_run：create_run 失败 → 锁已释放，重试可立即抢占"""
    from apps.api import agent_service

    monkeypatch.setattr(agent_service, "session_store", fresh_store)
    session_dir = tmp_dir(fresh_store)
    (session_dir / "survey.xlsx").write_bytes(b"stub")
    sid = fresh_store.register(session_dir=session_dir, original_filename="survey.xlsx",
                               user_id="u", module_type="air_conditioner")

    def boom(*a, **k):
        raise RuntimeError("db write failed")

    monkeypatch.setattr(fresh_store, "create_run", boom)
    with pytest.raises(RuntimeError, match="db write failed"):
        agent_service.start_agent_run(sid, mode="deterministic")

    token = fresh_store.claim_task("agent", sid, ttl_seconds=60)
    assert token, "失败路径必须释放任务锁，否则重试一直返回'正在执行'"
    fresh_store.release_task("agent", sid, token=token)


def test_start_agent_run_releases_lock_when_thread_start_fails(fresh_store, monkeypatch):
    """start_agent_run：线程创建失败 → 锁已释放"""
    from apps.api import agent_service

    monkeypatch.setattr(agent_service, "session_store", fresh_store)
    session_dir = tmp_dir(fresh_store)
    (session_dir / "survey.xlsx").write_bytes(b"stub")
    sid = fresh_store.register(session_dir=session_dir, original_filename="survey.xlsx",
                               user_id="u", module_type="air_conditioner")

    real_thread = threading.Thread

    class BoomThread(real_thread):
        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr(agent_service.threading, "Thread", BoomThread)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        agent_service.start_agent_run(sid, mode="deterministic")

    token = fresh_store.claim_task("agent", sid, ttl_seconds=60)
    assert token, "线程启动失败必须释放任务锁"
    fresh_store.release_task("agent", sid, token=token)


def test_run_agent_sync_releases_lock_when_create_run_fails(fresh_store, monkeypatch):
    """run_agent_sync（批量路径）：create_run 失败 → finally 释放锁"""
    from apps.api import agent_service

    monkeypatch.setattr(agent_service, "session_store", fresh_store)
    session_dir = tmp_dir(fresh_store)
    (session_dir / "survey.xlsx").write_bytes(b"stub")
    sid = fresh_store.register(session_dir=session_dir, original_filename="survey.xlsx",
                               user_id="u", module_type="air_conditioner")

    def boom(*a, **k):
        raise RuntimeError("db write failed")

    monkeypatch.setattr(fresh_store, "create_run", boom)
    with pytest.raises(RuntimeError, match="db write failed"):
        agent_service.run_agent_sync(sid, mode="deterministic")

    token = fresh_store.claim_task("agent", sid, ttl_seconds=60)
    assert token, "同步路径同样必须释放任务锁"
    fresh_store.release_task("agent", sid, token=token)


def test_services_thread_launch_failure_releases_lock(fresh_store, monkeypatch):
    """services._launch_locked_thread：线程启动失败释放对应任务锁"""
    from apps.api import services

    monkeypatch.setattr(services, "session_store", fresh_store)
    token = fresh_store.claim_task("extract", "sx", ttl_seconds=60)
    assert token

    class FakeThread:
        def start(self):
            raise RuntimeError("no threads available")

    with pytest.raises(RuntimeError, match="no threads available"):
        services._launch_locked_thread("extract", "sx", token, FakeThread(),
                                       services._extract_tasks)

    token2 = fresh_store.claim_task("extract", "sx", ttl_seconds=60)
    assert token2, "线程启动失败必须释放任务锁"
    fresh_store.release_task("extract", "sx", token=token2)
