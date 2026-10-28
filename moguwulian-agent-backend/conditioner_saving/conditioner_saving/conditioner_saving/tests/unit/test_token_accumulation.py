"""P1 审查报告专项修复测试（二）：Agent Token 用量不再重复累计

对应审查报告 [P1] Agent Token 用量会被重复累计：
- 每步的 decision 用量由 add_step() 实时累计进数据库（权威口径）；
- _finalize_run 不再叠加本轮 trace 用量（新任务计两次、恢复任务再叠加）。
同时覆盖"被白名单拒绝的步骤同样落库"（防漏计）。
"""
import json

import pytest

from core.session_manager import SessionManager


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from apps.api import config, db as db_mod
    from apps.api.storage import PersistentSessionStore

    monkeypatch.setattr(config.settings, "database_url",
                        f"sqlite:///{tmp_path / 'tokens.db'}")
    monkeypatch.setattr(db_mod, "_engine", None)
    s = PersistentSessionStore(manager=SessionManager(base_dir=tmp_path / "tmp"))
    # agent_service 通过模块级 session_store 落库，替换为独立实例
    import apps.api.agent_service as svc
    monkeypatch.setattr(svc, "session_store", s)
    yield s
    monkeypatch.setattr(db_mod, "_engine", None)


class _StubTrace:
    def __init__(self, usage):
        self.llm_usage = usage


class _StubCtx(dict):
    pass


class _StubOrchestrator:
    """_finalize_run 需要的最小编排器桩"""

    def __init__(self, trace_usage: dict, ctx: dict | None = None):
        self.trace = _StubTrace(trace_usage)
        self.ctx = ctx or {}

    def hydrate_from_session(self):  # resume 路径需要
        return 0


def _step(step_no: int, tool: str, tokens: int | None):
    step = {"step": step_no, "tool": tool, "args": {}, "ok": True,
            "observation": {}}
    if tokens is not None:
        step["decision"] = {"brain_calls": 1, "input_tokens": tokens,
                            "output_tokens": tokens // 10}
    return step


def test_finalize_run_does_not_double_count(store, tmp_path):
    """新任务：DB 用量 = 各步累计；finalize 不再叠加本轮 trace 用量"""
    import apps.api.agent_service as svc

    sid = store.register(tmp_path / "tmp" / "t1", "a.xlsx", "u1", "air_conditioner")
    run_id = store.create_run(sid, "llm")

    # 两个决策步，各 100 input tokens（add_step 实时累计）
    store.add_step(run_id, _step(1, "extract_survey", 100))
    store.add_step(run_id, _step(2, "finish", None))
    step_recorded = {"brain_calls": 1, "input_tokens": 100, "output_tokens": 10}

    # 编排器 trace 的本轮用量与步进记录一致（重复累计的来源）
    stub = _StubOrchestrator(trace_usage=dict(step_recorded),
                             ctx={"result": {"ok": True}})
    svc._finalize_run(run_id, sid, stub, {"status": "completed", "summary": "完成"})

    run = store.get_run(run_id)
    assert run["llm_usage"] == step_recorded, \
        f"finalize 后用量被重复累计: {run['llm_usage']}"


def test_finalize_run_resume_accumulates_exactly_once(store, tmp_path):
    """断点恢复：两轮步进累计 = 最终用量，不再叠加第二轮 trace"""
    import apps.api.agent_service as svc

    sid = store.register(tmp_path / "tmp" / "t2", "a.xlsx", "u1", "air_conditioner")
    run_id = store.create_run(sid, "llm")

    # 第一轮（中断前）
    store.add_step(run_id, _step(1, "extract_survey", 100))
    store.update_run(run_id, status="interrupted", summary="中断")

    # 第二轮（恢复续跑）——编排器 trace 只含本轮增量
    store.add_step(run_id, _step(2, "calculate_saving_rate", 200))
    store.add_step(run_id, _step(3, "finish", 50))
    total = {"brain_calls": 3, "input_tokens": 350, "output_tokens": 35}

    stub = _StubOrchestrator(
        trace_usage={"brain_calls": 2, "input_tokens": 250, "output_tokens": 25},
        ctx={"result": {"ok": True}},
    )
    svc._finalize_run(run_id, sid, stub, {"status": "completed", "summary": "完成"})

    run = store.get_run(run_id)
    assert run["llm_usage"] == total, \
        f"恢复场景用量失真: {run['llm_usage']} != {total}"


def test_denied_step_is_persisted(store, tmp_path):
    """决策越出白名单被拒绝的步骤同样经 on_step 落库（用量不漏计）"""
    from agent.orchestrator import AgentOrchestrator
    from agent.brain import BrainDecision

    class DenyBrain:
        def __init__(self):
            self.calls = 0

        def decide(self, state, available_tools, registry):
            self.calls += 1
            # 故意选择白名单外的工具 → 触发硬约束拒绝路径
            return BrainDecision(tool="bash", args={"cmd": "rm -rf /"},
                                 reason="越权尝试",
                                 llm_meta={"input_tokens": 77, "output_tokens": 7})

    sid = store.register(tmp_path / "tmp" / "t3", "a.xlsx", "u1", "air_conditioner")
    persisted = []
    orch = AgentOrchestrator(
        session_id=sid, session_dir=tmp_path / "tmp" / "t3",
        survey_path=tmp_path / "tmp" / "t3" / "a.xlsx",
        original_filename="a.xlsx",
        brain=DenyBrain(), max_steps=1,
        on_step=lambda step: persisted.append(step),
    )
    orch.run()

    assert len(persisted) == 1, "被拒绝的步骤必须同样落库"
    assert persisted[0]["decision"]["input_tokens"] == 77, "拒绝步骤的 token 不应漏计"
