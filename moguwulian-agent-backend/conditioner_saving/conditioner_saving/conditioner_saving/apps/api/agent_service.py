"""Agent 模式服务层：桥接 AgentOrchestrator 与 FastAPI

职责：
1. 按环境选择决策核心（配置了 API Key → LLMBrain 走 LLM 网关；否则零成本 RuleBrain）
2. 在后台线程运行 agent 循环，事件经 asyncio.Queue 供 SSE 消费
3. 运行与每步落库（agent_runs / agent_steps），支持进程重启后断点恢复
4. Human-in-the-loop：needs_input / pending_approval 终态创建待办（agent_actions），
   审批批准或补充数据后从断点/重跑恢复
5. 运行结束后把产物同步回 session_store，兼容既有 /result、/download 端点
"""
import asyncio
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent import AgentOrchestrator, RuleBrain
from agent.tools import build_registry_for_module
from .deps import session_store
from .lock_lease import LockLease, LockLostError

DEFAULT_MAX_STEPS = 10

# session_id → agent 后台 thread
_agent_tasks: dict = {}


def is_agent_running(session_id: str) -> bool:
    t = _agent_tasks.get(session_id)
    return t is not None and t.is_alive()


def _select_brain(mode: Optional[str] = None, user_id: Optional[str] = None):
    """选择决策核心

    - mode="deterministic" 或未配置 API Key → RuleBrain（零 LLM 成本）
    - 其余 → LLMBrain，LLM 调用经 LLM 网关（预算/计量/熔断）
    """
    if mode == "deterministic":
        return RuleBrain(), "deterministic"
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from agent import LLMBrain
            from core.llm_gateway import LLMGateway
            gateway = LLMGateway(user_id=user_id, purpose_default="agent.brain",
                                 store=session_store)
            return LLMBrain(client=gateway), "llm"
        except Exception:
            return None, "deterministic"
    return None, "deterministic"


def _notify_event(session_id: str, event: dict, loop=None) -> None:
    """事件 → ai_logs + SSE 队列"""
    ts = datetime.now().strftime("%H:%M:%S")
    etype = event.get("type", "")
    if etype == "thinking":
        msg = f"🧠 {event.get('message', '')}"
    elif etype == "tool_start":
        msg = f"🔧 调用工具 {event.get('tool')}"
    elif etype == "tool_end":
        mark = "✅" if event.get("ok") else "❌"
        msg = f"{mark} 工具 {event.get('tool')} 执行完成"
        obs = event.get("observation") or {}
        if not event.get("ok") and obs.get("error"):
            msg += f"：{obs['error'][:200]}"
    elif etype in ("final", "error", "run_start"):
        msg = f"ℹ️ {event.get('message', '')}"
    else:
        msg = json_safe(event)
    session_store.append_ai_log(session_id, ts, msg)

    if loop is not None:
        from apps.api.services import get_ai_queue
        q = get_ai_queue(session_id)
        if q is not None:
            payload = {"type": etype, "timestamp": ts, **{
                k: v for k, v in event.items() if k != "type"
            }}
            loop.call_soon_threadsafe(q.put_nowait, payload)


def _finalize_run(run_id: str, session_id: str, orchestrator: AgentOrchestrator,
                  final: dict) -> None:
    """运行收尾：结果落库、产物同步回 session_store、创建 HITL 待办"""
    # P1 修复（审查报告）：token 用量不做二次合并。每步的 decision 用量
    # 已由 add_step() 实时累计进数据库 llm_usage（新任务与断点恢复同一路径），
    # 这里再叠加 orchestrator.trace.llm_usage（本轮 trace 用量）会把本轮
    # 用量重复计一次——新任务约计两倍、恢复任务再叠加一轮。
    # 数据库用量即权威口径，本函数只更新状态/摘要/结束时间。
    session_store.update_run(
        run_id, status=final["status"], summary=final.get("summary", ""),
        finished_at=datetime.now(),
    )

    s_live = session_store.get(session_id)
    formatted = orchestrator.ctx.get("formatted_result")
    result = orchestrator.ctx.get("result")
    if formatted is None and result is not None:
        try:
            filled = orchestrator.ctx.get("filled_data") or {}
            from core.excel_calculator import ExcelCalculator
            formatted = ExcelCalculator().format_export_results(filled, result)
        except Exception:
            formatted = None
    if s_live is not None:
        s_live["extracted_data"] = orchestrator.ctx.get("extracted_data")
        s_live["quality_report"] = orchestrator.ctx.get("quality_report")
        s_live["filling_suggestions"] = orchestrator.ctx.get("suggestions")
        s_live["result"] = result
        s_live["formatted_result"] = formatted

    # Human-in-the-loop：求助/审批 → 待办收件箱。
    # 注意顺序：必须先建待办、再置 stage——stage 是"本轮结束"的信号，
    # 轮询方看到 stage 后应能立刻查到待办（顺序颠倒会产生竞态窗口）。
    if final["status"] in ("needs_input", "pending_approval"):
        try:
            session_store.create_action(
                run_id=run_id, session_id=session_id,
                action_type="needs_input" if final["status"] == "needs_input" else "approval",
                reason=final.get("summary", ""),
                detail={"missing_items": final.get("missing_items", [])},
                user_id=(session_store.get(session_id) or {}).get("user_id"),
            )
        except Exception:
            pass

    if s_live is not None:
        stage = {"completed": "result", "needs_input": "suggest",
                 "pending_approval": "suggest"}.get(final["status"], "error")
        session_store.update_stage(session_id, stage)

    # 通知（M3）：完成/求助/审批/失败 推送 webhook（未配置则跳过）
    try:
        from .notify import notify_run_finished
        notify_run_finished(session_id, final)
    except Exception:
        pass


def run_agent_sync(session_id: str, mode: Optional[str] = None,
                   require_fill_approval: bool = False) -> dict:
    """同步执行一次 agent 主线（批量任务用；交互路径走 start_agent_run 线程）"""
    s = session_store.get(session_id)
    if s is None:
        raise RuntimeError(f"session {session_id!r} 不存在")
    session_dir: Path = s["session_dir"]
    original_filename: str = s["original_filename"] or ""
    survey_path = session_dir / original_filename
    if not survey_path.exists():
        raise RuntimeError(f"调研表文件缺失: {survey_path.name}")

    brain, brain_label = _select_brain(mode, user_id=s.get("user_id"))
    registry = build_registry_for_module(s.get("module_type", "air_conditioner"))

    # P1 修复（审查报告）：先抢锁、再建运行记录——原顺序在抢锁失败时
    # 会留下一条 running 幽灵记录。
    # P2 修复（审查报告2）：抢锁成功后的任何一步（create_run/update_run/
    # 编排器构建/执行）失败都必须释放锁，否则遗留一小时任务锁、重试一直
    # 返回"正在执行"。
    lock_token = session_store.claim_task("agent", session_id, ttl_seconds=3600)
    if not lock_token:
        raise RuntimeError(f"session {session_id!r} 的 agent 任务正在其他进程执行")

    try:
        run_id = session_store.create_run(session_id, brain_label)
        session_store.update_run(run_id, mode=brain_label)
        orchestrator = AgentOrchestrator(
            session_id=session_id, session_dir=session_dir,
            survey_path=survey_path, original_filename=original_filename,
            brain=brain, registry=registry,
            require_fill_approval=require_fill_approval,
            on_step=lambda step: session_store.add_step(run_id, step),
        )
        final = orchestrator.run()
        _finalize_run(run_id, session_id, orchestrator, final)
        return final
    finally:
        session_store.release_task("agent", session_id, token=lock_token)


def start_agent_run(
    session_id: str,
    mode: Optional[str] = None,
    main_loop: Optional[asyncio.AbstractEventLoop] = None,
    resume: bool = False,
    require_fill_approval: bool = False,
) -> Path:
    """启动 agent 运行线程，返回轨迹文件路径

    resume=True 时从最近一次 interrupted 的运行断点续跑
    （状态产物从会话目录文件恢复，步数与用量累计到同一 run 记录）。
    """
    s = session_store.get(session_id)
    if s is None:
        raise RuntimeError(f"session {session_id!r} 不存在")
    if is_agent_running(session_id):
        raise RuntimeError(f"session {session_id!r} 已有 agent 任务在运行")

    session_dir: Path = s["session_dir"]
    original_filename: str = s["original_filename"] or ""
    survey_path = session_dir / original_filename
    if not survey_path.exists():
        raise RuntimeError(f"调研表文件缺失: {survey_path.name}")

    _register_main_loop(main_loop)
    brain, brain_label = _select_brain(mode, user_id=s.get("user_id"))
    registry = build_registry_for_module(s.get("module_type", "air_conditioner"))

    # P1 修复（审查报告）：先抢锁、再建运行记录，避免抢锁失败留下幽灵记录
    # P2 修复（审查报告2）：抢锁后到线程真正接管锁之前（create_run /
    # update_run / 线程创建启动）任何一步失败都要释放锁，否则遗留一小时
    # 任务锁、重试持续返回"正在执行"。线程启动成功后由 worker 的 finally
    # 负责释放（本 except 不会触发——启动后语句不再可能抛异常）。
    lock_token = _acquire_agent_lock(session_id)
    try:
        run_id = session_store.create_run(session_id, brain_label)
        session_store.update_run(run_id, mode=brain_label)
        return _start_run_thread(
            session_id, run_id=run_id, brain=brain, registry=registry,
            require_fill_approval=require_fill_approval,
            main_loop=main_loop, survey_path=survey_path,
            original_filename=original_filename, lock_token=lock_token,
        )
    except BaseException:
        _release_agent_lock(session_id, lock_token)
        raise


def resume_agent_run(session_id: str, main_loop=None,
                     approval_granted: bool = False) -> Path:
    """从最近一次 interrupted / pending_approval 运行断点恢复

    approval_granted=True：恢复待审批的运行并跳过审批门（人工已批准填充）。
    """
    s = session_store.get(session_id)
    if s is None:
        raise RuntimeError(f"session {session_id!r} 不存在")
    if is_agent_running(session_id):
        raise RuntimeError(f"session {session_id!r} 已有 agent 任务在运行")

    statuses = ["interrupted", "pending_approval"] if approval_granted else ["interrupted"]
    run = session_store.get_latest_run(session_id, statuses=statuses)
    if run is None:
        raise RuntimeError("没有可恢复的中断或待审批运行")

    _register_main_loop(main_loop)
    # P2 修复（审查报告2）：同 start_agent_run——线程接管锁之前的失败
    # 必须释放锁
    lock_token = _acquire_agent_lock(session_id)
    try:
        return _start_run_thread(
            session_id, resume_run=run, approval_granted=approval_granted,
            main_loop=main_loop, lock_token=lock_token,
        )
    except BaseException:
        _release_agent_lock(session_id, lock_token)
        raise


def _release_agent_lock(session_id: str, token: Optional[str]) -> None:
    """释放 agent 任务锁（尽力而为：释放失败不掩盖原始异常）"""
    try:
        session_store.release_task("agent", session_id, token=token)
    except Exception:
        pass


def _acquire_agent_lock(session_id: str) -> str:
    """抢占 agent 跨进程任务锁（防多副本重复执行同一会话的 agent 主线）

    返回租约 token——释放与续租都必须携带该 token。
    """
    token = session_store.claim_task("agent", session_id, ttl_seconds=3600)
    if not token:
        raise RuntimeError(
            f"session {session_id!r} 的 agent 任务正在其他进程/实例中执行")
    return token


def _start_lock_heartbeat(task_type: str, session_id: str, token: str,
                          ttl_seconds: int) -> "LockLease":
    """启动锁心跳续租（审查报告2 P1-3：续租失败必须停止任务）

    每 ttl/3 续租一次；renew_task() 返回 False（锁已被新实例接管）时
    置位 lease.lock_lost 并写 ai_log，任务主体在步骤边界经 check()
    抛 LockLostError 协作式退出。
    """

    def on_lost(message: str) -> None:
        session_store.append_ai_log(
            session_id, datetime.now().strftime("%H:%M:%S"), message)

    lease = LockLease(task_type, session_id, token, ttl_seconds,
                      session_store.renew_task, on_lost=on_lost)
    return lease.start()


def _start_run_thread(
    session_id: str,
    run_id: Optional[str] = None,
    resume_run: Optional[dict] = None,
    brain=None,
    registry=None,
    require_fill_approval: bool = False,
    approval_granted: bool = False,
    main_loop=None,
    survey_path: Optional[Path] = None,
    original_filename: Optional[str] = None,
    lock_token: Optional[str] = None,
) -> Path:
    """组装并启动 agent 运行线程（新建运行 / 断点恢复共用）"""
    s = session_store.get(session_id)
    session_dir: Path = s["session_dir"]
    original_filename = original_filename if original_filename is not None else (s["original_filename"] or "")
    survey_path = survey_path or (session_dir / original_filename)
    trace_path = session_dir / "agent_trace.json"

    start_step = 1
    max_steps = DEFAULT_MAX_STEPS
    if resume_run is not None:
        run_id = resume_run["run_id"]
        start_step = (resume_run["step_count"] or 0) + 1
        max_steps = DEFAULT_MAX_STEPS + (resume_run["step_count"] or 0)
        approval_granted = approval_granted or resume_run["status"] == "pending_approval"
        # P2 修复：恢复运行必须重建原决策核心与工具注册表。
        # 原实现 resume 路径不带 brain/registry（为 None），编排器会退化到
        # 默认工具集——空压机任务可能按中央空调工具恢复，LLM 模式静默变规则模式。
        if brain is None or registry is None:
            orig_mode = resume_run.get("mode")
            brain, brain_label = _select_brain(
                None if orig_mode == "llm" else orig_mode,
                user_id=s.get("user_id"),
            )
            registry = build_registry_for_module(
                s.get("module_type", "air_conditioner"))
            session_store.append_ai_log(
                session_id, datetime.now().strftime("%H:%M:%S"),
                f"🔁 断点恢复：重建决策核心（{brain_label}）与"
                f"{s.get('module_type', 'air_conditioner')} 工具注册表",
            )
        session_store.update_run(run_id, status="running", summary=None,
                                 finished_at=None)
    else:
        session_store.update_run(run_id, mode=(brain.__class__.__name__ == "LLMBrain" and "llm" or "deterministic"))

    def worker() -> None:
        # P1 修复：持锁期间定期续租（心跳），超长任务不再被 TTL 误抢占；
        # 释放必须携带租约 token，避免误删（易主后的）他人锁。
        # 续租失败（锁易主）时 lease.lock_lost 置位，编排器在步骤边界
        # 经 abort_check 协作式停止（审查报告2 P1-3）。
        lease = (None if lock_token is None else
                 _start_lock_heartbeat("agent", session_id, lock_token, 3600))
        try:
            _run_worker(session_id, run_id, brain, registry, resume_run,
                        approval_granted, require_fill_approval, max_steps,
                        start_step, main_loop, survey_path, original_filename,
                        trace_path, abort_check=None if lease is None else lease.check)
        finally:
            if lease is not None:
                lease.release(session_store.release_task)
            else:
                try:
                    session_store.release_task("agent", session_id, token=lock_token)
                except Exception:
                    pass

    # P2 修复（审查报告2）：stage 更新与登记放在线程启动前——t.start()
    # 之后不得再有可抛异常的语句，锁的释放责任才能干净地移交 worker
    session_store.update_stage(session_id, "agent_running")
    t = threading.Thread(target=worker, daemon=True, name=f"agent-{session_id}")
    t.start()
    _agent_tasks[session_id] = t
    return trace_path


def _run_worker(session_id: str, run_id: str, brain, registry, resume_run,
                approval_granted: bool, require_fill_approval: bool,
                max_steps: int, start_step: int, main_loop,
                survey_path: Path, original_filename: str,
                trace_path: Path, abort_check=None) -> None:
    """agent 运行线程主体（锁的获取/释放由调用方管理）

    abort_check：锁租约协作式中止挂点（易主时抛 LockLostError），
    传入编排器在步骤边界检查。
    """
    s = session_store.get(session_id)
    session_dir: Path = s["session_dir"]

    def on_event(event: dict) -> None:
        _notify_event(session_id, event, loop=main_loop)

    def on_step(step: dict) -> None:
        """每步落库（断点恢复依据 + token 计量）"""
        try:
            session_store.add_step(run_id, step)
        except Exception:
            pass

    try:
        orchestrator = AgentOrchestrator(
            session_id=session_id,
            session_dir=session_dir,
            survey_path=survey_path,
            original_filename=original_filename,
            brain=brain,
            registry=registry,
            max_steps=max_steps,
            start_step=start_step,
            require_fill_approval=require_fill_approval,
            approval_granted=approval_granted,
            on_event=on_event,
            on_step=on_step,
            abort_check=abort_check,
        )
        if resume_run is not None:
            orchestrator.hydrate_from_session()
        final = orchestrator.run()

        _finalize_run(run_id, session_id, orchestrator, final)

        # 终止事件进 SSE 队列
        if main_loop is not None:
            from apps.api.services import get_ai_queue
            q = get_ai_queue(session_id)
            if q is not None:
                main_loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "done" if final["status"] != "error" else "error",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "status": final["status"],
                    "message": final.get("summary", ""),
                })
    except LockLostError as e:
        # 审查报告2 P1-3：锁已被其他实例接管。本实例只标记自己的运行
        # 记录为 interrupted 并写日志；不动会话 stage（会话已归新实例管），
        # 也不发 done/error 事件与新实例抢通道。
        try:
            session_store.update_run(
                run_id, status="interrupted", summary=str(e),
                finished_at=datetime.now(),
            )
            session_store.append_ai_log(
                session_id, datetime.now().strftime("%H:%M:%S"),
                f"⛔ Agent 运行中止: {e}",
            )
        except Exception:
            pass
    except Exception as e:  # 兜底：线程内未捕获异常
        session_store.update_run(
            run_id, status="error",
            summary=f"{type(e).__name__}: {e}", finished_at=datetime.now(),
        )
        session_store.append_ai_log(
            session_id, datetime.now().strftime("%H:%M:%S"),
            f"❌ Agent 运行异常: {type(e).__name__}: {e}",
        )
        session_store.update_stage(session_id, "error")
        if main_loop is not None:
            from apps.api.services import get_ai_queue
            q = get_ai_queue(session_id)
            if q is not None:
                main_loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "error",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": f"{type(e).__name__}: {e}",
                })


def _register_main_loop(loop) -> None:
    if loop is not None:
        from .events import set_main_loop
        set_main_loop(loop)


def json_safe(event: dict) -> str:
    """事件兜底序列化（截断）"""
    try:
        import json
        return json.dumps(event, ensure_ascii=False)[:300]
    except Exception:
        return str(event)[:300]
