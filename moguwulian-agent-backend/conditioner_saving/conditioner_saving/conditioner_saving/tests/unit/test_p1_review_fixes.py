"""P1 审查报告专项修复测试（一）：任务锁租约 / ZIP 结构校验 / 指标单位 / 完整率口径

对应审查报告：
- [P1] 数据库任务锁不是安全租约 → 租约 token、条件更新、按 token 删除、心跳续租
- [P1] 新增覆盖率指标存在单位错误 → interval_covers
- 其他：XLSX 验证只检查 4 字节魔数（伪 ZIP 与 ZIP 炸弹）
- 其他：DataFiller filled_count 混算月度/顶层字段导致完整率失真
"""
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.session_manager import SessionManager

from apps.api.models import TaskLock


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


# ==================== 任务锁安全租约 ====================

def test_claim_returns_unique_lease_token(fresh_store):
    """抢占返回唯一租约 token；重复抢占被拒"""
    t1 = fresh_store.claim_task("agent", "s1", ttl_seconds=60)
    assert t1, "claim 应返回租约 token"
    t2 = fresh_store.claim_task("agent", "s1", ttl_seconds=60)
    assert t2 is None, "持锁期间的第二次抢占必须失败"
    assert t1 != t2


def test_release_requires_matching_token(fresh_store):
    """按 token 删除：错误 token / 缺 token 都不能释放锁"""
    token = fresh_store.claim_task("agent", "s2", ttl_seconds=60)

    fresh_store.release_task("agent", "s2", token="wrong-token")
    assert fresh_store.claim_task("agent", "s2", ttl_seconds=60) is None, \
        "错误 token 的释放不能移除锁"

    fresh_store.release_task("agent", "s2", token=None)
    assert fresh_store.claim_task("agent", "s2", ttl_seconds=60) is None, \
        "缺 token 的释放被拒绝（不能无条件删除）"

    fresh_store.release_task("agent", "s2", token=token)
    assert fresh_store.claim_task("agent", "s2", ttl_seconds=60), \
        "正确 token 的释放必须移除锁"


def test_old_task_cannot_delete_new_owner_lock(fresh_store):
    """审查报告场景：旧任务超时被接管后，旧任务结束不得删除新锁"""
    old_token = fresh_store.claim_task("sim", "s3", ttl_seconds=60)

    # 旧锁陈旧 → 新任务接管
    with_provider = fresh_store
    from apps.api.db import make_session
    with make_session() as db:
        row = db.get(TaskLock, "sim:s3")
        row.claimed_at = datetime.now() - timedelta(seconds=3600)
        db.commit()
    new_token = with_provider.claim_task("sim", "s3", ttl_seconds=60)
    assert new_token and new_token != old_token

    # 旧任务此时结束，用旧 token 释放——新锁必须仍在
    fresh_store.release_task("sim", "s3", token=old_token)
    assert fresh_store.claim_task("sim", "s3", ttl_seconds=60) is None, \
        "旧任务的释放不得删除新持有者的锁"

    fresh_store.release_task("sim", "s3", token=new_token)


def test_heartbeat_renew_keeps_lock(fresh_store):
    """心跳续租：持锁方续租成功；锁易主后旧 token 续租失败"""
    token = fresh_store.claim_task("sim", "s4", ttl_seconds=60)
    assert fresh_store.renew_task("sim", "s4", token, ttl_seconds=60) is True
    assert fresh_store.claim_task("sim", "s4", ttl_seconds=60) is None, "续租后锁仍有效"

    # 续租刷新 claimed_at：即使按墙钟已接近 TTL 也不会被判陈旧
    from apps.api.db import make_session
    with make_session() as db:
        row = db.get(TaskLock, "sim:s4")
        row.claimed_at = datetime.now() - timedelta(seconds=59)
        db.commit()
    assert fresh_store.renew_task("sim", "s4", token, ttl_seconds=60) is True
    assert fresh_store.claim_task("sim", "s4", ttl_seconds=60) is None

    fresh_store.release_task("sim", "s4", token=token)


def test_renew_fails_after_takeover(fresh_store):
    """锁易主后：旧 token 续租返回 False，不刷新新持有者的租期"""
    old_token = fresh_store.claim_task("sim", "s5", ttl_seconds=60)
    from apps.api.db import make_session
    with make_session() as db:
        row = db.get(TaskLock, "sim:s5")
        row.claimed_at = datetime.now() - timedelta(seconds=3600)
        db.commit()
    new_token = fresh_store.claim_task("sim", "s5", ttl_seconds=60)
    assert new_token

    assert fresh_store.renew_task("sim", "s5", old_token) is False
    fresh_store.release_task("sim", "s5", token=new_token)


def test_concurrent_stale_takeover_single_winner(fresh_store):
    """并发抢占陈旧锁：原子条件更新保证只有一个赢家"""
    first = fresh_store.claim_task("sim", "s6", ttl_seconds=60)
    assert first
    from apps.api.db import make_session
    with make_session() as db:
        row = db.get(TaskLock, "sim:s6")
        row.claimed_at = datetime.now() - timedelta(seconds=3600)
        db.commit()

    results = []
    barrier = threading.Barrier(4)

    def racer():
        barrier.wait()
        results.append(fresh_store.claim_task("sim", "s6", ttl_seconds=60))

    threads = [threading.Thread(target=racer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [r for r in results if r]
    assert len(winners) == 1, f"并发抢占陈旧锁应只有一个赢家，实际 {len(winners)}"


# ==================== XLSX 结构校验（伪 ZIP / ZIP 炸弹） ====================

def _make_xlsx(path: Path, members: dict | None = None) -> None:
    members = members or {"[Content_Types].xml": "Types",
                          "xl/workbook.xml": "wb",
                          "xl/worksheets/sheet1.xml": "s"}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_real_xlsx_passes_structure_check(tmp_path):
    """真实 xlsx（openpyxl 生成）必须通过结构校验"""
    from openpyxl import Workbook

    from apps.api.upload_guard import verify_xlsx_structure

    p = tmp_path / "real.xlsx"
    wb = Workbook()
    wb.active["A1"] = "调研表"
    wb.save(p)
    verify_xlsx_structure(p)  # 不抛即通过


def test_fake_zip_rejected(tmp_path):
    """审查报告复现：4 字节 PK 魔数 + 伪 ZIP 数据必须被拒绝"""
    from apps.api.upload_guard import UploadRejected, verify_xlsx_structure

    p = tmp_path / "fake.xlsx"
    p.write_bytes(b"PK\x03\x04" + b"not really a zip" * 10)
    with pytest.raises(UploadRejected):
        verify_xlsx_structure(p)


def test_xlsx_missing_required_members_rejected(tmp_path):
    """缺 [Content_Types].xml / xl/workbook.xml 的伪容器被拒绝"""
    from apps.api.upload_guard import UploadRejected, verify_xlsx_structure

    p1 = tmp_path / "no_content_types.xlsx"
    with zipfile.ZipFile(p1, "w") as zf:
        zf.writestr("xl/workbook.xml", "wb")
    with pytest.raises(UploadRejected):
        verify_xlsx_structure(p1)

    p2 = tmp_path / "no_workbook.xlsx"
    with zipfile.ZipFile(p2, "w") as zf:
        zf.writestr("[Content_Types].xml", "t")
    with pytest.raises(UploadRejected):
        verify_xlsx_structure(p2)


def test_zip_bomb_rejected(tmp_path):
    """ZIP 炸弹：声明解压超上限的成员被拒绝"""
    from apps.api.upload_guard import UploadRejected, verify_xlsx_structure

    p = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("[Content_Types].xml", "t")
        zf.writestr("xl/workbook.xml", "w")
        zf.writestr("xl/big.bin", b"0" * (300 * 1024 * 1024))  # 解压 300MB
    with pytest.raises(UploadRejected):
        verify_xlsx_structure(p)


def test_zip_member_count_cap(tmp_path):
    """成员数量超上限被拒绝"""
    from apps.api.upload_guard import UploadRejected, verify_xlsx_structure

    p = tmp_path / "many.xlsx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("[Content_Types].xml", "t")
        zf.writestr("xl/workbook.xml", "w")
        for i in range(1500):
            zf.writestr(f"xl/media/img{i}.bin", "x")
    with pytest.raises(UploadRejected):
        verify_xlsx_structure(p)


def test_zip_corrupt_member_rejected(tmp_path):
    """成员数据损坏（解压/CRC 失败）被拒绝"""
    from apps.api.upload_guard import UploadRejected, verify_xlsx_structure

    p = tmp_path / "bad.xlsx"
    _make_xlsx(p, {"[Content_Types].xml": "types-content-here",
                   "xl/workbook.xml": "workbook-content-here",
                   "xl/worksheets/sheet1.xml": "sheet-data" * 200})
    # 精确定位最后一个成员的压缩数据起点（本地头 30 字节 + 文件名）并破坏，
    # 避免翻到文件中无引用区域导致的不确定性
    with zipfile.ZipFile(p) as zf:
        info = zf.getinfo("xl/worksheets/sheet1.xml")
        data_offset = info.header_offset + 30 + len("xl/worksheets/sheet1.xml")
    raw = bytearray(p.read_bytes())
    raw[data_offset + 5] ^= 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(UploadRejected):
        verify_xlsx_structure(p)


# ==================== 覆盖率指标单位 ====================

def test_interval_covers_unit_consistency():
    """审查报告最小复现：expected=11.9（百分数）落在 0.086~0.131（小数）内"""
    from evaluation.metrics import interval_covers

    assert interval_covers(11.9, 0.086, 0.131) is True
    assert interval_covers(11.9, 0.131, 0.086) is True   # 端点顺序无关
    assert interval_covers(5.0, 0.086, 0.131) is False   # 区间外
    assert interval_covers(11.9, None, 0.131) is None    # 端点缺失


def test_aggregate_coverage_metric_sane():
    """聚合口径：修复后覆盖率不再是 0%"""
    from evaluation.metrics import aggregate

    results = [{
        "key": "patchy", "expected_outcome": "completed", "actual_outcome": "completed",
        "outcome_correct": True, "known_gap": None,
        "expected_rate": 11.9, "rate_min": 0.086, "rate_max": 0.131,
        "rate_range": "8.6%~13.1%", "rate_deviation_pp": 0.0,
        "rate_point_deviation_pp": 0.85, "rate_interval_width_pp": 4.5,
        "rate_interval_covers_expected": True, "rate_pass": True,
        "expected_region": "华南", "actual_region": "华南", "region_correct": True,
        "note": "",
    }]
    agg = aggregate(results)
    assert agg["rate_interval_coverage"] == 1.0


# ==================== 确定性脚本超时 ====================

def test_deterministic_script_timeout_enforced(tmp_path):
    """审查报告复现：脚本挂死时超时必须生效（原先阻塞读 stdout 直到 EOF，
    wait(timeout) 永远不执行——线程与任务锁被永久占用）"""
    import shutil
    import sys
    import time

    import core.skill_caller as sc
    from core.skill_caller import SkillCaller, SkillError

    scripts = tmp_path / ".claude" / "skills" / "hang-skill" / "scripts"
    scripts.mkdir(parents=True)
    (tmp_path / ".claude" / "skills" / "hang-skill" / "SKILL.md").write_text("t")
    (scripts / "hang.py").write_text(
        "import time\nprint('started', flush=True)\ntime.sleep(30)\n")

    caller = SkillCaller(timeout=2, project_root=tmp_path)

    fake = {"extract-and-check-data": ("hang-skill", "hang.py")}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sc, "_DETERMINISTIC_SKILLS", fake)
        t0 = time.time()
        with pytest.raises(SkillError, match="超时"):
            caller._run_deterministic_script("extract-and-check-data",
                                             "in.xlsx", "out.json",
                                             check_output=False)
        elapsed = time.time() - t0
    shutil.rmtree(tmp_path / ".claude")
    assert elapsed < 10, f"超时未及时生效（耗时 {elapsed:.1f}s）"


# ==================== 仿真沙箱参数映射 ====================

def test_sandbox_staging_maps_inputs_and_outputs(tmp_path):
    """plant-simulation 沙箱映射：输入拷入 io/，输出（目录/文件）映射到 io/"""
    from core.skill_caller import SkillCaller

    (tmp_path / ".claude" / "skills").mkdir(parents=True)  # SkillCaller 前置要求
    caller = SkillCaller(timeout=10, project_root=tmp_path)
    sandbox = tmp_path / "sandbox"

    # 仿真任务：args[0]=输入 JSON，args[1]=输出目录（无后缀），args[2]=任务类型
    input_json = tmp_path / "sim_input_all.json"
    input_json.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "sim_output"

    mapped, backplan = caller._stage_sandbox(
        "plant-simulation",
        (str(input_json), str(out_dir), "new_station"),
        sandbox,
    )
    # 输入被拷入沙箱 io/ 并重写指向
    assert Path(mapped[0]).parent.name == "io"
    assert Path(mapped[0]).read_text(encoding="utf-8") == "{}"
    # 输出目录映射为沙箱内同名目录并预建
    assert Path(mapped[1]).parent.name == "io"
    assert Path(mapped[1]).is_dir()
    assert backplan and Path(backplan[0][1]) == out_dir

    # 回拷：沙箱产物出现到原输出目录
    (Path(mapped[1]) / "result.csv").write_text("8760", encoding="utf-8")
    caller._collect_sandbox_outputs(backplan)
    assert (out_dir / "result.csv").read_text(encoding="utf-8") == "8760"

    # generate_operating：实际调用传输入文件路径（services 预写
    # sim_operating_input.txt），同样拷入沙箱
    op_input = tmp_path / "sim_operating_input.txt"
    op_input.write_text("运行规律描述", encoding="utf-8")
    inline, backplan2 = caller._stage_sandbox(
        "plant-simulation",
        (str(op_input), str(tmp_path / "operating.json"), "generate_operating"),
        tmp_path / "sandbox2",
    )
    assert Path(inline[0]).parent.name == "io"
    assert Path(inline[0]).read_text(encoding="utf-8") == "运行规律描述"
    assert Path(inline[1]).parent.name == "io"

    # generate_operating 兼容内联数据：args[0] 不是存在的文件时保持原样
    legacy, _ = caller._stage_sandbox(
        "plant-simulation",
        ('{"monthly_table": []}', str(tmp_path / "operating2.json"),
         "generate_operating"),
        tmp_path / "sandbox3",
    )
    assert legacy[0] == '{"monthly_table": []}'


# ==================== DataFiller 完整率口径 ====================

def test_data_filler_top_level_fill_does_not_distort_completeness(tmp_path):
    """顶层 electricity_price 填充不计入月度完整率分母（原始完整率不再偏低/为负）"""
    import json

    from core.data_filler import DataFiller

    extracted = {
        "available_fields": ["total_energy"],
        "monthly_data": [
            {"month": 1, "total_energy": None},
            {"month": 2, "total_energy": 100.0},
        ],
    }
    suggestions = {
        "fillable_suggestions": [
            {"field": "monthly_data[0].total_energy", "suggested_value": 80.0},
            {"field": "electricity_price", "suggested_value": 0.72},
        ],
        "unfillable_fields": [],
        "defaults_version": "v-test",
    }
    src = tmp_path / "extracted.json"
    sug = tmp_path / "suggestions.json"
    out = tmp_path / "filled.json"
    src.write_text(json.dumps(extracted, ensure_ascii=False), encoding="utf-8")
    sug.write_text(json.dumps(suggestions, ensure_ascii=False), encoding="utf-8")

    DataFiller().fill_data(str(src), str(sug), str(out))
    meta = json.loads(out.read_text(encoding="utf-8"))["_metadata"]

    # 分母 = 2 个月 × 1 个可用字段 = 2；填充 1 个月度字段
    assert meta["monthly_fill_count"] == 1
    assert meta["fill_count"] == 2            # 总填充数含顶层字段
    assert meta["original_completeness"] == 0.5
    assert meta["final_completeness"] == 1.0  # 只剩 1 个可算字段且被填充
