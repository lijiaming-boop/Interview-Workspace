"""check_env.py 的回归测试。

分两层:
- 纯函数层: 路径切分/比较、版本比较、RT Core 判定、平台常量探测、unnorm_key 校验。
- 集成层: 用合成 env 与合成目录跑完整 checker, 断言 FAIL / WARN 落在预期位置。

全部不依赖 Isaac Sim / torch / numpy 之外的东西, 可以在本机直接跑:
    python -m pytest 12-IsaacLab-Bridge-Franka/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_env  # noqa: E402
from check_env import (  # noqa: E402
    EnvChecker,
    check_unnorm_key,
    detect_robot_platform,
    driver_status,
    expected_constants,
    find_openvla_roots,
    has_rt_cores,
    norm_path,
    parse_version,
    path_in_list,
    pick_preferred_openvla_root,
    render_text,
    split_path_list,
    to_json,
    version_ge,
)


# --------------------------------------------------------------------------- #
# 路径工具
# --------------------------------------------------------------------------- #


def test_split_path_list_does_not_break_windows_drives():
    # 必须用 os.pathsep(';') 切, 不能拿 ':' 切, 否则 C:\ 会被切碎
    parts = split_path_list(r"C:\a\b;D:\c", sep=";")
    assert parts == [r"C:\a\b", r"D:\c"]


def test_split_path_list_posix_and_empty():
    assert split_path_list("/a:/b", sep=":") == ["/a", "/b"]
    assert split_path_list("", sep=":") == []
    assert split_path_list(None, sep=":") == []
    assert split_path_list("/a::/b", sep=":") == ["/a", "/b"]


def test_path_in_list_exact_and_ancestor(tmp_path):
    target = tmp_path / "proj" / "bridge"
    target.mkdir(parents=True)
    entries = [str(tmp_path / "proj")]

    covered, how = path_in_list(entries, target)
    assert covered is True
    assert how == "ancestor"

    covered, how = path_in_list([str(target)], target)
    assert (covered, how) == (True, "exact")

    covered, how = path_in_list([str(tmp_path / "other")], target)
    assert (covered, how) == (False, "missing")


def test_path_in_list_skips_ancestor_when_disabled(tmp_path):
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    covered, _ = path_in_list([str(tmp_path / "a")], child, allow_ancestor=False)
    assert covered is False


def test_path_in_list_survives_mixed_relative_paths():
    # commonpath 在混合绝对/相对路径上会抛 ValueError, 不能让它冒出来
    covered, how = path_in_list(["relative/dir"], "/abs/dir", allow_ancestor=True)
    assert covered is False
    assert how == "missing"


def test_norm_path_expands_and_strips_quotes(monkeypatch):
    monkeypatch.setenv("CHECK_ENV_TEST_DIR", "/tmp/x")
    got = norm_path('"$CHECK_ENV_TEST_DIR/y/../z"')
    assert got == norm_path("/tmp/x/z")


# --------------------------------------------------------------------------- #
# 版本
# --------------------------------------------------------------------------- #


def test_parse_version_handles_short_and_long():
    assert parse_version("580.65.06") == (580, 65, 6)
    assert parse_version("581.86") == (581, 86)
    assert parse_version("garbage") == (0,)


def test_version_ge_pads_missing_segments():
    assert version_ge((580, 65, 6), (580, 65, 6)) is True
    assert version_ge((581, 86), (580, 65, 6)) is True
    assert version_ge((580, 64), (580, 65, 6)) is False
    assert version_ge((535, 0), (580, 65, 6)) is False


@pytest.mark.parametrize(
    "driver,expected",
    [("580.65.06", "OK"), ("581.86", "OK"), ("535.104.05", "WARN"), ("n/a", "SKIP")],
)
def test_driver_status(driver, expected):
    status, _ = driver_status(driver)
    assert status == expected


# --------------------------------------------------------------------------- #
# RT Core —— Isaac Sim 的硬门槛
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA A10", True),
        ("NVIDIA A10G", True),
        ("NVIDIA RTX A6000", True),
        ("NVIDIA GeForce RTX 4050 Laptop GPU", True),
        ("NVIDIA L40S", True),
        ("Tesla T4", True),
        # 无 RT Core: 官方明确不支持
        ("NVIDIA A100-SXM4-40GB", False),
        ("NVIDIA A800 80GB PCIe", False),
        ("NVIDIA H100 PCIe", False),
        ("NVIDIA H200", False),
        ("Tesla V100-SXM2-32GB", False),
        ("NVIDIA B200", False),
        # 认不出
        ("Some Future GPU", None),
    ],
)
def test_has_rt_cores(name, expected):
    assert has_rt_cores(name) is expected


def test_a100_is_not_matched_by_a10_marker():
    # 'a100' 同时含 'a10', 顺序错了会把不支持的卡判成支持
    assert has_rt_cores("NVIDIA A100-SXM4-40GB") is False


# --------------------------------------------------------------------------- #
# 平台常量探测
# --------------------------------------------------------------------------- #


def test_detect_original_platform_priority():
    # 与上游 prismatic/vla/constants.py 的 if/elif 顺序一致
    assert detect_robot_platform(["cmd", "--libero", "--bridge"]) == "LIBERO"
    assert detect_robot_platform(["cmd", "--aloha", "--bridge"]) == "ALOHA"
    assert detect_robot_platform(["cmd", "--bridge"]) == "BRIDGE"
    # 一个都匹配不到 -> 兜底 LIBERO, 这是最危险的一条
    assert detect_robot_platform(["cmd", "--policy", "openvla"]) == "LIBERO"


def test_platform_detected_from_real_commands():
    # README 里的两条真实命令都应落到 BRIDGE
    libero_style = (
        "./isaaclab.sh -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py "
        "--policy openvla --checkpoint /ckpt --unnorm_key bridge_orig --domain_mode raw"
    )
    assert detect_robot_platform(libero_style.split()) == "BRIDGE"
    aligned_style = "./isaaclab.sh -p run_closed_loop.py --domain_mode bridge_aligned --policy heuristic"
    assert detect_robot_platform(aligned_style.split()) == "BRIDGE"


def test_expected_constants_match_upstream_bridge_block():
    assert expected_constants("BRIDGE") == {
        "NUM_ACTIONS_CHUNK": 5,
        "ACTION_DIM": 7,
        "PROPRIO_DIM": 7,
    }
    # 猜错时的代价: proprio 8 维 vs 本项目只有 7 维
    assert expected_constants("LIBERO")["PROPRIO_DIM"] == 8


# --------------------------------------------------------------------------- #
# dataset_statistics.json
# --------------------------------------------------------------------------- #


def test_check_unnorm_key_ok():
    stats = {"bridge_orig": {"action": {"q01": []}, "proprio": {"q01": []}}}
    status, _ = check_unnorm_key(stats, "bridge_orig")
    assert status == "OK"


def test_check_unnorm_key_reports_available_keys():
    stats = {"libero_spatial": {"action": {}, "proprio": {}}}
    status, detail = check_unnorm_key(stats, "bridge_orig")
    assert status == "FAIL"
    assert "libero_spatial" in detail


def test_check_unnorm_key_missing_subitem():
    stats = {"bridge_orig": {"action": {}}}
    status, detail = check_unnorm_key(stats, "bridge_orig")
    assert status == "FAIL"
    assert "proprio" in detail


@pytest.mark.parametrize("stats", [{}, {"bridge_orig": "not-a-dict"}])
def test_check_unnorm_key_rejects_bad_shapes(stats):
    status, _ = check_unnorm_key(stats, "bridge_orig")
    assert status == "FAIL"


# --------------------------------------------------------------------------- #
# 自动定位 openvla-oft checkout
# --------------------------------------------------------------------------- #


def _make_fake_openvla_root(base: Path, name: str) -> Path:
    root = base / name
    (root / "experiments" / "robot").mkdir(parents=True, exist_ok=True)
    (root / "experiments" / "robot" / "openvla_utils.py").write_text("", encoding="utf-8")
    return root


def test_find_openvla_roots_depth_limited(tmp_path):
    good = _make_fake_openvla_root(tmp_path / "ws", "patch-check3")
    # 干扰项: 根目录本身深度 4, 超过 max_depth=3 的扫描范围
    deep = _make_fake_openvla_root(tmp_path / "ws" / "a" / "b" / "c", "d")

    hits = find_openvla_roots(tmp_path / "ws", max_depth=3)
    assert good in hits
    assert deep not in hits
    # 真实工作区里 checkout 在 <ws>/02-实验代码/patch-check3（深度 2）, 必须落在范围内
    assert len(hits) == 1


def test_pick_preferred_openvla_root_prefers_patch(tmp_path):
    ws = tmp_path / "ws"
    baseline = _make_fake_openvla_root(ws, "openvla-oft-baseline")
    patched = _make_fake_openvla_root(ws, "patch-check3")
    assert pick_preferred_openvla_root([baseline, patched]) == patched
    assert pick_preferred_openvla_root([baseline]) == baseline
    assert pick_preferred_openvla_root([]) is None


def test_pick_preferred_prefers_highest_patch_check_number(tmp_path):
    # 本工作区真的有 patch-check / patch-check2 / patch-check3 三份,
    # 取"第一个带 patch 的"会选中最老的那一份
    ws = tmp_path / "ws"
    v1 = _make_fake_openvla_root(ws, "patch-check")
    v2 = _make_fake_openvla_root(ws, "patch-check2")
    v3 = _make_fake_openvla_root(ws, "patch-check3")
    assert pick_preferred_openvla_root([v1, v2, v3]) == v3
    assert pick_preferred_openvla_root([v3, v1, v2]) == v3
    assert pick_preferred_openvla_root([v1, v2]) == v2


def test_probe_glibc_is_linux_only(monkeypatch):
    # Git Bash 下的 ldd 是 cygwin 版, 照抄会得出"GLIBC 通过"的假结论
    monkeypatch.setattr(check_env.sys, "platform", "win32")
    assert check_env.probe_glibc() is None


def test_multiple_candidates_are_surfaced(tmp_path):
    ws = tmp_path / "ws"
    _make_fake_openvla_root(ws, "patch-check2")
    chosen = _make_fake_openvla_root(ws, "patch-check3")
    checker = EnvChecker(
        workspace_root=ws,
        env={},
        policy="openvla",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    report = checker.run()
    entry = next(c for c in report.checks if c.name == "openvla-oft 候选目录")
    assert entry.status == "WARN"
    assert chosen.name in entry.detail
    assert "patch-check2" in entry.detail
    # 未显式指定 + 多候选 -> 建议里必须给出 patch-check3
    assert "patch-check3" in "\n".join(report.hints)


# --------------------------------------------------------------------------- #
# 集成: 不依赖 Isaac 的完整跑通
# --------------------------------------------------------------------------- #


def _synthetic_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    _make_fake_openvla_root(ws, "patch-check3")
    return ws


def test_heuristic_mode_passes_without_openvla_root(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={},
        policy="heuristic",
        simulate_argv="./isaaclab.sh -p run_closed_loop.py --policy heuristic --domain_mode bridge_aligned",
        ignore_hardware=True,
    )
    report = checker.run()
    env_failures = [c for c in report.failures if c.group == EnvChecker.ENV_GROUP]
    assert env_failures == []
    # heuristic 不需要 checkpoint / openvla_root, 但应明确告知
    names = {c.name for c in report.warnings}
    assert "CHECKPOINT" in names


def test_openvla_mode_fails_without_openvla_root_and_checkpoint(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={},
        policy="openvla",
        simulate_argv="./isaaclab.sh -p run_closed_loop.py --policy openvla --unnorm_key bridge_orig",
        ignore_hardware=True,
    )
    report = checker.run()
    failed = {c.name for c in report.failures}
    # run_domain_gap_eval.* 用 ${OPENVLA_OFT_ROOT:?} 强制要求
    assert "OPENVLA_OFT_ROOT" in failed
    assert "CHECKPOINT" in failed

    entry = next(c for c in report.failures if c.name == "OPENVLA_OFT_ROOT")
    # 没设置但工作区里能找到 -> 提示里直接给出可用路径
    assert "patch-check3" in entry.detail or "patch-check3" in entry.hint


def test_openvla_mode_passes_with_full_env(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    ovla = ws / "patch-check3"
    (ovla / "prismatic" / "vla").mkdir(parents=True)
    (ovla / "prismatic" / "vla" / "constants.py").write_text("ACTION_DIM = 7\n", encoding="utf-8")

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "dataset_statistics.json").write_text(
        json.dumps({"bridge_orig": {"action": {}, "proprio": {}}}), encoding="utf-8"
    )
    bridge_dir = ws / "12-IsaacLab-Bridge-Franka"
    bridge_dir.mkdir()
    for name in check_env.BRIDGE_FILES:
        (bridge_dir / name).write_text("", encoding="utf-8")

    env = {
        "OFT_ROOT": str(ws),
        "OPENVLA_OFT_ROOT": str(ovla),
        "PYTHONPATH": f"{bridge_dir}{check_env.os.pathsep}{ovla}",
        "ISAACLAB_LAUNCHER": str(bridge_dir / "run_domain_gap_eval.sh"),
        "SEED": "7",
        "UNNORM_KEY": "bridge_orig",
    }
    checker = EnvChecker(
        bridge_dir=bridge_dir,
        workspace_root=ws,
        env=env,
        policy="openvla",
        checkpoint=str(ckpt),
        simulate_argv="./isaaclab.sh -p run_closed_loop.py --policy openvla --unnorm_key bridge_orig",
        ignore_hardware=True,
    )
    report = checker.run()
    assert report.failures == [], [f"{c.name}: {c.detail}" for c in report.failures]
    names = {c.name for c in report.checks if c.status == "OK"}
    assert {"OFT_ROOT", "OPENVLA_OFT_ROOT", "PYTHONPATH", "ISAACLAB_LAUNCHER", "dataset_statistics.json"} <= names


def test_wrong_unnorm_key_is_caught(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "dataset_statistics.json").write_text(
        json.dumps({"libero_spatial_no_noops": {"action": {}, "proprio": {}}}), encoding="utf-8"
    )
    checker = EnvChecker(
        workspace_root=ws,
        env={"OFT_ROOT": str(ws)},
        policy="openvla",
        checkpoint=str(ckpt),
        unnorm_key="bridge_orig",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    report = checker.run()
    entry = next(c for c in report.checks if c.name == "dataset_statistics.json")
    assert entry.status == "FAIL"
    assert "libero_spatial_no_noops" in entry.detail


def test_platform_fallback_is_a_failure(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={},
        policy="heuristic",
        # 命令里一个 bridge 字样都没有 -> 上游会兜底 LIBERO
        simulate_argv="./isaaclab.sh -p run_closed_loop.py --policy heuristic --unnorm_key libero_spatial",
        ignore_hardware=True,
    )
    report = checker.run()
    entry = next(c for c in report.checks if c.name == "平台常量探测")
    assert entry.status == "FAIL"
    assert "LIBERO" in entry.detail


def test_seed_must_be_integer(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={"SEED": "seven"},
        policy="heuristic",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    report = checker.run()
    entry = next(c for c in report.checks if c.name == "SEED")
    assert entry.status == "FAIL"


def test_eula_env_var_is_flagged_not_encouraged(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={"OMNI_KIT_ACCEPT_EULA": "yes"},
        policy="heuristic",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    report = checker.run()
    entry = next(c for c in report.checks if "EULA" in c.name)
    assert entry.status == "WARN"


def test_bad_oft_root_is_a_failure(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={"OFT_ROOT": str(tmp_path / "does-not-exist")},
        policy="heuristic",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    report = checker.run()
    entry = next(c for c in report.checks if c.name == "OFT_ROOT")
    assert entry.status == "FAIL"


# --------------------------------------------------------------------------- #
# 渲染与序列化
# --------------------------------------------------------------------------- #


def _sample_report(tmp_path: Path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={},
        policy="openvla",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    return checker.run()


def test_render_text_contains_sections_and_fix_hints(tmp_path):
    report = _sample_report(tmp_path)
    text = render_text(report)
    for marker in ("环境变量", "路径与代码资产", "必须先修复", "可直接粘贴的配置命令"):
        assert marker in text
    assert "export " in text or "$env:" in text


def test_to_json_is_parseable_and_reports_passed_flag(tmp_path):
    report = _sample_report(tmp_path)
    payload = json.loads(to_json(report))
    assert payload["passed"] is False
    assert payload["counts"]["FAIL"] >= 1
    assert any(c["name"] == "OPENVLA_OFT_ROOT" for c in payload["checks"])
    assert all({"group", "name", "status"} <= set(c) for c in payload["checks"])


def test_ignore_hardware_skips_group(tmp_path):
    ws = _synthetic_workspace(tmp_path)
    checker = EnvChecker(
        workspace_root=ws,
        env={},
        policy="heuristic",
        simulate_argv="--bridge",
        ignore_hardware=True,
    )
    report = checker.run()
    hw = [c for c in report.checks if c.group == EnvChecker.HW_GROUP]
    assert len(hw) == 1
    assert hw[0].status == "SKIP"


def test_parse_args_defaults():
    args = check_env.parse_args([])
    assert args.policy == "openvla"
    assert args.ignore_hardware is False
    assert args.strict_hardware is False
    assert args.require_isaac is False


class _StubChecker:
    """替身: 让 main() 的退出码逻辑可以离线测试, 不必扫真实工作区。"""

    def __init__(self, report):
        self._report = report

    def run(self):
        return self._report


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    ws = _synthetic_workspace(tmp_path)

    def fake_factory(**kwargs):
        return _StubChecker(_sample_report(tmp_path))

    monkeypatch.setattr(check_env, "EnvChecker", fake_factory)
    # 有 FAIL -> 1
    assert check_env.main(["--ignore-hardware"]) == 1
    capsys.readouterr()

    ok_report = _sample_report(tmp_path)
    for check in ok_report.failures:
        check.status = "WARN"
    monkeypatch.setattr(check_env, "EnvChecker", lambda **kwargs: _StubChecker(ok_report))
    # 只有 WARN -> 0
    assert check_env.main(["--ignore-hardware"]) == 0
    # --strict 下 WARN 也使退出码变成 2
    assert check_env.main(["--ignore-hardware", "--strict"]) == 2

    assert ws.is_dir()
