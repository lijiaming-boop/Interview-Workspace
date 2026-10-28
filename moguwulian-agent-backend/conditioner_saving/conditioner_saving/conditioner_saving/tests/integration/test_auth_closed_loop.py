"""P0/P1 审查报告专项修复测试（三）：认证闭环 + 相似案例跨用户隔离

对应审查报告：
- [P0] 生产认证链路既未闭环，又可被绕过：
  * require_auth=True 且 AUTH_SECRET 为空 → 服务端 fail-closed（不再信任
    客户端可伪造的明文 X-User-Info，攻击者无法自封 admin）
  * 未签名 / 错签名身份头 → 401
  * 网关签名身份头（X-User-Info + X-User-Info-Sig）→ 放行
- [P1] 相似案例接口仍泄露其他用户数据：普通用户只能检索到自己的历史案例
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

from apps.api import config
from apps.api.deps import session_store
from apps.api.main import app

AUTH_TEST_SECRET = "closed-loop-test-secret"


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """本文件所有用例都在强制认证 + 签名密钥下运行"""
    monkeypatch.setattr(config.settings, "require_auth", True)
    monkeypatch.setattr(config.settings, "auth_secret", AUTH_TEST_SECRET)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _signed_header(sub, role=None, secret=AUTH_TEST_SECRET):
    from apps.api.deps import identity_signature
    payload = {"sub": sub, "name": sub}
    if role:
        payload["role"] = role
    raw = json.dumps(payload)
    return {"X-User-Info": raw,
            "X-User-Info-Sig": identity_signature(raw, secret)}


def _upload(client, headers):
    from openpyxl import Workbook

    buf = io.BytesIO()
    Workbook().save(buf)
    buf.seek(0)
    return client.post(
        "/api/upload?module_type=air_conditioner",
        files={"file": ("表.xlsx", buf, "application/octet-stream")},
        headers=headers,
    )


# ==================== 认证闭环 ====================

def test_missing_header_rejected(client):
    """无身份头 → 401"""
    resp = _upload(client, {})
    assert resp.status_code == 401


def test_unsigned_header_rejected(client):
    """审查报告复现：伪造的明文 X-User-Info（自封 admin）必须 401"""
    forged = {"X-User-Info": json.dumps({"sub": "attacker", "role": "admin"})}
    resp = _upload(client, forged)
    assert resp.status_code == 401
    assert "签名" in resp.json()["detail"]


def test_wrong_signature_rejected(client):
    """密钥不匹配的签名 → 401"""
    resp = _upload(client, _signed_header("attacker", role="admin",
                                          secret="attacker-guessed-secret"))
    assert resp.status_code == 401


def test_signed_header_accepted_and_role_enforced(client):
    """网关签名身份头放行；role 来自签名载荷而非客户端额外头"""
    ok = _upload(client, _signed_header("sales-a"))
    assert ok.status_code == 200
    sid = ok.json()["session_id"]

    # 归属人可访问
    assert client.get(f"/api/session/{sid}",
                      headers=_signed_header("sales-a")).status_code == 200
    # 非特权角色越权 → 403
    assert client.get(f"/api/session/{sid}",
                      headers=_signed_header("sales-b")).status_code == 403
    # 签名载荷中 role=admin → 特权放行
    assert client.get(f"/api/session/{sid}",
                      headers=_signed_header("boss", role="admin")).status_code == 200


def test_fail_closed_when_secret_missing(monkeypatch):
    """require_auth=True 而 AUTH_SECRET 为空 → 500 拒绝处理（不信任明文身份头）

    直接调用依赖函数验证（启动守卫会让 TestClient 无法进入请求阶段）。
    """
    from fastapi import HTTPException

    from apps.api.deps import get_current_user

    monkeypatch.setattr(config.settings, "require_auth", True)
    monkeypatch.setattr(config.settings, "auth_secret", "")
    forged = json.dumps({"sub": "attacker", "role": "admin"})
    with pytest.raises(HTTPException) as exc:
        get_current_user(x_user_info=forged, x_user_info_sig=None)
    assert exc.value.status_code == 500
    assert "AUTH_SECRET" in exc.value.detail


def test_startup_fails_fast_without_secret(monkeypatch):
    """启动守卫：REQUIRE_AUTH=true 且未配置 AUTH_SECRET → 拒绝启动"""
    import asyncio

    import apps.api.deps as deps_mod

    monkeypatch.setattr(config.settings, "require_auth", True)
    monkeypatch.setattr(config.settings, "auth_secret", "")

    class _FakeApp:
        pass

    with pytest.raises(RuntimeError, match="AUTH_SECRET"):
        asyncio.run(deps_mod.startup_banner(_FakeApp()).__aenter__())


# ==================== 相似案例跨用户隔离 ====================

def _seed_result_session(owner: str, region: str) -> str:
    sid = session_store.register(
        __import__("core.session_manager", fromlist=["SessionManager"])
        .SessionManager().create_session_directory(),
        "survey.xlsx", owner, "air_conditioner")
    live = session_store.get(sid)
    live["stage"] = "result"
    live["extracted_data"] = {"project_region": region, "station_type": "standard"}
    live["result"] = {"energy_saving_rate_min": 0.10, "energy_saving_rate_max": 0.14,
                      "energy_saving_rate_range": "10%~14%"}
    live.save()
    return sid


def test_similar_cases_isolated_for_normal_user(client, monkeypatch):
    """普通用户检索相似案例：只返回自己名下的历史案例"""
    mine = _seed_result_session("user-me", "华南")
    _seed_result_session("user-other", "华南")
    _seed_result_session("user-other2", "华南")

    resp = client.get(f"/api/history/{mine}/similar",
                      headers=_signed_header("user-me"))
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert all(r["user_id"] == "user-me" for r in items), \
        "相似案例不得泄露其他用户的案例"


def test_similar_cases_privileged_sees_all(client):
    """特权角色（admin）可跨全库检索相似案例（源会话自身按设计排除）"""
    mine = _seed_result_session("user-me2", "华北")
    other = _seed_result_session("user-other3", "华北")

    resp = client.get(f"/api/history/{mine}/similar",
                      headers=_signed_header("boss", role="admin"))
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert other in {r["session_id"] for r in items}, \
        "特权角色应能检索到跨用户的相似案例"


def test_history_endpoint_still_isolated(client):
    """/history 基线回归：普通用户只见自己的记录"""
    _seed_result_session("user-hist-me", "华东")
    _seed_result_session("user-hist-other", "华东")

    resp = client.get("/api/history", headers=_signed_header("user-hist-me"))
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items and all(r["user_id"] == "user-hist-me" for r in items)
