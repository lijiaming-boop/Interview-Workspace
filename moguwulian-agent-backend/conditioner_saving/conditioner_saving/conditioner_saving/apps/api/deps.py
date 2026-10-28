"""共享依赖：会话存储、用户认证、配置"""
import hashlib
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException, status


from .config import settings
from .storage import PersistentSessionStore


# ==================== 会话存储 ====================

# 单例（进程内共享；M1 起为落库持久化实现，SQLite/Postgres 由配置决定）
session_store = PersistentSessionStore()


# ==================== 用户认证与访问控制 ====================

def _parse_user(x_user_info: Optional[str]) -> dict:
    import json
    try:
        return json.loads(x_user_info)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid X-User-Info: {e}",
        )


def identity_signature(payload: str, secret: str) -> str:
    """计算身份头 HMAC-SHA256 签名（与 proxy/main.go 保持一致）"""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"),
                    hashlib.sha256).hexdigest()


def get_current_user(
    x_user_info: Optional[str] = Header(default=None, alias=None),
    x_user_info_sig: Optional[str] = Header(default=None, alias=None),
) -> dict:
    """从网关注入的身份头解析用户信息

    - require_auth=False（开发/测试）：缺失时回退 anonymous，零配置可跑
    - require_auth=True（生产）：缺失即 401，所有请求必须经 OIDC 网关；
      且必须配置 auth_secret 并携带匹配的 X-User-Info-Sig（HMAC-SHA256），
      未签名/签名不符一律 401——杜绝客户端直接伪造身份头。
    P1 修复（审查报告：认证链路可被绕过）：require_auth=True 而
    auth_secret 为空时不再信任未签名身份头（此前攻击者可自行提交
    X-User-Info 伪造 admin 角色），改为 fail-closed 拒绝服务并提示配置。
    """
    header = x_user_info
    if header is None:
        # 运行时读取 settings，便于测试切换；Header alias 无法动态，
        # 由 routers 通过 Request.headers 统一兜底（见 require_user）
        if settings.require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"未认证：缺少 {settings.auth_header} 身份头，请通过网关访问",
            )
        return {"sub": "anonymous", "name": "anonymous", "email": ""}

    if settings.require_auth:
        # P1 fail-closed：强制认证但未配置签名密钥属于服务端配置错误，
        # 绝不能退化为信任客户端可伪造的明文身份头
        if not settings.auth_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="认证配置错误：REQUIRE_AUTH=true 但未设置 AUTH_SECRET，"
                       "拒绝处理带身份头的请求（请部署网关并共享签名密钥）",
            )
        if not x_user_info_sig or not hmac.compare_digest(
                identity_signature(header, settings.auth_secret), x_user_info_sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="身份签名缺失或不匹配：拒绝信任未签名的身份头",
            )
    return _parse_user(header)


def is_privileged(user: dict) -> bool:
    roles = set()
    for key in ("role", "roles"):
        v = user.get(key)
        if isinstance(v, str):
            roles.add(v)
        elif isinstance(v, list):
            roles.update(str(x) for x in v)
    return bool(roles & settings.privileged_role_set)


def user_key(user: dict) -> Optional[str]:
    """身份唯一标识归一化（审查报告2 P1：身份可能只有 id 没有 sub）

    上传等写入口用 sub or id 落库，读取/过滤端必须用同一口径取值，
    否则仅含 id 的身份在存储层过滤条件为 None，列表过滤失效、越权
    校验放行。
    """
    return user.get("sub") or user.get("id")


def ensure_session_access(s: dict, user: dict, action: str = "access") -> None:
    """会话归属校验（ACL）

    - require_auth=False（开发）：放行，保持零配置
    - 特权角色（admin/sr）：放行
    - 其余：仅会话归属人可访问，越权 403 并写审计
    """
    if not settings.require_auth:
        return
    if is_privileged(user):
        return
    owner = s.get("user_id")
    uid = user_key(user)
    if owner and owner == uid:
        return
    session_store.audit(
        user_id=uid, action=f"denied.{action}",
        session_id=s.get("session_id"),
        detail={"owner": owner},
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="无权访问该会话资源",
    )


def audit(user: dict, action: str, session_id: Optional[str] = None,
          detail: Optional[dict] = None) -> None:
    """业务审计（上传/下载/运行等敏感动作）"""
    session_store.audit(user_id=user_key(user), action=action,
                        session_id=session_id, detail=detail)

# ==================== 项目根路径 ====================

# apps/api/main.py -> 上溯两级到项目根
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# ==================== 启动生命周期 ====================


@asynccontextmanager
async def startup_banner(app):
    """启动时打印关键环境信息，便于排查；并做启动恢复"""
    from .db import db_ping, init_db, mark_stale_runs_interrupted, recover_stuck_sessions

    # P1 fail-fast（审查报告：生产认证链路未闭环）：强制认证但未配置
    # 签名密钥的部署直接拒绝启动，而不是带病运行等待身份被伪造
    if settings.require_auth and not settings.auth_secret:
        raise RuntimeError(
            "启动失败：REQUIRE_AUTH=true 但未设置 AUTH_SECRET——身份头无法签名校验，"
            "拒绝启动。请设置 AUTH_SECRET（与网关 COOKIE_SECRET 一致）后重试。"
        )

    init_db()  # 幂等：确保 schema 存在（多 worker 各自执行亦安全）
    recovered = mark_stale_runs_interrupted()
    stuck = recover_stuck_sessions()
    print("=" * 60)
    print("  空调/空压节能测算 FastAPI 后端")
    print("=" * 60)
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  ENV          : {settings.environment}")
    print(f"  DATABASE     : {'OK' if db_ping() else '❌ 不可用'} ({settings.database_url.split('://')[0]})")
    print(f"  REQUIRE_AUTH : {settings.require_auth}")
    print(f"  AUTH_SECRET  : {'已配置' if settings.auth_secret else '未配置（仅开发模式允许）'}")
    print(f"  ANTHROPIC_KEY: {'已配置' if os.getenv('ANTHROPIC_API_KEY') else '❌ 缺失'}")
    print(f"  ANTHROPIC_BASE_URL: {os.getenv('ANTHROPIC_BASE_URL', '(default)')}")
    skills_dir = PROJECT_ROOT / ".claude" / "skills"
    print(f"  skills 目录  : {skills_dir} ({'存在' if skills_dir.exists() else '❌ 不存在'})")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "")
    if base_url.startswith("http://"):
        print(f"  ⚠️ 安全告警  : ANTHROPIC_BASE_URL 使用明文 HTTP（{base_url}），"
              "客户数据将未加密传输——上线前必须切换 HTTPS 或私有化端点")
    if recovered:
        print(f"  启动恢复     : {recovered} 个中断的 agent 运行可 resume")
    if stuck:
        print(f"  启动恢复     : {stuck} 个旧路径卡死会话已置为 error（需重新发起）")
    print("=" * 60)

    yield

    print("FastAPI 已关闭")
