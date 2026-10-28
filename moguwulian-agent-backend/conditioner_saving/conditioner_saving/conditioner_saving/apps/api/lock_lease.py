"""任务锁租约心跳（审查报告2 P1-3：续租失败后任务必须停止）

租约互斥的前提是持锁方周期性证明存活：renew_task() 返回 False 表示
锁已被其他实例接管，旧任务必须停止执行——否则新旧任务会并发写同一
会话，TTL 租约失去互斥意义。

支持双后端与双执行模型：
1. Redis 跨进程分布式锁 (`RedisLeaseStore`)：基于 `SET key token NX EX ttl` 原子抢锁，
   配合 Lua 脚本实现 compare-and-expire 续租与 compare-and-delete 释放，杜绝误删他人锁。
2. 异步协程心跳续租 (`AsyncRedisLeaseLock`) 与后台守护线程续租 (`LockLease`)：
   每隔 `max(5.0, ttl_seconds / 3)` 周期续租，锁易主时置位 `lock_lost` 并在安全点抛出 `LockLostError`。
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Callable, Optional


# Redis 原子续租 Lua 脚本：仅当持锁 token 匹配时才延长过期时间（毫秒）
_LUA_RENEW_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("PEXPIRE", KEYS[1], tonumber(ARGV[2]))
else
    return 0
end
"""

# Redis 原子释放 Lua 脚本：仅当持锁 token 匹配时才删除锁键
_LUA_RELEASE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""


class LockLostError(RuntimeError):
    """任务锁已被其他实例接管——协作式中止信号"""


class RedisLeaseStore:
    """基于 Redis 租约机制的跨进程分布式锁原语（SET NX EX + Lua CAS 续租/释放）"""

    def __init__(self, redis_client: Any, key_prefix: str = "cs:lock:") -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    def _lock_key(self, task_type: str, session_id: str) -> str:
        return f"{self._prefix}{task_type}:{session_id}"

    def claim_task(
        self, task_type: str, session_id: str, ttl_seconds: int = 300
    ) -> Optional[str]:
        """原子抢占租约锁：SET key token NX EX ttl_seconds"""
        token = uuid.uuid4().hex
        key = self._lock_key(task_type, session_id)
        acquired = self._redis.set(key, token, nx=True, ex=max(1, int(ttl_seconds)))
        return token if acquired else None

    def renew_task(
        self, task_type: str, session_id: str, token: Optional[str] = None, ttl_seconds: int = 300
    ) -> bool:
        """原子 CAS 续租：仅当 owner token 一致时刷新 TTL"""
        if not token:
            return False
        key = self._lock_key(task_type, session_id)
        ttl_ms = max(1000, int(ttl_seconds * 1000))
        res = self._redis.eval(_LUA_RENEW_SCRIPT, 1, key, token, str(ttl_ms))
        return bool(int(res or 0) == 1)

    def release_task(
        self, task_type: str, session_id: str, token: Optional[str] = None
    ) -> bool:
        """原子 CAS 释放：仅当 owner token 一致时删除锁键"""
        if not token:
            return False
        key = self._lock_key(task_type, session_id)
        res = self._redis.eval(_LUA_RELEASE_SCRIPT, 1, key, token)
        return bool(int(res or 0) == 1)


class AsyncRedisLeaseLock:
    """基于后台 asyncio 协程心跳续租的跨进程分布式锁管理器"""

    def __init__(
        self,
        task_type: str,
        session_id: str,
        token: Optional[str],
        ttl_seconds: int,
        renew: Callable[..., bool],
        release: Optional[Callable[..., Any]] = None,
        on_lost: Optional[Callable[[str], None]] = None,
        interval: Optional[float] = None,
    ) -> None:
        self.task_type = task_type
        self.session_id = session_id
        self.token = token
        self.ttl_seconds = ttl_seconds
        self._renew = renew
        self._release = release
        self._on_lost = on_lost
        self._interval = interval
        self.stop_event = asyncio.Event()
        self.lock_lost = threading.Event()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> "AsyncRedisLeaseLock":
        """启动后台续租协程"""
        if self.token and self._task is None:
            self._task = asyncio.create_task(
                self._heartbeat_loop(),
                name=f"coro-lockbeat-{self.task_type}-{self.session_id}",
            )
        return self

    async def _heartbeat_loop(self) -> None:
        interval = self._interval or max(5.0, self.ttl_seconds / 3.0)
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                renewed = await asyncio.to_thread(
                    self._renew,
                    self.task_type,
                    self.session_id,
                    self.token,
                    ttl_seconds=self.ttl_seconds,
                )
            except Exception:
                # 瞬时网络/存储抖动：下一轮重试，不误判易主
                continue
            if not renewed:
                self.lock_lost.set()
                if self._on_lost is not None:
                    try:
                        self._on_lost(
                            f"⛔ 任务锁已被其他实例接管，本实例任务停止执行"
                            f"（{self.task_type}:{self.session_id}）"
                        )
                    except Exception:
                        pass
                return

    def check(self) -> None:
        """协作式中止检查点：锁易主时抛 LockLostError"""
        if self.lock_lost.is_set():
            raise LockLostError(
                f"任务锁已被其他实例接管，停止执行: {self.task_type}:{self.session_id}"
            )

    async def aclose(self) -> None:
        """停止后台续租协程并按 token 释放分布式锁"""
        self.stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self.token and self._release is not None:
            try:
                await asyncio.to_thread(
                    self._release, self.task_type, self.session_id, token=self.token
                )
            except Exception:
                pass

    async def __aenter__(self) -> "AsyncRedisLeaseLock":
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()


class LockLease:
    """持锁任务的心跳续租器（agent 主线与分阶段线程共用）"""

    def __init__(self, task_type: str, session_id: str, token: Optional[str],
                 ttl_seconds: int, renew: Callable[..., bool],
                 on_lost: Optional[Callable[[str], None]] = None,
                 interval: Optional[float] = None) -> None:
        self.task_type = task_type
        self.session_id = session_id
        self.token = token
        self.ttl_seconds = ttl_seconds
        self._renew = renew
        self._on_lost = on_lost
        self._interval = interval
        self.stop = threading.Event()
        self.lock_lost = threading.Event()

    def start(self) -> "LockLease":
        """启动心跳线程（无 token 时为空操作）"""
        if self.token:
            threading.Thread(target=self._beat, daemon=True,
                             name=f"lockbeat-{self.task_type}-{self.session_id}").start()
        return self

    def _beat(self) -> None:
        interval = self._interval or max(5.0, self.ttl_seconds / 3)
        while not self.stop.wait(interval):
            try:
                renewed = self._renew(self.task_type, self.session_id,
                                      self.token, ttl_seconds=self.ttl_seconds)
            except Exception:
                # 瞬时 DB 故障：下一轮重试，不据此判定易主
                continue
            if not renewed:
                # 条件更新未命中 = 锁已易主/被删除：明确信号，必须停止
                self.lock_lost.set()
                if self._on_lost is not None:
                    try:
                        self._on_lost(
                            f"⛔ 任务锁已被其他实例接管，本实例任务停止执行"
                            f"（{self.task_type}:{self.session_id}）")
                    except Exception:
                        pass
                return

    def check(self) -> None:
        """协作式中止挂点：锁易主时抛 LockLostError"""
        if self.lock_lost.is_set():
            raise LockLostError(
                f"任务锁已被其他实例接管，停止执行: "
                f"{self.task_type}:{self.session_id}")

    def release(self, release: Callable[..., None]) -> None:
        """停止心跳并按 token 释放锁（释放失败不抛）"""
        self.stop.set()
        if self.token:
            try:
                release(self.task_type, self.session_id, token=self.token)
            except Exception:
                pass
