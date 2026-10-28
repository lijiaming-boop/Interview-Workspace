"""task locks 租约字段（P1 修复：任务锁不是安全租约）

新增 lease_token（唯一租约，释放/续租必须携带）与 heartbeat_at
（持锁方心跳续租时间），配合 claim_task 的原子条件 UPDATE。

Revision ID: b7e4a1c9d2f3
Revises: a1f2c3d4e5b6
Create Date: 2026-08-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e4a1c9d2f3'
down_revision: Union[str, Sequence[str], None] = 'a1f2c3d4e5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升级：task_locks 增加 lease_token / heartbeat_at"""
    op.add_column('task_locks',
                  sa.Column('lease_token', sa.String(length=64),
                            nullable=False, server_default=''))
    op.add_column('task_locks',
                  sa.Column('heartbeat_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """降级：移除租约字段"""
    op.drop_column('task_locks', 'heartbeat_at')
    op.drop_column('task_locks', 'lease_token')
