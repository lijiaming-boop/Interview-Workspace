"""task locks（P2 改造：跨进程任务互斥）

Revision ID: a1f2c3d4e5b6
Revises: ccf6763665cc
Create Date: 2026-08-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1f2c3d4e5b6'
down_revision: Union[str, Sequence[str], None] = 'ccf6763665cc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升级：新增 task_locks 表（多 worker/多副本下防重复执行）"""
    op.create_table(
        'task_locks',
        sa.Column('task_id', sa.String(length=160), nullable=False),
        sa.Column('owner', sa.String(length=160), nullable=False),
        sa.Column('claimed_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('task_id'),
    )


def downgrade() -> None:
    """降级：删除 task_locks 表"""
    op.drop_table('task_locks')
