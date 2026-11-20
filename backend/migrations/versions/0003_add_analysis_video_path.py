"""Add video_path column to analyses table

Revision ID: 0003_add_analysis_video_path
Revises: 0002_add_user_role
Create Date: 2026-11-18 19:40:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_add_analysis_video_path"
down_revision: str | None = "0002_add_user_role"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "video_path",
                sa.String(length=512),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.drop_column("video_path")
