"""Add analysis_outbox table for reliable transactional outbox pattern

Revision ID: 0004_add_analysis_outbox
Revises: 0003_add_analysis_video_path
Create Date: 2026-11-18 20:55:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_add_analysis_outbox"
down_revision: str | None = "0003_add_analysis_video_path"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_outbox",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_outbox_analysis_id", "analysis_outbox", ["analysis_id"])
    op.create_index("ix_analysis_outbox_status", "analysis_outbox", ["status"])


def downgrade() -> None:
    op.drop_index("ix_analysis_outbox_status", table_name="analysis_outbox")
    op.drop_index("ix_analysis_outbox_analysis_id", table_name="analysis_outbox")
    op.drop_table("analysis_outbox")
