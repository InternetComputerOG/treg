"""archive row retention cursor - where the retention job has got to in archivekey

Revision ID: 0108
Revises: 0107
Create Date: 2026-11-16

`archiveretentioncursor` holds one row: the last `archivekey.id` the row-retention job
(`treg-worker archive retain`) finished, so each bounded run resumes instead of rereading the
front of the table. A new, empty table: no existing table is locked or rewritten.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0108"
down_revision: str | Sequence[str] | None = "0107"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "archiveretentioncursor",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("after_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("passes", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("archiveretentioncursor")
