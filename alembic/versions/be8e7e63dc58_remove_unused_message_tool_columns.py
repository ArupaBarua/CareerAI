"""remove unused message tool columns

Revision ID: be8e7e63dc58
Revises: 2f8ca093479e
Create Date: 2026-09-20 15:22:27.596186

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'be8e7e63dc58'
down_revision: Union[str, Sequence[str], None] = '2f8ca093479e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.drop_column(
        "messages",
        "tool_name",
    )

    op.drop_column(
        "messages",
        "tool_call_id",
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.add_column(
        "messages",
        sa.Column(
            "tool_name",
            sa.String(length=100),
            nullable=True,
        ),
    )

    op.add_column(
        "messages",
        sa.Column(
            "tool_call_id",
            sa.String(length=255),
            nullable=True,
        ),
    )
