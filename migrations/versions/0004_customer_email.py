"""Customer email address for email notifications (E8-4). Nullable: not every customer has one.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE customers ADD COLUMN email TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE customers DROP COLUMN IF EXISTS email")
