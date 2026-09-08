"""Create durable platform verification evidence."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "verification",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("environment", sa.String(), nullable=False),
        sa.Column("component", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("evidence", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("verification")
