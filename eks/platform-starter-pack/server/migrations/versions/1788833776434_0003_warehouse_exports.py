"""Track durable metadata exports and validated warehouse snapshots."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "export",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner", sa.String(), nullable=False),
        sa.Column("environment", sa.String(), nullable=False),
        sa.Column("request_key", sa.String(), nullable=False),
        sa.Column("trigger", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("workflow_name", sa.String(), nullable=True),
        sa.Column("manifest_key", sa.String(), nullable=True),
        sa.Column("row_counts", sa.JSON(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_key"),
    )
    for name in ("owner", "environment", "status", "deleted_at"):
        op.create_index(f"ix_export_{name}", "export", [name])


def downgrade():
    op.drop_table("export")
