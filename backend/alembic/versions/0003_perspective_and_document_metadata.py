"""Add perspective support, document metadata, project documents, and quality approval fields

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-18
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Update images table
    with op.batch_alter_table("images") as batch:
        batch.add_column(
            sa.Column("approval_status", sa.String(20), server_default="pending", nullable=False)
        )
        batch.add_column(
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("1"), nullable=False)
        )

    # 2. Update ai_results table
    with op.batch_alter_table("ai_results") as batch:
        batch.add_column(
            sa.Column("quality_status", sa.String(20), server_default="passed", nullable=False)
        )
        batch.add_column(
            sa.Column("quality_warnings", sa.JSON(), nullable=True)
        )

    # 3. Create document_metadata table
    op.create_table(
        "document_metadata",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("applicant_name", sa.String(255), server_default="", nullable=False),
        sa.Column("product_title", sa.String(255), server_default="", nullable=False),
        sa.Column("application_number", sa.String(100), nullable=True),
        sa.Column("filing_date", sa.String(50), server_default="", nullable=False),
        sa.Column("novelty_statement", sa.Text(), server_default="", nullable=False),
        sa.Column("disclaimer", sa.Text(), server_default="", nullable=False),
        sa.Column("agent_name", sa.String(255), nullable=True),
        sa.Column("agent_registration_number", sa.String(100), nullable=True),
        sa.Column("agent_designation", sa.String(100), server_default="Agent for the Applicant", nullable=True),
        sa.Column("signature_key", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", name="uq_document_metadata_project_id"),
    )
    with op.batch_alter_table("document_metadata") as batch:
        batch.create_index("ix_document_metadata_project_id", ["project_id"], unique=False)

    # 4. Create project_documents table
    op.create_table(
        "project_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("page_count", sa.Integer(), server_default="7", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    with op.batch_alter_table("project_documents") as batch:
        batch.create_index("ix_project_documents_project_id", ["project_id"], unique=False)


def downgrade() -> None:
    op.drop_table("project_documents")
    op.drop_table("document_metadata")

    with op.batch_alter_table("ai_results") as batch:
        batch.drop_column("quality_warnings")
        batch.drop_column("quality_status")

    with op.batch_alter_table("images") as batch:
        batch.drop_column("is_active")
        batch.drop_column("approval_status")
