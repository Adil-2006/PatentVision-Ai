from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

REQUIRED_VIEW_TYPES = ("Perspective", "Front", "Rear", "Right", "Left", "Top", "Bottom")
VIEW_TYPES = ("Perspective", "Front", "Rear", "Left", "Right", "Top", "Bottom", "Other")
IMAGE_STATUSES = ("Uploaded", "Queued", "Processing", "Review required", "Completed", "Failed")
ACTIVE_STATUSES = ("Queued", "Processing")
APPROVAL_STATUSES = ("pending", "approved", "rejected")
QUALITY_STATUSES = ("passed", "review_required", "failed")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    projects: Mapped[list["Project"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="active")
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    owner: Mapped[User] = relationship(back_populates="projects")
    images: Mapped[list["Image"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Image.id",
    )
    document_metadata: Mapped["DocumentMetadata | None"] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    documents: Mapped[list["ProjectDocument"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProjectDocument.version.desc()",
    )


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(50))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    view_type: Mapped[str] = mapped_column(String(20), default="Other")
    processing_status: Mapped[str] = mapped_column(String(20), default="Uploaded")
    approval_status: Mapped[str] = mapped_column(String(20), default="pending")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="images")
    result: Mapped["AIResult | None"] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )
    job: Mapped["ProcessingJob | None"] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )


class AIResult(Base):
    __tablename__ = "ai_results"
    __table_args__ = (UniqueConstraint("image_id", name="uq_ai_results_image_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), index=True
    )
    background_removed_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    outline_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    patent_drawing_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    drawing_svg_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="Queued")
    quality_status: Mapped[str] = mapped_column(String(20), default="passed")
    quality_warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    failed_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    image: Mapped[Image] = relationship(back_populates="result")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (UniqueConstraint("image_id", name="uq_processing_jobs_image_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), index=True
    )
    job_status: Mapped[str] = mapped_column(String(20), default="Queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    failed_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    image: Mapped[Image] = relationship(back_populates="job")


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    __table_args__ = (UniqueConstraint("project_id", name="uq_document_metadata_project_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    applicant_name: Mapped[str] = mapped_column(String(255), default="")
    product_title: Mapped[str] = mapped_column(String(255), default="")
    application_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filing_date: Mapped[str] = mapped_column(String(50), default="")
    novelty_statement: Mapped[str] = mapped_column(Text, default="")
    disclaimer: Mapped[str] = mapped_column(Text, default="")
    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent_registration_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    agent_designation: Mapped[str | None] = mapped_column(String(100), default="Agent for the Applicant")
    signature_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    project: Mapped[Project] = relationship(back_populates="document_metadata")


class ProjectDocument(Base):
    __tablename__ = "project_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    storage_key: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(255))
    page_count: Mapped[int] = mapped_column(Integer, default=7)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="documents")
