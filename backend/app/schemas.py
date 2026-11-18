from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

ViewType = Literal["Perspective", "Front", "Rear", "Left", "Right", "Top", "Bottom", "Other"]
ImageStatus = Literal["Uploaded", "Queued", "Processing", "Review required", "Completed", "Failed"]
ApprovalStatus = Literal["pending", "approved", "rejected"]
QualityStatus = Literal["passed", "review_required", "failed"]


def _as_utc(v: datetime) -> datetime:
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- auth ----
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        local, _, domain = v.partition("@")
        if not local or "." not in domain or " " in v:
            raise ValueError("must be a valid email address")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(ORM):
    id: int
    name: str
    email: str
    created_at: UtcDatetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---- projects ----
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    status: Literal["active", "archived"] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("must not be blank")
        return v


class ProjectOut(ORM):
    id: int
    name: str
    description: str
    status: str
    user_id: int
    created_at: UtcDatetime
    updated_at: UtcDatetime
    image_count: int = 0
    status_counts: dict[str, int] = {}


# ---- images / results ----
class ResultOut(BaseModel):
    id: int
    image_id: int
    status: ImageStatus
    quality_status: QualityStatus = "passed"
    quality_warnings: list[str] | None = None
    error_message: str | None
    failed_stage: str | None
    metrics: dict | None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    # Public API URLs (never filesystem paths). None until that asset exists.
    urls: dict[str, str | None]


class JobOut(ORM):
    id: int
    image_id: int
    job_status: ImageStatus
    started_at: UtcDatetime | None
    completed_at: UtcDatetime | None
    error_message: str | None
    failed_stage: str | None
    retry_count: int


class ImageOut(BaseModel):
    id: int
    project_id: int
    original_filename: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    view_type: ViewType
    processing_status: ImageStatus
    approval_status: ApprovalStatus = "pending"
    is_active: bool = True
    created_at: UtcDatetime
    original_url: str
    result: ResultOut | None = None
    job: JobOut | None = None


class ImageUpdate(BaseModel):
    view_type: ViewType


class ReviewActionIn(BaseModel):
    action: Literal["approve", "reject"]
    notes: str | None = None


# ---- views and project summary ----
class ProjectViewsOut(BaseModel):
    project_id: int
    views: dict[str, ImageOut | None]
    other_images: list[ImageOut]
    is_ready_to_process: bool
    is_ready_for_document: bool
    missing_views: list[str]
    unapproved_views: list[str]
    view_order: list[str]


# ---- document metadata & generated documents ----
class DocumentMetadataIn(BaseModel):
    applicant_name: str = Field(default="", max_length=255)
    product_title: str = Field(default="", max_length=255)
    application_number: str | None = Field(default=None, max_length=100)
    filing_date: str = Field(default="", max_length=50)
    novelty_statement: str = Field(default="", max_length=2000)
    disclaimer: str = Field(default="", max_length=4000)
    agent_name: str | None = Field(default=None, max_length=255)
    agent_registration_number: str | None = Field(default=None, max_length=100)
    agent_designation: str | None = Field(default="Agent for the Applicant", max_length=100)


class DocumentMetadataOut(ORM):
    id: int
    project_id: int
    applicant_name: str
    product_title: str
    application_number: str | None
    filing_date: str
    novelty_statement: str
    disclaimer: str
    agent_name: str | None
    agent_registration_number: str | None
    agent_designation: str | None
    has_signature: bool = False
    signature_url: str | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ProjectDocumentOut(ORM):
    id: int
    project_id: int
    version: int
    filename: str
    page_count: int = 7
    created_at: UtcDatetime
    download_url: str


class DocumentGenerationAccepted(BaseModel):
    project_id: int
    document: ProjectDocumentOut



class ProcessAccepted(BaseModel):
    image_id: int
    status: ImageStatus
    job: JobOut


class ProjectProcessAccepted(BaseModel):
    project_id: int
    queued: list[int]
    skipped: list[int]


class DashboardOut(BaseModel):
    project_count: int
    image_count: int
    status_counts: dict[str, int]
    recent_projects: list[ProjectOut]
    recent_results: list[ImageOut]
