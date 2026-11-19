"""Document metadata management and 7-page PDF patent illustration document endpoints."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_owned_project
from app.api.files import file_response
from app.core.database import get_db
from app.core.errors import ApiError
from app.core.ratelimit import rate_limit
from app.core.storage import storage
from app.models import (
    REQUIRED_VIEW_TYPES,
    DocumentMetadata,
    Image,
    Project,
    ProjectDocument,
    User,
    utcnow,
)
from app.schemas import (
    DocumentGenerationAccepted,
    DocumentMetadataIn,
    DocumentMetadataOut,
    ProjectDocumentOut,
)
from app.services.document_generation import (
    DEFAULT_DISCLAIMER_BULLETS,
    DEFAULT_NOVELTY_TEMPLATE,
    PDF_VIEW_ORDER,
    generate_patent_pdf,
)

router = APIRouter(tags=["documents"])


def _metadata_out(meta: DocumentMetadata) -> DocumentMetadataOut:
    has_sig = bool(meta.signature_key and storage.exists(meta.signature_key))
    sig_url = f"/projects/{meta.project_id}/document-metadata/signature" if has_sig else None
    return DocumentMetadataOut(
        id=meta.id,
        project_id=meta.project_id,
        applicant_name=meta.applicant_name,
        product_title=meta.product_title,
        application_number=meta.application_number,
        filing_date=meta.filing_date,
        novelty_statement=meta.novelty_statement,
        disclaimer=meta.disclaimer,
        agent_name=meta.agent_name,
        agent_registration_number=meta.agent_registration_number,
        agent_designation=meta.agent_designation,
        has_signature=has_sig,
        signature_url=sig_url,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )


def _get_or_create_metadata(db: Session, project: Project) -> DocumentMetadata:
    meta = db.scalar(
        select(DocumentMetadata).where(DocumentMetadata.project_id == project.id)
    )
    if meta is None:
        today_str = datetime.now(timezone.utc).strftime("%d/%m/%Y")
        default_novelty = DEFAULT_NOVELTY_TEMPLATE.format(title=project.name)
        default_disclaimer = "\n".join(f"• {b}" for b in DEFAULT_DISCLAIMER_BULLETS)
        meta = DocumentMetadata(
            project_id=project.id,
            applicant_name="",
            product_title=project.name,
            application_number=None,
            filing_date=today_str,
            novelty_statement=default_novelty,
            disclaimer=default_disclaimer,
            agent_name=None,
            agent_registration_number=None,
            agent_designation="Agent for the Applicant",
            signature_key=None,
        )
        db.add(meta)
        db.commit()
        db.refresh(meta)
    return meta


@router.get("/projects/{project_id}/document-metadata", response_model=DocumentMetadataOut)
def get_document_metadata(
    project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    project = get_owned_project(db, user, project_id)
    meta = _get_or_create_metadata(db, project)
    return _metadata_out(meta)


@router.put("/projects/{project_id}/document-metadata", response_model=DocumentMetadataOut)
def update_document_metadata(
    project_id: int,
    body: DocumentMetadataIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(db, user, project_id)
    meta = _get_or_create_metadata(db, project)
    meta.applicant_name = body.applicant_name
    meta.product_title = body.product_title
    meta.application_number = body.application_number
    meta.filing_date = body.filing_date
    if not body.novelty_statement or any(k in body.novelty_statement.lower() for k in ["brushmate", "brush", "jug", "pitcher"]):
        meta.novelty_statement = DEFAULT_NOVELTY_TEMPLATE.format(title=body.product_title or project.name)
    else:
        meta.novelty_statement = body.novelty_statement
    meta.disclaimer = body.disclaimer
    meta.agent_name = body.agent_name
    meta.agent_registration_number = body.agent_registration_number
    meta.agent_designation = body.agent_designation
    meta.updated_at = utcnow()
    db.commit()
    return _metadata_out(meta)


@router.post(
    "/projects/{project_id}/document-metadata/signature",
    response_model=DocumentMetadataOut,
    dependencies=[Depends(rate_limit("upload"))],
)
def upload_signature(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(db, user, project_id)
    meta = _get_or_create_metadata(db, project)
    data = file.file.read()
    if len(data) > 2 * 1024 * 1024:
        raise ApiError(413, "file_too_large", "Signature image must be under 2 MB.")
    if meta.signature_key:
        storage.delete(meta.signature_key)
    key = f"signatures/{project.id}/signature_{uuid_hex()}.png"
    storage.save_bytes(key, data, "image/png")
    meta.signature_key = key
    meta.updated_at = utcnow()
    db.commit()
    return _metadata_out(meta)


@router.delete("/projects/{project_id}/document-metadata/signature", status_code=204)
def delete_signature(
    project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    project = get_owned_project(db, user, project_id)
    meta = _get_or_create_metadata(db, project)
    if meta.signature_key:
        storage.delete(meta.signature_key)
        meta.signature_key = None
        meta.updated_at = utcnow()
        db.commit()
    return Response(status_code=204)


@router.get("/projects/{project_id}/document-metadata/signature")
def get_signature_image(
    project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    project = get_owned_project(db, user, project_id)
    meta = _get_or_create_metadata(db, project)
    if not meta.signature_key or not storage.exists(meta.signature_key):
        raise ApiError(404, "signature_not_found", "No signature has been uploaded.")
    return file_response(meta.signature_key, "image/png", None, "Re-upload the signature.")


@router.post(
    "/projects/{project_id}/documents",
    response_model=DocumentGenerationAccepted,
    dependencies=[Depends(rate_limit("process"))],
)
def generate_document(
    project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Generate the official 7-page patent drawing PDF following reference layout."""
    project = get_owned_project(db, user, project_id)
    meta = _get_or_create_metadata(db, project)

    # 1. Check all seven views exist and are active
    images = db.scalars(
        select(Image)
        .where(Image.project_id == project.id, Image.is_active == True)
        .order_by(Image.id.desc())
    ).all()
    active_images: dict[str, Image] = {}
    for img in images:
        if img.view_type in REQUIRED_VIEW_TYPES and img.view_type not in active_images:
            active_images[img.view_type] = img
    missing = [v for v in REQUIRED_VIEW_TYPES if v not in active_images]
    if missing:
        raise ApiError(
            400,
            "missing_required_views",
            f"Cannot generate patent document: missing {len(missing)} required view(s): {', '.join(missing)}.",
            "Upload photos for all seven required views.",
        )

    # 2. Check all seven views are approved, completed, and have valid drawings in storage
    unapproved = [
        v
        for v, img in active_images.items()
        if img.approval_status != "approved"
        or img.processing_status != "Completed"
        or not img.result
        or not img.result.patent_drawing_path
        or not storage.exists(img.result.patent_drawing_path)
    ]
    if unapproved:
        raise ApiError(
            400,
            "views_not_approved",
            f"Cannot generate patent document: {len(unapproved)} view(s) are not ready/approved: {', '.join(unapproved)}.",
            "Review and approve all seven technical drawings before generating the document.",
        )

    # 3. Collect drawing PNG bytes
    drawings: dict[str, bytes] = {}
    for v in REQUIRED_VIEW_TYPES:
        img = active_images[v]
        if not img.result or not img.result.patent_drawing_path:
            raise ApiError(
                400,
                "drawing_missing",
                f"Drawing file for view '{v}' is missing from storage.",
                "Process or retry the view.",
            )
        drawings[v] = storage.read_bytes(img.result.patent_drawing_path)

    sig_bytes = None
    if meta.signature_key and storage.exists(meta.signature_key):
        sig_bytes = storage.read_bytes(meta.signature_key)

    pdf_bytes = generate_patent_pdf(meta, drawings, sig_bytes)

    # Determine next document version
    latest = db.scalar(
        select(ProjectDocument)
        .where(ProjectDocument.project_id == project.id)
        .order_by(ProjectDocument.version.desc())
    )
    version = (latest.version + 1) if latest else 1
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in (meta.product_title or project.name))
    filename = f"{safe_title}_Patent_Drawings_v{version}.pdf"
    doc_key = f"documents/{project.id}/{filename}"
    storage.save_bytes(doc_key, pdf_bytes, "application/pdf")

    # Check if a document record already exists for this exact project and storage key to prevent duplicate entries
    existing_doc = db.scalar(
        select(ProjectDocument).where(
            ProjectDocument.project_id == project.id,
            ProjectDocument.storage_key == doc_key,
        )
    )
    if existing_doc:
        existing_doc.created_at = utcnow()
        doc = existing_doc
    else:
        doc = ProjectDocument(
            project_id=project.id,
            version=version,
            storage_key=doc_key,
            filename=filename,
            page_count=7,
        )
        db.add(doc)
    db.commit()
    db.refresh(doc)

    doc_out = ProjectDocumentOut(
        id=doc.id,
        project_id=doc.project_id,
        version=doc.version,
        filename=doc.filename,
        page_count=doc.page_count,
        created_at=doc.created_at,
        download_url=f"/documents/{doc.id}/download",
    )
    return DocumentGenerationAccepted(project_id=project.id, document=doc_out)


@router.get("/projects/{project_id}/documents", response_model=list[ProjectDocumentOut])
def list_documents(
    project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    project = get_owned_project(db, user, project_id)
    docs = db.scalars(
        select(ProjectDocument)
        .where(ProjectDocument.project_id == project.id)
        .order_by(ProjectDocument.version.desc())
    ).all()
    return [
        ProjectDocumentOut(
            id=d.id,
            project_id=d.project_id,
            version=d.version,
            filename=d.filename,
            page_count=d.page_count,
            created_at=d.created_at,
            download_url=f"/documents/{d.id}/download",
        )
        for d in docs
    ]


def _get_owned_document(db: Session, user: User, document_id: int) -> ProjectDocument:
    doc = db.scalar(
        select(ProjectDocument)
        .join(Project, ProjectDocument.project_id == Project.id)
        .where(ProjectDocument.id == document_id, Project.user_id == user.id)
    )
    if doc is None:
        raise ApiError(404, "document_not_found", "Document not found.")
    return doc


@router.get("/projects/{project_id}/documents/{document_id}/download")
@router.get("/documents/{document_id}/download")
def download_document(
    document_id: int,
    project_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    doc = _get_owned_document(db, user, document_id)
    if project_id is not None and doc.project_id != project_id:
        raise ApiError(404, "document_not_found", "Document does not belong to the requested project.")
    return file_response(doc.storage_key, "application/pdf", doc.filename, "Regenerate the document.")


@router.get("/projects/{project_id}/documents/{document_id}/preview")
@router.get("/documents/{document_id}/preview")
def preview_document(
    document_id: int,
    project_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    doc = _get_owned_document(db, user, document_id)
    if project_id is not None and doc.project_id != project_id:
        raise ApiError(404, "document_not_found", "Document does not belong to the requested project.")
    return file_response(doc.storage_key, "application/pdf", None, "Regenerate the document.")


def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex
