import uuid

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_owned_image, get_owned_project
from app.api.files import file_response
from app.core.config import get_settings
from app.core.database import get_db
from app.core.errors import ApiError
from app.core.ratelimit import rate_limit
from app.core.storage import storage
from app.models import REQUIRED_VIEW_TYPES, VIEW_TYPES, Image, User, utcnow
from app.schemas import ImageOut, ImageUpdate, ProjectViewsOut, ReviewActionIn
from app.services.presenters import image_out
from app.services.processing import expire_stale
from app.services.uploads import validate_upload

router = APIRouter(tags=["images"])

_CHUNK = 1024 * 1024


def _read_limited(file: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := file.file.read(_CHUNK):
        buf.extend(chunk)
        if len(buf) > limit:
            raise ApiError(
                413,
                "file_too_large",
                f"The file exceeds the {limit / (1024 * 1024):.0f} MB upload limit.",
                "Upload a smaller image.",
            )
    return bytes(buf)


@router.post(
    "/projects/{project_id}/images",
    response_model=ImageOut,
    status_code=201,
    dependencies=[Depends(rate_limit("upload"))],
)
def upload_image(
    project_id: int,
    file: UploadFile = File(...),
    view_type: str = Form("Other"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(db, user, project_id)
    if view_type not in VIEW_TYPES:
        raise ApiError(422, "invalid_view", f"view_type must be one of {', '.join(VIEW_TYPES)}.")

    settings = get_settings()
    data = _read_limited(file, settings.max_upload_bytes)
    info = validate_upload(data, file.filename, settings.max_image_pixels)

    # Safe per-view replacement: if a required view is uploaded, clean up old assets and invalidate prior result/approval
    if view_type in REQUIRED_VIEW_TYPES:
        existing = db.scalar(
            select(Image).where(
                Image.project_id == project.id,
                Image.view_type == view_type,
                Image.is_active == True,
            )
        )
        if existing:
            if existing.processing_status in ("Queued", "Processing"):
                raise ApiError(
                    409,
                    "already_processing",
                    f"The current {view_type} view is being processed.",
                    "Wait for the job to finish before replacing.",
                )
            old_key = existing.storage_key
            old_id = existing.id
            db.delete(existing)
            db.flush()
            storage.delete(old_key)
            storage.delete_tree(f"generated/{project.id}/{old_id}")

    key = f"uploads/{project.id}/{uuid.uuid4().hex}.{info.ext}"
    storage.save_bytes(key, data, info.mime_type)  # original bytes are preserved untouched
    try:
        image = Image(
            project_id=project.id,
            original_filename=info.safe_name,
            storage_key=key,
            mime_type=info.mime_type,
            size_bytes=len(data),
            width=info.width,
            height=info.height,
            view_type=view_type,
            processing_status="Uploaded",
            approval_status="pending",
            is_active=True,
        )
        db.add(image)
        project.updated_at = utcnow()
        db.commit()
    except Exception:
        db.rollback()
        storage.delete(key)
        raise
    return image_out(image)


@router.post(
    "/projects/{project_id}/views/{view_type}",
    response_model=ImageOut,
    status_code=201,
    dependencies=[Depends(rate_limit("upload"))],
)
def upload_or_replace_view(
    project_id: int,
    view_type: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload or replace one of the fixed required view slots."""
    if view_type not in VIEW_TYPES:
        raise ApiError(422, "invalid_view", f"view_type must be one of {', '.join(VIEW_TYPES)}.")
    return upload_image(project_id, file=file, view_type=view_type, db=db, user=user)


@router.get("/projects/{project_id}/views", response_model=ProjectViewsOut)
def list_project_views(
    project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """List all fixed required view slots, their active status, and document readiness."""
    project = get_owned_project(db, user, project_id)
    expire_stale(db, project.images)
    active_images = [img for img in project.images if getattr(img, "is_active", True)]

    views_map: dict[str, ImageOut | None] = {v: None for v in REQUIRED_VIEW_TYPES}
    other_images: list[ImageOut] = []

    for img in active_images:
        if img.view_type in REQUIRED_VIEW_TYPES:
            views_map[img.view_type] = image_out(img)
        else:
            other_images.append(image_out(img))

    missing_views = [v for v in REQUIRED_VIEW_TYPES if views_map[v] is None]
    unapproved_views = [
        v
        for v in REQUIRED_VIEW_TYPES
        if views_map[v] is None or views_map[v].approval_status != "approved"
    ]

    return ProjectViewsOut(
        project_id=project.id,
        views=views_map,
        other_images=other_images,
        is_ready_to_process=len(missing_views) == 0,
        is_ready_for_document=len(missing_views) == 0 and len(unapproved_views) == 0,
        missing_views=missing_views,
        unapproved_views=unapproved_views,
        view_order=list(REQUIRED_VIEW_TYPES),
    )


@router.delete("/projects/{project_id}/views/{view_type}", status_code=204)
def delete_view(
    project_id: int,
    view_type: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Remove the active image for a specific view slot and clear its artifacts."""
    project = get_owned_project(db, user, project_id)
    image = db.scalar(
        select(Image).where(
            Image.project_id == project.id,
            Image.view_type == view_type,
            Image.is_active == True,
        )
    )
    if not image:
        raise ApiError(404, "view_not_found", f"No image found for view slot '{view_type}'.")
    if image.processing_status in ("Queued", "Processing"):
        raise ApiError(409, "already_processing", "This image is being processed.", "Wait for the job to finish.")
    key, pid, iid = image.storage_key, image.project_id, image.id
    db.delete(image)
    db.commit()
    storage.delete(key)
    storage.delete_tree(f"generated/{pid}/{iid}")
    return Response(status_code=204)


@router.post("/images/{image_id}/review", response_model=ImageOut)
def review_image(
    image_id: int,
    body: ReviewActionIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Approve or reject a processed technical drawing."""
    image = get_owned_image(db, user, image_id)
    if body.action == "approve":
        if (
            image.processing_status == "Failed"
            or not image.result
            or not image.result.patent_drawing_path
        ):
            raise ApiError(
                400,
                "cannot_approve",
                "Cannot approve an image that has failed processing or has not produced a drawing.",
                "Process or retry the image first.",
            )
        image.approval_status = "approved"
        image.processing_status = "Completed"
        if image.result:
            image.result.status = "Completed"
    elif body.action == "reject":
        image.approval_status = "rejected"
    db.commit()
    return image_out(image)


@router.get("/projects/{project_id}/images", response_model=list[ImageOut])
def list_images(project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    project = get_owned_project(db, user, project_id)
    expire_stale(db, project.images)
    return [image_out(i) for i in project.images]


@router.get("/images/{image_id}", response_model=ImageOut)
def get_image(image_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    image = get_owned_image(db, user, image_id)
    expire_stale(db, [image])
    return image_out(image)


@router.patch("/images/{image_id}", response_model=ImageOut)
def update_image(
    image_id: int, body: ImageUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    image = get_owned_image(db, user, image_id)
    image.view_type = body.view_type
    db.commit()
    return image_out(image)


@router.get("/images/{image_id}/original")
def download_original(
    image_id: int, download: bool = False, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    image = get_owned_image(db, user, image_id)
    return file_response(
        image.storage_key,
        image.mime_type,
        image.original_filename if download else None,
        "Re-upload the image.",
    )


@router.get("/images/{image_id}/diagnostics")
def list_diagnostics(
    image_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    image = get_owned_image(db, user, image_id)
    stages = [
        {"stage": "original_normalized", "label": "1. Normalized Photograph"},
        {"stage": "segmentation_mask", "label": "2. Solid Segmentation Mask"},
        {"stage": "photo_edges", "label": "3. Photographic Edges (CLAHE + Canny)"},
        {"stage": "internal_edges", "label": "4. Internal Structural Details"},
        {"stage": "external_silhouette", "label": "5. External Product Silhouette"},
        {"stage": "combined_edges", "label": "6. Combined Technical Edges"},
        {"stage": "cleaned_contours", "label": "7. Cleaned Contours"},
        {"stage": "patent_drawing", "label": "8. Final Patent Illustration"},
    ]
    available = []
    prefix = f"diagnostics/{image.project_id}/{image.id}"
    for s in stages:
        key = f"{prefix}/{s['stage']}.png"
        if storage.exists(key):
            available.append(
                {
                    "stage": s["stage"],
                    "label": s["label"],
                    "url": f"/images/{image.id}/diagnostics/{s['stage']}",
                }
            )
    return {"image_id": image.id, "stages": available}


@router.get("/images/{image_id}/diagnostics/{stage_name}")
def download_diagnostic(
    image_id: int,
    stage_name: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    image = get_owned_image(db, user, image_id)
    key = f"diagnostics/{image.project_id}/{image.id}/{stage_name}.png"
    if not storage.exists(key):
        raise ApiError(
            404,
            "diagnostic_not_found",
            f"Diagnostic stage '{stage_name}' was not found for this image.",
            "Process or reprocess the image first.",
        )
    return file_response(key, "image/png", None, "Reprocess the image.")


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    image = get_owned_image(db, user, image_id)
    if image.processing_status in ("Queued", "Processing"):
        raise ApiError(409, "already_processing", "This image is being processed.", "Wait for the job to finish.")
    key, pid, iid = image.storage_key, image.project_id, image.id
    db.delete(image)
    db.commit()
    storage.delete(key)
    storage.delete_tree(f"generated/{pid}/{iid}")
    storage.delete_tree(f"diagnostics/{pid}/{iid}")
    return Response(status_code=204)
