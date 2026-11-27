"""Job lifecycle: claim -> process -> persist, with failures always recorded and visible.

Jobs run inside the request that starts them (about a second per photo), which works on serverless
platforms where work after the response is not guaranteed. A job left Queued/Processing longer than
`stale_job_minutes` lost its worker (crash or timeout) and is surfaced as Failed so it can be retried.
"""
import logging
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.ai.background import BackgroundRemover, get_background_remover
from app.ai.contour import ContourSet
from app.ai.errors import PipelineError
from app.ai.pipeline import run_pipeline
from app.ai.rendering import render_svg
from app.ai.segmentation import Segmenter, get_segmenter
from app.core import database
from app.core.config import get_settings
from app.core.errors import ApiError
from app.core.storage import storage
from app.models import ACTIVE_STATUSES, AIResult, Image, ProcessingJob, utcnow

log = logging.getLogger("patentvision.processing")

# (pipeline output attribute, result column, file name, content type)
_OUTPUTS = (
    ("background_removed", "background_removed_path", "background_removed.png", "image/png"),
    ("mask", "mask_path", "mask.png", "image/png"),
    ("outline", "outline_path", "outline.png", "image/png"),
    ("drawing", "patent_drawing_path", "drawing.png", "image/png"),
    ("drawing_svg", "drawing_svg_path", "drawing.svg", "image/svg+xml"),
)
_slot_lock = threading.Lock()
_slots: threading.BoundedSemaphore | None = None


def _job_slots() -> threading.BoundedSemaphore:
    global _slots
    with _slot_lock:
        if _slots is None:
            _slots = threading.BoundedSemaphore(max(1, get_settings().max_concurrent_jobs))
        return _slots


@lru_cache
def _segmenter(kind: str, checkpoint: str | None, model_type: str) -> Segmenter:
    """One segmenter per config so heavyweight models load once and are reused."""
    return get_segmenter(kind, checkpoint, model_type)


@lru_cache
def _remover(kind: str) -> BackgroundRemover:
    """One background remover per config so u2net weights stay warm across views."""
    return get_background_remover(kind)


def generated_prefix(image: Image) -> str:
    return f"generated/{image.project_id}/{image.id}"


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clear_outputs(result: AIResult, image: Image) -> None:
    try:
        storage.delete_tree(generated_prefix(image))
        storage.delete_tree(f"diagnostics/{image.project_id}/{image.id}")
    except Exception:  # cleanup must never hide the real failure
        log.exception("could not delete partial outputs for image %s", image.id)
    for _, attr, _, _ in _OUTPUTS:
        setattr(result, attr, None)
    result.metrics = None
    result.quality_status = "passed"
    result.quality_warnings = None


def expire_stale(db: Session, images: list[Image]) -> None:
    """Mark jobs that have been in flight longer than the stale timeout as Failed (interrupted)."""
    cutoff = utcnow() - timedelta(minutes=get_settings().stale_job_minutes)
    for image in images:
        if image.processing_status not in ACTIVE_STATUSES or image.result is None:
            continue
        last_change = _as_utc(image.result.updated_at)
        if last_change < cutoff:
            _mark_failed(
                db, image.id, "interrupted", "Processing was interrupted before it finished. Retry processing."
            )


def queue_image(db: Session, image: Image) -> ProcessingJob:
    """Claim an image for processing. Atomic: concurrent requests can never both start a job."""
    expire_stale(db, [image])
    claimed = db.execute(
        update(Image)
        .where(Image.id == image.id, Image.processing_status.not_in(ACTIVE_STATUSES))
        .values(processing_status="Queued")
        .execution_options(synchronize_session=False)
    ).rowcount
    if not claimed:
        db.rollback()
        raise ApiError(
            409,
            "already_processing",
            "This image is already queued or being processed.",
            "Wait for the current job to finish.",
        )
    db.refresh(image)

    job = image.job
    if job is None:
        job = ProcessingJob(image_id=image.id, retry_count=0)
        db.add(job)
    elif job.job_status in ("Failed", "Completed"):
        job.retry_count += 1
    job.job_status = "Queued"
    job.started_at = job.completed_at = None
    job.error_message = job.failed_stage = None

    image.approval_status = "pending"

    result = image.result
    if result is None:
        result = AIResult(image_id=image.id)
        db.add(result)
    result.status = "Queued"
    result.quality_status = "passed"
    result.quality_warnings = None
    result.error_message = result.failed_stage = None
    result.updated_at = utcnow()
    db.commit()
    db.refresh(job)
    return job


def run_job(
    image_id: int,
    detail_level: str | None = None,
    edge_low: float | None = None,
    edge_high: float | None = None,
    smoothing_strength: float | None = None,
    min_contour_length: float | None = None,
    texture_suppression: bool | None = None,
    line_width: float | None = None,
) -> None:
    """Process a claimed image. Owns its own DB session."""
    with _job_slots():
        db = database.SessionLocal()
        try:
            _execute(
                db,
                image_id,
                detail_level=detail_level,
                edge_low=edge_low,
                edge_high=edge_high,
                smoothing_strength=smoothing_strength,
                min_contour_length=min_contour_length,
                texture_suppression=texture_suppression,
                line_width=line_width,
            )
        except Exception:  # last-resort guard: never leave a job stuck in Processing
            log.exception("job for image %s crashed outside the pipeline", image_id)
            db.rollback()
            _mark_failed(db, image_id, "persistence", "Saving the result failed. Retry processing.")
        finally:
            db.close()
            import gc
            gc.collect()


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    return obj


def _execute(
    db: Session,
    image_id: int,
    detail_level: str | None = None,
    edge_low: float | None = None,
    edge_high: float | None = None,
    smoothing_strength: float | None = None,
    min_contour_length: float | None = None,
    texture_suppression: bool | None = None,
    line_width: float | None = None,
) -> None:
    image = db.get(Image, image_id)
    if image is None or image.processing_status != "Queued":
        return
    job, result = image.job, image.result
    image.processing_status = job.job_status = result.status = "Processing"
    job.started_at = utcnow()
    db.commit()

    current = {"stage": "load"}
    try:
        if not storage.exists(image.storage_key):
            raise PipelineError(
                "load", "The original image file is missing from storage.", "Re-upload the image."
            )
        settings = get_settings()
        product_desc = image.project.description if image.project else None
        product_name = image.project.name if image.project else None
        out = run_pipeline(
            storage.read_bytes(image.storage_key),
            settings,
            on_stage=lambda s: current.update(stage=s),
            remover=_remover(settings.background_remover),
            segmenter=_segmenter(settings.segmenter, settings.sam_checkpoint, settings.sam_model_type),
            view_type=image.view_type,
            product_description=product_desc,
            product_name=product_name,
            detail_level=detail_level,
            edge_low=edge_low,
            edge_high=edge_high,
            smoothing_strength=smoothing_strength,
            min_contour_length=min_contour_length,
            texture_suppression=texture_suppression,
            line_width=line_width,
        )
        current["stage"] = "save"
        prefix = generated_prefix(image)
        files = {
            attr: storage.save_bytes(f"{prefix}/{name}", getattr(out, field), content_type)
            for field, attr, name, content_type in _OUTPUTS
        }
        if hasattr(out, "diagnostics") and out.diagnostics:
            diag_urls = {}
            for diag_name, diag_bytes in out.diagnostics.items():
                storage.save_bytes(f"diagnostics/{image.project_id}/{image.id}/{diag_name}.png", diag_bytes, "image/png")
                diag_urls[diag_name] = f"/images/{image.id}/diagnostics/{diag_name}"
            out.metrics["diagnostic_urls"] = diag_urls
        if not all(storage.exists(key) for key in files.values()):
            raise PipelineError("save", "Generated files could not be saved.", "Retry processing.")

        current["stage"] = "persistence"
        for attr, key in files.items():
            setattr(result, attr, key)
        result.metrics = _sanitize_for_json(out.metrics)
        result.quality_status = out.metrics.get("quality_status", "passed")
        result.quality_warnings = out.metrics.get("quality_warnings") or []
        result.error_message = result.failed_stage = None
        job.error_message = job.failed_stage = None

        final_status = "Review required" if result.quality_status == "review_required" else "Completed"
        image.processing_status = job.job_status = result.status = final_status

        job.completed_at = utcnow()
        db.commit()
    except PipelineError as exc:
        db.rollback()
        _mark_failed(db, image_id, exc.stage, exc.user_message())
    except Exception:
        log.exception("unexpected error in stage %s for image %s", current["stage"], image_id)
        db.rollback()
        _mark_failed(
            db,
            image_id,
            current["stage"],
            f"An unexpected error occurred during {current['stage'].replace('_', ' ')}. Retry processing.",
        )


def _mark_failed(db: Session, image_id: int, stage: str, message: str) -> None:
    image = db.get(Image, image_id)
    if image is None:
        return
    job, result = image.job, image.result
    image.processing_status = "Failed"
    image.approval_status = "pending"
    if job:
        job.job_status, job.error_message, job.failed_stage, job.completed_at = "Failed", message, stage, utcnow()
    if result:
        _clear_outputs(result, image)
        result.status, result.error_message, result.failed_stage = "Failed", message, stage
    db.commit()
    log.warning("image %s failed at %s: %s", image_id, stage, message)


# Project-level background job tracking
_project_jobs_lock = threading.Lock()
_active_project_jobs: dict[int, dict] = {}


def reset_stuck_project_images(db: Session, project_id: int) -> int:
    """Force-reset any images in ACTIVE_STATUSES (Queued/Processing) to Failed so they can be retried immediately."""
    images = db.scalars(
        select(Image).where(Image.project_id == project_id, Image.processing_status.in_(ACTIVE_STATUSES))
    ).all()
    count = 0
    for image in images:
        _mark_failed(db, image.id, "user_reset", "Processing was reset. You can now retry generating this drawing.")
        count += 1
    with _project_jobs_lock:
        _active_project_jobs.pop(project_id, None)
    return count


def get_project_processing_status(db: Session, project_id: int) -> dict:
    """Inspect in-flight worker and database to return real-time processing progress."""
    with _project_jobs_lock:
        job_info = _active_project_jobs.get(project_id, {}).copy()

    images = db.scalars(
        select(Image).where(Image.project_id == project_id, Image.is_active == True).order_by(Image.id)
    ).all()

    total = len(images)
    completed_count = sum(1 for i in images if i.processing_status in ("Completed", "Review required"))
    failed_count = sum(1 for i in images if i.processing_status == "Failed")
    processing_count = sum(1 for i in images if i.processing_status == "Processing")
    queued_count = sum(1 for i in images if i.processing_status == "Queued")

    # Determine overall status
    is_worker_alive = bool(job_info and job_info.get("thread") and job_info["thread"].is_alive())
    has_active = processing_count > 0 or queued_count > 0 or is_worker_alive

    if has_active:
        overall_status = "processing"
    elif completed_count > 0 and (completed_count + failed_count) == total:
        overall_status = "completed" if failed_count == 0 else "failed"
    else:
        overall_status = "idle"

    current_view = job_info.get("current_view")
    current_stage = job_info.get("current_stage")

    if has_active and not current_stage:
        active_img = next((i for i in images if i.processing_status == "Processing"), None)
        if active_img:
            current_view = active_img.view_type
            current_stage = f"Processing {active_img.view_type} View..."
        else:
            queued_img = next((i for i in images if i.processing_status == "Queued"), None)
            if queued_img:
                current_view = queued_img.view_type
                current_stage = f"Queued: {queued_img.view_type} View..."

    percent = int((completed_count / max(1, total)) * 100) if total > 0 else 0

    views_status = []
    for img in images:
        drawing_url = None
        if img.result and img.result.patent_drawing_path:
            drawing_url = f"/ai/results/{img.result.id}/download?kind=drawing"
        views_status.append({
            "id": img.id,
            "view_type": img.view_type,
            "status": img.processing_status,
            "error_message": img.job.error_message if img.job else None,
            "failed_stage": img.job.failed_stage if img.job else None,
            "drawing_url": drawing_url,
        })

    return {
        "project_id": project_id,
        "status": overall_status,
        "total": total,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "processing_count": processing_count,
        "queued_count": queued_count,
        "current_view": current_view,
        "current_stage": current_stage,
        "percent": percent,
        "is_complete": bool(total > 0 and completed_count == total),
        "views": views_status,
        "error_message": job_info.get("error_message"),
    }


def _run_project_processing_worker(
    project_id: int,
    image_ids: list[int],
    detail_level: str | None = None,
    texture_suppression: bool | None = None,
    line_width: float | None = None,
) -> None:
    total = len(image_ids)
    log.info("Starting background processing for project %s (%s images)", project_id, total)
    for idx, image_id in enumerate(image_ids, start=1):
        with database.SessionLocal() as db:
            img = db.get(Image, image_id)
            view_name = img.view_type if img else f"Image {image_id}"

        stage_msg = f"Processing image {idx}/{total}: {view_name} View..."
        with _project_jobs_lock:
            if project_id in _active_project_jobs:
                _active_project_jobs[project_id]["current_index"] = idx
                _active_project_jobs[project_id]["current_view"] = view_name
                _active_project_jobs[project_id]["current_stage"] = stage_msg

        log.info("[Project %s] %s", project_id, stage_msg)
        try:
            run_job(
                image_id,
                detail_level=detail_level,
                texture_suppression=texture_suppression,
                line_width=line_width,
            )
        except Exception:
            log.exception("[Project %s] Error processing image %s", project_id, image_id)
        finally:
            import gc
            gc.collect()

    log.info("[Project %s] Background processing finished", project_id)
    with _project_jobs_lock:
        if project_id in _active_project_jobs:
            _active_project_jobs[project_id]["current_stage"] = "Completed"
            _active_project_jobs[project_id]["is_complete"] = True


def start_project_processing_job(
    project_id: int,
    image_ids: list[int],
    detail_level: str | None = None,
    texture_suppression: bool | None = None,
    line_width: float | None = None,
    sync: bool = False,
) -> dict:
    """Launch processing for a list of image IDs. Runs in worker thread unless sync=True."""
    if sync:
        _run_project_processing_worker(
            project_id,
            image_ids,
            detail_level=detail_level,
            texture_suppression=texture_suppression,
            line_width=line_width,
        )
        return {"status": "completed", "queued": image_ids}

    worker = threading.Thread(
        target=_run_project_processing_worker,
        args=(project_id, image_ids),
        kwargs={
            "detail_level": detail_level,
            "texture_suppression": texture_suppression,
            "line_width": line_width,
        },
        daemon=True,
    )
    with _project_jobs_lock:
        _active_project_jobs[project_id] = {
            "thread": worker,
            "image_ids": image_ids,
            "total": len(image_ids),
            "current_index": 0,
            "current_view": None,
            "current_stage": "Initializing pipeline...",
            "is_complete": False,
        }
    worker.start()
    return {"status": "processing", "queued": image_ids}

