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
from sqlalchemy import update
from sqlalchemy.orm import Session

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

        image.processing_status = job.job_status = result.status = "Completed"

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
