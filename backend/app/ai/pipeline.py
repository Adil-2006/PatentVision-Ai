"""End-to-end pipeline:
load -> background removal -> localization -> segmentation -> mask validation ->
contour extraction/smoothing -> drawing rendering -> output validation.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.ai.background import BackgroundRemover, ClassicalBackgroundRemover, cutout_rgba
from app.ai.branding import clean_branding_and_text, detect_branding_regions
from app.ai.contour import extract_contours
from app.ai.errors import PipelineError
from app.ai.imaging import encode_png, load_image, scale_box
from app.ai.localization import gap_for, localize_product
from app.ai.mask_validation import choose_mask
from app.ai.rendering import overlay_outline, render_drawing, render_svg, validate_drawing
from app.ai.segmentation import Segmenter, get_segmenter
from app.core.config import Settings, get_settings

STAGES = (
    "load",
    "background_removal",
    "localization",
    "segmentation",
    "mask_validation",
    "contour_extraction",
    "branding_cleanup",
    "rendering",
    "output_validation",
)


@dataclass
class PipelineOutput:
    background_removed: bytes  # RGBA PNG
    mask: bytes  # binary PNG, white product on black
    outline: bytes  # PNG: extracted outline over the source photo
    drawing: bytes  # PNG: white background, black outline
    drawing_svg: bytes  # SVG: the same drawing as vector paths
    metrics: dict


def run_pipeline(
    data: bytes,
    settings: Settings | None = None,
    on_stage: Callable[[str], None] | None = None,
    remover: BackgroundRemover | None = None,
    segmenter: Segmenter | None = None,
) -> PipelineOutput:
    settings = settings or get_settings()
    remover = remover or ClassicalBackgroundRemover()
    segmenter = segmenter or get_segmenter(
        settings.segmenter, settings.sam_checkpoint, settings.sam_model_type
    )
    stage = on_stage or (lambda _name: None)

    stage("load")
    loaded = load_image(data, settings.processing_max_side)
    bgr = loaded.bgr
    w, h = loaded.size

    stage("background_removal")
    foreground = remover.remove(bgr, loaded.prior_mask)

    stage("localization")
    loc = localize_product(foreground)
    if loc.foreground.shape != bgr.shape[:2]:
        raise PipelineError("localization", "Detector and segmenter coordinate spaces do not match.")

    stage("segmentation")
    candidates = segmenter.segment(bgr, loc.box, loc.foreground)

    stage("mask_validation")
    mask, chosen, reports = choose_mask(
        candidates, loc.foreground, gap_for((h, w)), settings.keep_openings
    )

    stage("contour_extraction")
    contours = extract_contours(mask, keep_openings=settings.keep_openings)

    stage("branding_cleanup")
    branding_regions = detect_branding_regions(bgr, mask)
    contours, branding_warnings = clean_branding_and_text(contours, branding_regions, (h, w))

    stage("rendering")
    drawing = render_drawing(
        contours, long_side=settings.drawing_long_side, line_width=settings.drawing_line_width
    )

    stage("output_validation")
    drawing_report = validate_drawing(drawing)

    # Drawing Quality Validation: Never mark contaminated or incomplete drawings as Completed
    quality_warnings: list[str] = list(branding_warnings)
    chosen_report = next((r for r in reports if r.name == chosen), None)
    if chosen_report:
        if chosen_report.contamination > 0.08:
            quality_warnings.append(
                f"Elevated background contamination detected ({chosen_report.contamination * 100:.1f}%)."
            )
        if chosen_report.border_contact > 0.05:
            quality_warnings.append(
                f"Product boundary touches image border ({chosen_report.border_contact * 100:.1f}%)."
            )
    if chosen == "foreground-mask" and len(reports) > 1:
        quality_warnings.append("Primary segmentation failed; conservative foreground mask used.")
    if len(contours.outer) > 1:
        quality_warnings.append(
            f"Multiple outer components detected ({len(contours.outer)}). User review required."
        )

    quality_status = "review_required" if quality_warnings else "passed"

    metrics = {
        "original_size": list(loaded.original_size),
        "processed_size": [w, h],
        "scale": round(loaded.scale, 5),
        "box": list(loc.box),
        "box_original": list(scale_box(loc.box, 1.0 / loaded.scale)),
        "segmenter": settings.segmenter,
        "mask_candidate": chosen,
        "mask_area_ratio": round(float(np.count_nonzero(mask)) / mask.size, 4),
        "fallback_used": chosen == "foreground-mask" and len(reports) > 1,
        "candidates": [r.as_dict() for r in reports],
        "contours": len(contours.outer),
        "openings": len(contours.holes),
        "contour_points": contours.point_count,
        "drawing": drawing_report,
        "branding_regions": [r.as_dict() for r in branding_regions],
        "quality_status": quality_status,
        "quality_warnings": quality_warnings,
    }
    return PipelineOutput(
        background_removed=encode_png(cutout_rgba(bgr, foreground)),
        mask=encode_png(mask),
        outline=encode_png(overlay_outline(bgr, contours)),
        drawing=encode_png(drawing.image),
        drawing_svg=render_svg(
            contours, long_side=settings.drawing_long_side, line_width=settings.drawing_line_width
        ).encode(),
        metrics=metrics,
    )
