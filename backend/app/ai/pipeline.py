"""End-to-end pipeline:
load -> background removal -> localization -> segmentation -> mask validation ->
contour extraction/smoothing -> branding cleanup -> rendering -> output validation.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.ai.background import BackgroundRemover, ClassicalBackgroundRemover, RembgBackgroundRemover, cutout_rgba, get_background_remover
from app.ai.branding import clean_branding_and_text, detect_branding_regions
from app.ai.contour import ContourSet, extract_contours
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
    diagnostics: dict[str, bytes] = field(default_factory=dict)


def extract_photo_edges(bgr: np.ndarray, view_type: str | None = None) -> np.ndarray:
    """Extract sharp patent-style edges: CLAHE + Bilateral + Canny (L2 gradient)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vt_norm = (view_type or "").strip().lower().replace("_", "").replace(" ", "")
    is_bottom = vt_norm in ("bottom", "bottomview")
    
    if is_bottom:
        # For textured bottom bases: suppress high-frequency 3D-print micro-speckles while preserving concentric rings/grooves
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=60, sigmaSpace=60)
        return cv2.Canny(blurred, 35, 90, L2gradient=True)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=45, sigmaSpace=45)
        return cv2.Canny(blurred, 25, 75, L2gradient=True)


def fill_enclosed_cavities(mask: np.ndarray) -> np.ndarray:
    """Fill fully enclosed background cavities inside the product (such as transparent bodies, inner liquid chambers)."""
    h, w = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    n_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inv)
    solid = mask.copy()
    for i in range(1, n_inv):
        x, y, bw, bh, area = stats_inv[i]
        # Enclosed component does not touch the outer canvas borders
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            solid[labels_inv == i] = 255
    return solid


def validate_view_geometry(view_type: str | None, contours: ContourSet, shape: tuple[int, int]) -> list[str]:
    """View-specific geometry and orientation validation."""
    warnings: list[str] = []
    if not contours.outer or not view_type:
        return warnings
    all_pts = np.concatenate(contours.outer)
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    bw, bh = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    if bh <= 0 or bw <= 0:
        warnings.append("Extracted geometry has degenerate dimensions.")
        return warnings
    ratio = bw / bh
    vt = view_type.strip().lower().replace("_", "").replace(" ", "")
    if vt in ("top", "topview", "bottom", "bottomview"):
        if ratio < 0.40 or ratio > 2.5:
            warnings.append(f"{view_type} has unusual aspect ratio ({ratio:.2f}); verify capture orientation.")
    elif vt in ("front", "frontview", "rear", "rearview", "left", "leftside", "right", "rightside"):
        if ratio > 3.0:
            warnings.append(f"{view_type} has disproportionate width ({ratio:.2f}); verify boundary isolation.")
    return warnings


def run_pipeline(
    data: bytes,
    settings: Settings | None = None,
    on_stage: Callable[[str], None] | None = None,
    remover: BackgroundRemover | None = None,
    segmenter: Segmenter | None = None,
    view_type: str | None = None,
) -> PipelineOutput:
    settings = settings or get_settings()
    remover = remover or get_background_remover(settings.background_remover)
    segmenter = segmenter or get_segmenter(
        settings.segmenter, settings.sam_checkpoint, settings.sam_model_type
    )
    stage = on_stage or (lambda _name: None)

    stage("load")
    loaded = load_image(data, settings.processing_max_side)
    bgr = loaded.bgr
    w, h = loaded.size

    # Step 1: Extract genuine photographic edges from the original image
    photo_edges = extract_photo_edges(bgr, view_type=view_type)

    stage("background_removal")
    foreground = remover.remove(bgr, loaded.prior_mask)

    stage("localization")
    loc = localize_product(foreground)
    if loc.foreground.shape != bgr.shape[:2]:
        raise PipelineError("localization", "Detector and segmenter coordinate spaces do not match.")

    stage("segmentation")
    fg_name = "rembg" if isinstance(remover, RembgBackgroundRemover) else "foreground-mask"
    candidates = segmenter.segment(bgr, loc.box, loc.foreground, view_type=view_type, foreground_name=fg_name)

    stage("mask_validation")
    raw_mask, chosen, reports = choose_mask(
        candidates, loc.foreground, gap_for((h, w)), settings.keep_openings
    )
    # When keep_openings is False, fill internal cavities so transparent or white product parts remain solid
    if not settings.keep_openings:
        mask = fill_enclosed_cavities(raw_mask)
    else:
        mask = raw_mask

    # Step 3 & 4: Branding cleanup and mask-restricted internal edge extraction
    branding_regions = detect_branding_regions(bgr, mask)

    # Restrict photo edges to the interior of the product mask (Colab Cell 11)
    eroded_mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    internal_edges = cv2.bitwise_and(photo_edges, photo_edges, mask=eroded_mask)
    if branding_regions:
        for r in branding_regions:
            bx, by, bw, bh = r.bbox
            if by + bh <= int(h * 0.70):
                internal_edges[by : by + bh, bx : bx + bw] = 0

    vt_norm = (view_type or "").strip().lower().replace("_", "").replace(" ", "")
    is_bottom_view = vt_norm in ("bottom", "bottomview")
    min_area = 8 if is_bottom_view else 4

    # Filter small noise specks from internal edges
    n_ie, labels_ie, stats_ie, _ = cv2.connectedComponentsWithStats(internal_edges, connectivity=8)
    clean_internal = np.zeros_like(internal_edges)
    for i in range(1, n_ie):
        if stats_ie[i, cv2.CC_STAT_AREA] >= min_area:
            clean_internal[labels_ie == i] = 255

    stage("contour_extraction")
    # Step 5: External product silhouette extracted separately
    silhouette_contours = extract_contours(mask, keep_openings=settings.keep_openings)
    smoothed_outer: list[np.ndarray] = []
    for oc in silhouette_contours.outer:
        length = float(cv2.arcLength(oc.astype(np.float32), True))
        eps_out = 0.0010 * length
        sm = cv2.approxPolyDP(oc.astype(np.float32), eps_out, True)
        smoothed_outer.append(sm.reshape(-1, 2).astype(np.float32))

    # Extract internal structural contours with mid-body reflection arc suppression
    raw_int_cnts, _ = cv2.findContours(clean_internal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    internal_contours: list[np.ndarray] = []
    mid_y0, mid_y1 = int(h * 0.36), int(h * 0.66)
    is_side_view = vt_norm in (
        "front", "frontview", "rear", "rearview", "left", "leftview", "leftside", "right", "rightview", "rightside"
    )
    min_cnt_len = 35 if is_bottom_view else 5
    min_cnt_pts = 5 if is_bottom_view else 3
    eps_scale = 0.0010 if is_bottom_view else 0.0008

    for c in raw_int_cnts:
        length = float(cv2.arcLength(c, False))
        if length < min_cnt_len or len(c) < min_cnt_pts:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        # Suppress wide, horizontal reflection streaks in smooth mid-body section only on side views
        if is_side_view and y >= mid_y0 and y + bh <= mid_y1:
            if bw / float(max(1, bh)) > 2.5 or bw > 0.30 * w:
                continue
        eps_int = eps_scale * length
        sm = cv2.approxPolyDP(c.astype(np.float32), eps_int, False)
        pts = sm.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 2:
            internal_contours.append(pts)

    # Step 6: Combine external silhouette and internal structural lines when view_type is set or internal preservation requested
    has_internal_active = view_type is not None or getattr(settings, "preserve_internal_features", False)
    contours = ContourSet(
        outer=smoothed_outer if smoothed_outer else silhouette_contours.outer,
        holes=silhouette_contours.holes,
        internal=internal_contours if has_internal_active else [],
    )

    stage("branding_cleanup")
    contours, branding_warnings = clean_branding_and_text(contours, branding_regions, (h, w))

    stage("rendering")
    drawing = render_drawing(
        contours, long_side=settings.drawing_long_side, line_width=settings.drawing_line_width
    )
    drawing_svg_bytes = render_svg(
        contours, long_side=settings.drawing_long_side, line_width=settings.drawing_line_width
    ).encode("utf-8")

    # Step 7: Build diagnostic visualizations (Part 9)
    sil_canvas = np.full((h, w), 255, np.uint8)
    for c in contours.outer + contours.holes:
        pts = np.round(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(sil_canvas, [pts], isClosed=True, color=0, thickness=2, lineType=cv2.LINE_AA)

    comb_canvas = sil_canvas.copy()
    for c in internal_contours:
        pts = np.round(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(comb_canvas, [pts], isClosed=False, color=0, thickness=1, lineType=cv2.LINE_AA)

    diagnostics = {
        "original_normalized": encode_png(bgr),
        "segmentation_mask": encode_png(mask),
        "photo_edges": encode_png(photo_edges),
        "internal_edges": encode_png(clean_internal),
        "external_silhouette": encode_png(sil_canvas),
        "combined_edges": encode_png(comb_canvas),
        "cleaned_contours": encode_png(comb_canvas),
        "patent_drawing": encode_png(drawing.image),
    }

    stage("output_validation")
    drawing_report = validate_drawing(drawing)
    view_warnings = validate_view_geometry(view_type, contours, (w, h))

    # Drawing Quality Validation: Never mark contaminated or incomplete drawings as Completed
    quality_warnings: list[str] = list(branding_warnings) + list(view_warnings)
    chosen_report = next((r for r in reports if r.name == chosen), None)
    if chosen_report:
        if chosen != "center-grabcut" and chosen_report.contamination > 0.08:
            quality_warnings.append(
                f"Elevated background contamination detected ({chosen_report.contamination * 100:.1f}%)."
            )
        if chosen_report.border_contact > 0.05:
            quality_warnings.append(
                f"Product boundary touches image border ({chosen_report.border_contact * 100:.1f}%)."
            )
    if chosen == "foreground-mask" and len(reports) > 1 and not isinstance(remover, RembgBackgroundRemover):
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
        "internal_lines": len(contours.internal),
        "contour_points": contours.point_count,
        "drawing": drawing_report,
        "branding_regions": [r.as_dict() for r in branding_regions],
        "quality_status": quality_status,
        "quality_warnings": quality_warnings,
        "diagnostics": {k: True for k in diagnostics.keys()},
    }
    return PipelineOutput(
        background_removed=encode_png(cutout_rgba(bgr, foreground)),
        mask=encode_png(mask),
        outline=encode_png(overlay_outline(bgr, contours)),
        drawing=encode_png(drawing.image),
        drawing_svg=drawing_svg_bytes,
        metrics=metrics,
        diagnostics=diagnostics,
    )
