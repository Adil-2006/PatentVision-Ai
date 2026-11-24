"""End-to-end two-stage pipeline:
Stage 1 (Background Removal & Validation):
  load & detect existing transparency -> automatic background removal -> localization ->
  segmentation -> mask validation -> save background-removed RGBA cutout & preview.
Stage 2 (Patent Technical Drawing Generation from Isolated Product):
  mask-isolated detail-preserving edge extraction on original photo -> outer silhouette extraction ->
  internal structural contour extraction -> combine contours -> branding cleanup ->
  rendering (PNG + SVG) -> measurable quality & output validation.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.ai.background import (
    BackgroundRemover,
    ClassicalBackgroundRemover,
    RembgBackgroundRemover,
    cutout_preview_bgr,
    cutout_rgba,
    get_background_remover,
)
from app.ai.branding import clean_branding_and_text, detect_branding_regions
from app.ai.contour import ContourSet, extract_contours
from app.ai.errors import PipelineError
from app.ai.imaging import encode_png, load_image, scale_box
from app.ai.localization import gap_for, localize_product
from app.ai.mask_validation import assess_border_cropping, choose_mask
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


def _chaikin_smooth(pts: np.ndarray, iterations: int = 1, closed: bool = False) -> np.ndarray:
    """Corner-cutting algorithm to produce smooth, publication-grade technical curves without staircase artifacts."""
    if len(pts) < 3:
        return pts
    curr = pts.copy()
    for _ in range(iterations):
        res = []
        n = len(curr)
        end = n if closed else n - 1
        if not closed:
            res.append(curr[0])
        for i in range(end):
            p0 = curr[i]
            p1 = curr[(i + 1) % n]
            res.append(0.75 * p0 + 0.25 * p1)
            res.append(0.25 * p0 + 0.75 * p1)
        if not closed:
            res.append(curr[-1])
        curr = np.array(res, dtype=np.float32)
    return curr


def extract_photo_edges(
    bgr: np.ndarray,
    view_type: str | None = None,
    edge_low: float | None = None,
    edge_high: float | None = None,
    smoothing_strength: float | None = None,
    texture_suppression: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """Extract sharp patent-style edges:
    Multi-scale Top-Hat/Black-Hat feature extraction + Bilateral Filter + Canny + morphological healing."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vt_norm = (view_type or "").strip().lower().replace("_", "").replace(" ", "")
    is_bottom = vt_norm in ("bottom", "bottomview")

    # Detect high-frequency surface texture (woven fabric mesh, knurling, cloth grain)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hp = gray.astype(float) - cv2.GaussianBlur(gray, (5, 5), 1.0).astype(float)
    hp_energy = float(np.mean(np.abs(hp)))
    is_textured = bool(texture_suppression or (lap_var > 180.0 and hp_energy > 2.0))

    if is_textured:
        # Heavily textured surfaces (speaker mesh, fabric, knurling):
        # Multi-scale feature enhancement: Top-Hat + Black-Hat isolates embossed symbols/grooves
        s_val = float(smoothing_strength if smoothing_strength is not None else 35.0)
        low_t = float(edge_low if edge_low is not None else 25.0)
        high_t = float(edge_high if edge_high is not None else 70.0)
        blurred = cv2.bilateralFilter(gray, d=7, sigmaColor=s_val, sigmaSpace=s_val)
        k_hat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, k_hat)
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, k_hat)
        struct_enh = cv2.add(tophat, blackhat)
        enhanced_gray = cv2.addWeighted(blurred, 0.85, struct_enh, 1.15, 0)
        raw_edges = cv2.Canny(enhanced_gray, int(low_t), int(high_t), L2gradient=True)
        k_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        photo_edges = cv2.morphologyEx(raw_edges, cv2.MORPH_CLOSE, k_heal)
        edges_weak = cv2.Canny(enhanced_gray, int(max(8.0, low_t * 0.6)), int(low_t), L2gradient=True)
    else:
        # Standard smooth product baseline (mug, plastic bottle, pitcher, tools):
        default_low = 35.0 if is_bottom else 25.0
        default_high = 90.0 if is_bottom else 75.0
        default_sigma = 60.0 if is_bottom else 45.0
        default_clahe = 1.5 if is_bottom else 2.0
        s_val = float(smoothing_strength if smoothing_strength is not None else default_sigma)
        low_t = float(edge_low if edge_low is not None else default_low)
        high_t = float(edge_high if edge_high is not None else default_high)
        clahe = cv2.createCLAHE(clipLimit=default_clahe, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.bilateralFilter(enhanced, d=7, sigmaColor=s_val, sigmaSpace=s_val)
        raw_edges = cv2.Canny(blurred, int(low_t), int(high_t), L2gradient=True)
        k_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        photo_edges = cv2.morphologyEx(raw_edges, cv2.MORPH_CLOSE, k_heal)
        edges_weak = cv2.Canny(blurred, int(max(8.0, low_t * 0.6)), int(low_t), L2gradient=True)

    return photo_edges, gray, blurred, edges_weak, low_t, high_t, is_textured


def fill_enclosed_cavities(mask: np.ndarray) -> np.ndarray:
    """Fill fully enclosed background cavities inside the product (such as transparent bodies, inner liquid chambers)."""
    h, w = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    n_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inv)
    solid = mask.copy()
    for i in range(1, n_inv):
        x, y, bw, bh, area = stats_inv[i]
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
    product_description: str | None = None,
    product_name: str | None = None,
    detail_level: str | None = None,
    edge_low: float | None = None,
    edge_high: float | None = None,
    smoothing_strength: float | None = None,
    min_contour_length: float | None = None,
    texture_suppression: bool | None = None,
    line_width: float | None = None,
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
    vt_norm = (view_type or "").strip().lower().replace("_", "").replace(" ", "")

    # STAGE 1: Mandatory Automatic Background Removal (or Transparency Validation) BEFORE Edge/Drawing Extraction
    stage("background_removal")
    foreground = remover.remove(bgr, loaded.prior_mask)
    bg_method = getattr(remover, "last_method", "prior_alpha" if loaded.has_transparency else "classical")
    bg_fallback = bool(getattr(remover, "fallback_used", False))

    stage("localization")
    loc = localize_product(foreground)
    if loc.foreground.shape != bgr.shape[:2]:
        raise PipelineError("localization", "Detector and segmenter coordinate spaces do not match.")

    stage("segmentation")
    if loaded.has_transparency:
        fg_name = "prior-alpha"
    elif bg_method == "rembg":
        fg_name = "rembg"
    else:
        fg_name = "foreground-mask"
    candidates = segmenter.segment(bgr, loc.box, loc.foreground, view_type=view_type, foreground_name=fg_name)

    stage("mask_validation")
    raw_mask, chosen, reports = choose_mask(
        candidates, loc.foreground, gap_for((h, w)), settings.keep_openings
    )
    if not settings.keep_openings:
        mask = fill_enclosed_cavities(raw_mask)
    else:
        mask = raw_mask

    # Morphological mask cleanup to bridge specular chamfers and eliminate small notches
    k_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_mask)

    # Bottom text banner suppression (for cropped mobile UI screenshots)
    bot_y = int(h * 0.84)
    row_sum = (mask_clean[bot_y:, :] > 0).sum(axis=1)
    for r_idx in range(len(row_sum) - 1):
        if row_sum[r_idx] == 0 and row_sum[r_idx + 1] > 0:
            mask_clean[bot_y + r_idx:, :] = 0
            break

    # Build validated Stage 1 outputs (background-removed RGBA cutout & white-matte preview)
    cutout_rgba_img = cutout_rgba(bgr, mask_clean)
    cutout_preview_img = cutout_preview_bgr(bgr, mask_clean)
    cutout_rgba_bytes = encode_png(cutout_rgba_img)
    cutout_preview_bytes = encode_png(cutout_preview_img)

    # STAGE 2: Detail-Preserving Edge & Structural Contour Extraction from Original Photo (Mask-Restricted)
    stage("contour_extraction")
    # Isolate original photo pixels inside the validated foreground mask with rim-matched exterior
    # so room walls/furniture outside the mask never bleed into CLAHE or bilateral smoothing.
    eroded_rim = cv2.erode(mask_clean, np.ones((3, 3), np.uint8), iterations=1)
    rim_pixels = (mask_clean > 0) & (eroded_rim == 0)
    fill_color = (
        np.median(bgr[rim_pixels], axis=0).astype(np.uint8)
        if rim_pixels.any()
        else np.array([255, 255, 255], np.uint8)
    )
    bgr_isolated = np.empty_like(bgr)
    bgr_isolated[:, :] = fill_color
    keep_region = cv2.dilate(mask_clean, np.ones((3, 3), np.uint8), iterations=1) > 0
    bgr_isolated[keep_region] = bgr[keep_region]

    photo_edges, gray, smoothed, edges_weak, low_t, high_t, is_textured = extract_photo_edges(
        bgr_isolated,
        view_type=view_type,
        edge_low=edge_low,
        edge_high=edge_high,
        smoothing_strength=smoothing_strength,
        texture_suppression=texture_suppression,
    )

    branding_regions = detect_branding_regions(bgr, mask_clean)

    # Restrict photo edges strictly inside the interior of the validated product mask
    erode_iters = max(2, round(min(h, w) / 150)) if min(h, w) >= 600 else 2
    eroded_mask = cv2.erode(mask_clean, np.ones((3, 3), np.uint8), iterations=erode_iters)
    internal_edges = cv2.bitwise_and(photo_edges, photo_edges, mask=eroded_mask)

    is_bottom_view = vt_norm in ("bottom", "bottomview")
    min_area = 8 if (is_bottom_view or is_textured) else 4

    # Filter small noise specks (< 4px, or 8px for bottom / textured)
    n_ie, labels_ie, stats_ie, _ = cv2.connectedComponentsWithStats(internal_edges, connectivity=8)
    clean_internal = np.zeros_like(internal_edges)
    for i in range(1, n_ie):
        if stats_ie[i, cv2.CC_STAT_AREA] >= min_area:
            clean_internal[labels_ie == i] = 255

    # External product silhouette extracted separately from validated Stage 1 mask
    silhouette_contours = extract_contours(mask_clean, keep_openings=settings.keep_openings)
    smoothed_outer: list[np.ndarray] = []
    for oc in silhouette_contours.outer:
        length = float(cv2.arcLength(oc.astype(np.float32), True))
        eps_out = 0.0008 * length
        sm = cv2.approxPolyDP(oc.astype(np.float32), eps_out, True)
        smoothed = _chaikin_smooth(sm.reshape(-1, 2).astype(np.float32), iterations=1, closed=True)
        smoothed_outer.append(smoothed)

    # Extract internal structural contours directly from photographic edges
    raw_int_cnts, _ = cv2.findContours(clean_internal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    internal_contours: list[np.ndarray] = []
    default_min_len = 20.0 if is_bottom_view else (8.0 if is_textured else 5.0)
    min_cnt_len = float(min_contour_length if min_contour_length is not None else default_min_len)
    min_cnt_pts = 4 if (is_bottom_view or is_textured) else 3
    eps_scale = 0.0010 if is_bottom_view else 0.0008

    suppress_tex = bool(texture_suppression or getattr(settings, "texture_suppression", False) or is_textured)

    for c in raw_int_cnts:
        length = float(cv2.arcLength(c, False))
        if len(c) < min_cnt_pts or length < 5.0:
            continue

        x_c, y_c, bw_c, bh_c = cv2.boundingRect(c)
        dist_ends = float(np.linalg.norm(c[0][0] - c[-1][0]))

        # Topological loop closure for buttons, screws, port openings, and rings
        is_loop = (dist_ends <= 6.0 and len(c) >= 5 and bw_c >= 3 and bh_c >= 3)
        is_long_line = (length >= max(min_cnt_len, 10.0) and (bw_c >= 5 or bh_c >= 5))
        is_symbol_stroke = (suppress_tex and length >= 6.0 and bw_c >= 3 and bh_c >= 3 and bw_c <= 0.30 * w and bh_c <= 0.30 * h)

        if suppress_tex:
            diag = float(np.sqrt(bw_c**2 + bh_c**2))
            is_wandering = (not is_loop) and (length > 3.0 * max(diag, 1.0))
            if is_wandering:
                continue
            if not (is_loop or is_long_line or is_symbol_stroke):
                continue
        else:
            if not is_loop and length < min_cnt_len:
                continue

        eps_int = eps_scale * length
        if is_loop:
            c_closed = np.vstack([c, c[0:1]])
            sm = cv2.approxPolyDP(c_closed.astype(np.float32), eps_int, True)
            smoothed_c = _chaikin_smooth(sm.reshape(-1, 2).astype(np.float32), iterations=1, closed=True)
        else:
            sm = cv2.approxPolyDP(c.astype(np.float32), eps_int, False)
            smoothed_c = _chaikin_smooth(sm.reshape(-1, 2).astype(np.float32), iterations=1, closed=False)

        pts = smoothed_c.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 2:
            internal_contours.append(pts)

    # Combine external silhouette and internal structural lines
    has_internal_active = view_type is not None or getattr(settings, "preserve_internal_features", False)

    contours = ContourSet(
        outer=smoothed_outer if smoothed_outer else silhouette_contours.outer,
        holes=silhouette_contours.holes,
        internal=internal_contours if has_internal_active else [],
    )

    stage("branding_cleanup")
    contours, branding_warnings = clean_branding_and_text(contours, branding_regions, (h, w))

    stage("rendering")
    default_lw = 4.8
    effective_line_width = float(line_width if line_width is not None else default_lw)
    drawing = render_drawing(
        contours, long_side=settings.drawing_long_side, line_width=effective_line_width
    )
    drawing_svg_bytes = render_svg(
        contours, long_side=settings.drawing_long_side, line_width=effective_line_width
    ).encode("utf-8")

    # Build diagnostic visualizations (including Stage 1 background-removed cutout & preview)
    sil_canvas = np.full((h, w), 255, np.uint8)
    for c in contours.outer + contours.holes:
        pts = np.round(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(sil_canvas, [pts], isClosed=True, color=0, thickness=2, lineType=cv2.LINE_AA)

    comb_canvas = sil_canvas.copy()
    for c in contours.internal:
        pts = np.round(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(comb_canvas, [pts], isClosed=False, color=0, thickness=1, lineType=cv2.LINE_AA)

    overlay_diag = overlay_outline(bgr, contours)

    diagnostics = {
        # Standard Diagnostic Stages
        "original_image": encode_png(bgr),
        "background_removed": cutout_rgba_bytes,
        "background_removed_preview": cutout_preview_bytes,
        "segmentation_mask": encode_png(mask_clean),
        "segmentation_overlay": encode_png(overlay_diag),
        "grayscale_image": encode_png(gray),
        "smoothed_image": encode_png(smoothed),
        "weak_edges": encode_png(edges_weak),
        "strong_edges": encode_png(photo_edges),
        "structural_edges": encode_png(clean_internal),
        "external_silhouette": encode_png(sil_canvas),
        "combined_edges": encode_png(comb_canvas),
        "patent_drawing": encode_png(drawing.image),
        # Backward-compatible aliases for test suites and frontend components
        "original_normalized": encode_png(bgr),
        "denoised_image": encode_png(smoothed),
        "photo_edges": encode_png(photo_edges),
        "internal_edges": encode_png(clean_internal),
        "refined_contours": encode_png(comb_canvas),
        "cleaned_contours": encode_png(comb_canvas),
        "final_patent_drawing": encode_png(drawing.image),
    }

    stage("output_validation")
    drawing_report = validate_drawing(drawing)
    view_warnings = validate_view_geometry(view_type, contours, (w, h))
    cropped_edges = assess_border_cropping(mask_clean)

    # Measure short disconnected fragments inside internal contours
    short_fragments = sum(1 for c in contours.internal if float(cv2.arcLength(c, False)) < 18.0)
    frag_ratio = round(short_fragments / max(1, len(contours.internal)), 4)

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
    if bg_fallback:
        quality_warnings.append("AI background removal was unavailable; fallback classical segmentation used.")
    if chosen == "foreground-mask" and len(reports) > 1 and bg_method != "rembg":
        quality_warnings.append("Primary segmentation failed; conservative foreground mask used.")
    if loaded.background_kind == "complex" and bg_method == "classical" and chosen != "center-grabcut" and not loaded.has_transparency:
        if not any("segmentation" in q.lower() for q in quality_warnings):
            quality_warnings.append("Complex room background isolated with classical segmentation; inspect boundary accuracy.")
    if len(contours.outer) > 1:
        quality_warnings.append(
            f"Multiple outer components detected ({len(contours.outer)}). User review required."
        )
    if short_fragments > 65 and frag_ratio > 0.70 and not is_textured:
        quality_warnings.append(
            f"Excessive fragmented internal noise detected ({short_fragments} short fragments); inspect detail quality."
        )
    if drawing_report.get("ink_fraction", 0.0) < 0.0012:
        quality_warnings.append("Generated drawing has very low contour completeness; inspect product outline.")

    quality_status = "review_required" if quality_warnings else "passed"

    quality_checks = {
        "input_kind": "transparent_cutout" if loaded.has_transparency else f"{loaded.background_kind}_background",
        "background_removal_method": bg_method,
        "foreground_area_ratio": round(float(np.count_nonzero(mask_clean)) / mask_clean.size, 4),
        "border_contact_ratio": round(chosen_report.border_contact, 4) if chosen_report else 0.0,
        "cropped_edges": cropped_edges,
        "contamination_ratio": round(chosen_report.contamination, 4) if chosen_report else 0.0,
        "outer_components": len(contours.outer),
        "openings": len(contours.holes),
        "fragmented_noise_count": short_fragments,
        "fragmented_noise_ratio": frag_ratio,
        "ink_fraction": drawing_report.get("ink_fraction"),
        "line_width_ratio": drawing_report.get("line_width_ratio"),
    }

    metrics = {
        "original_size": list(loaded.original_size),
        "processed_size": [w, h],
        "scale": round(loaded.scale, 5),
        "box": list(loc.box),
        "box_original": list(scale_box(loc.box, 1.0 / loaded.scale)),
        "segmenter": settings.segmenter,
        "background_remover_method": bg_method,
        "input_kind": quality_checks["input_kind"],
        "mask_candidate": chosen,
        "mask_area_ratio": quality_checks["foreground_area_ratio"],
        "fallback_used": (chosen == "foreground-mask" and len(reports) > 1) or bg_fallback,
        "candidates": [r.as_dict() for r in reports],
        "contours": len(contours.outer),
        "openings": len(contours.holes),
        "internal_lines": len(contours.internal),
        "contour_points": contours.point_count,
        "drawing": drawing_report,
        "branding_regions": [r.as_dict() for r in branding_regions],
        "quality_status": quality_status,
        "quality_warnings": quality_warnings,
        "quality_checks": quality_checks,
        "detail_level": detail_level or getattr(settings, "drawing_detail_level", "standard"),
        "texture_suppression": suppress_tex,
        "texture_detected": is_textured,
        "edge_thresholds": [low_t, high_t],
        "diagnostics": {k: True for k in diagnostics.keys()},
    }
    return PipelineOutput(
        background_removed=cutout_rgba_bytes,
        mask=encode_png(mask_clean),
        outline=encode_png(overlay_outline(bgr, contours)),
        drawing=encode_png(drawing.image),
        drawing_svg=drawing_svg_bytes,
        metrics=metrics,
        diagnostics=diagnostics,
    )
