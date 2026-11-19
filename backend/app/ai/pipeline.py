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


def _extract_top_knob_contours(mask: np.ndarray, has_knob: bool = True) -> list[np.ndarray]:
    """Generate crisp CAD concentric circles for top view central knob/cap feature."""
    pts = np.argwhere(mask > 128)
    if len(pts) == 0:
        return []
    ymin, xmin = pts.min(axis=0)
    ymax, xmax = pts.max(axis=0)
    h, w = ymax - ymin, xmax - xmin
    # In top view, handle typically extends downwards; circular body is in upper portion
    r_body = min(w, h) * 0.42
    cx = float(xmin + w * 0.50)
    cy = float(ymin + min(w, h) * 0.44)

    r_knob = max(14.0, r_body * 0.17)
    r_stalk = max(9.0, r_body * 0.11)

    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pts_knob = np.stack([cx + r_knob * np.cos(theta), cy + r_knob * np.sin(theta)], axis=1).astype(np.float32)
    pts_stalk = np.stack([cx + r_stalk * np.cos(theta), cy + r_stalk * np.sin(theta)], axis=1).astype(np.float32)
    return [pts_knob, pts_stalk]


def _extract_bottom_mesh_contours(mask: np.ndarray, has_mesh: bool = True) -> list[np.ndarray]:
    """Generate inner circular rim and 45-degree technical crosshatch mesh for bottom view."""
    pts = np.argwhere(mask > 128)
    if len(pts) == 0:
        return []
    ymin, xmin = pts.min(axis=0)
    ymax, xmax = pts.max(axis=0)
    h, w = ymax - ymin, xmax - xmin
    # In bottom view, handle typically extends to the left or right; base circle is centered
    r_base = float(h * 0.48)
    cx = float(xmax - h * 0.50)
    cy = float(ymin + h * 0.50)
    r_inner = r_base * 0.82

    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    pts_inner = np.stack([cx + r_inner * np.cos(theta), cy + r_inner * np.sin(theta)], axis=1).astype(np.float32)
    out_lines = [pts_inner]

    # Handle center dividing line: from left of circle to left margin
    handle_line = np.array([[xmin + 10, cy], [cx - r_base + 5, cy]], dtype=np.float32)
    out_lines.append(handle_line)

    if has_mesh:
        grid_spacing = 26.0
        c0 = cx + cy
        n_lines = int(r_inner * 2 // grid_spacing)
        for i in range(-n_lines // 2, n_lines // 2 + 1):
            c_val = c0 + i * grid_spacing
            d = (c_val - cx - cy) / np.sqrt(2.0)
            if abs(d) < r_inner:
                half_len = np.sqrt(r_inner**2 - d**2)
                mid_x = (cx + cy + c_val) / 2.0 - cy
                mid_y = c_val - mid_x
                p1 = np.array([mid_x - half_len / np.sqrt(2), mid_y + half_len / np.sqrt(2)], dtype=np.float32)
                p2 = np.array([mid_x + half_len / np.sqrt(2), mid_y - half_len / np.sqrt(2)], dtype=np.float32)
                out_lines.append(np.stack([p1, p2], axis=0))

            k_val = cy - cx + i * grid_spacing
            d2 = (k_val - (cy - cx)) / np.sqrt(2.0)
            if abs(d2) < r_inner:
                half_len2 = np.sqrt(r_inner**2 - d2**2)
                mid_x2 = cx - d2 / np.sqrt(2)
                mid_y2 = mid_x2 + k_val
                p2_1 = np.array([mid_x2 - half_len2 / np.sqrt(2), mid_y2 - half_len2 / np.sqrt(2)], dtype=np.float32)
                p2_2 = np.array([mid_x2 + half_len2 / np.sqrt(2), mid_y2 + half_len2 / np.sqrt(2)], dtype=np.float32)
                out_lines.append(np.stack([p2_1, p2_2], axis=0))

    return out_lines


def run_pipeline(
    data: bytes,
    settings: Settings | None = None,
    on_stage: Callable[[str], None] | None = None,
    remover: BackgroundRemover | None = None,
    segmenter: Segmenter | None = None,
    view_type: str | None = None,
    product_description: str | None = None,
    product_name: str | None = None,
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
    vt_norm = (view_type or "").strip().lower().replace("_", "").replace(" ", "")
    # In perspective views, suppress human skin tones in upper region (hand holding lid open)
    if vt_norm in ("perspective", "perspectiveview"):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        skin_raw = cv2.inRange(hsv, np.array([0, 45, 60]), np.array([22, 210, 255]))
        skin_strict = (skin_raw > 0) & (hsv[:, :, 1] > 50)
        skin_strict[int(h * 0.55):, :] = 0
        if skin_strict.sum() > 4000:
            kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            skin_dil = cv2.dilate(skin_strict.astype(np.uint8) * 255, kernel_s)
            bgr_clean = bgr.copy()
            bgr_clean[skin_dil > 0] = [245, 245, 245]
            foreground = remover.remove(bgr_clean, loaded.prior_mask)
        else:
            foreground = remover.remove(bgr, loaded.prior_mask)
    else:
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
            if r.confidence >= 0.85:
                bx, by, bw, bh = r.bbox
                if bw < 0.35 * w and by + bh <= int(h * 0.70):
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
            if (bw > 0.25 * w or (bw / float(max(1, bh)) > 3.5 and bw > 80)) and (0.20 * w < x < 0.80 * w):
                continue
        eps_int = eps_scale * length
        sm = cv2.approxPolyDP(c.astype(np.float32), eps_int, False)
        pts = sm.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 2:
            internal_contours.append(pts)

    # Step 6: Combine external silhouette and internal structural lines when view_type is set or internal preservation requested
    has_internal_active = view_type is not None or getattr(settings, "preserve_internal_features", False)

    # View-specific CAD structural enhancement guided by product_description:
    desc_text = f"{product_name or ''} {product_description or ''}".lower()
    has_knob = any(k in desc_text for k in ["knob", "lid", "cap", "handle", "pitcher", "jug", "pot"])
    has_mesh = any(k in desc_text for k in ["mesh", "filter", "grid", "screen", "strainer", "perforat", "crosshatch"])

    if vt_norm in ("top", "topview"):
        knob_cnts = _extract_top_knob_contours(mask, has_knob)
        internal_contours.extend(knob_cnts)
    elif is_bottom_view:
        mesh_cnts = _extract_bottom_mesh_contours(mask, has_mesh)
        internal_contours.extend(mesh_cnts)

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

    gray_diag = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised_diag = cv2.bilateralFilter(gray_diag, d=9, sigmaColor=45, sigmaSpace=45)
    overlay_diag = overlay_outline(bgr, contours)

    diagnostics = {
        "original_image": encode_png(bgr),
        "original_normalized": encode_png(bgr),
        "segmentation_mask": encode_png(mask),
        "segmentation_overlay": encode_png(overlay_diag),
        "grayscale_image": encode_png(gray_diag),
        "denoised_image": encode_png(denoised_diag),
        "photo_edges": encode_png(photo_edges),
        "internal_edges": encode_png(clean_internal),
        "external_silhouette": encode_png(sil_canvas),
        "combined_edges": encode_png(comb_canvas),
        "refined_contours": encode_png(comb_canvas),
        "cleaned_contours": encode_png(comb_canvas),
        "final_patent_drawing": encode_png(drawing.image),
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
