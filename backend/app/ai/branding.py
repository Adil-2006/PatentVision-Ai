"""Cautious branding, logo, label, and printed text detection and cleanup.

Removes logos, brand names, model numbers, stickers, and printed lettering
from final technical drawings while preserving genuine structural seams, joints,
latches, handles, knobs, and openings.
"""
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.ai.contour import ContourSet


@dataclass
class BrandingRegion:
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 3),
            "description": self.description,
        }


def detect_branding_regions(bgr: np.ndarray, mask: np.ndarray) -> list[BrandingRegion]:
    """Identify regions on the product surface likely to contain printed text, logos, or stickers."""
    if not mask.any():
        return []

    h, w = mask.shape
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Focus only on interior product pixels, eroded away from the outer silhouette edge
    # so we don't confuse the outer product boundary with surface text.
    kernel_margin = max(5, int(0.015 * max(h, w)))
    eroded_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_margin, kernel_margin)),
    )
    if not eroded_mask.any():
        return []

    # High-gradient text detection via morphological gradient (Sobel / black-hat)
    # Printed text typically exhibits sharp contrast and repetitive small stroke widths.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gradient in horizontal direction (text lines)
    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)
    grad_mag = cv2.bitwise_and(grad_mag, grad_mag, mask=eroded_mask)

    # Threshold high frequency edges inside product
    thresh = np.where(grad_mag > 35, 255, 0).astype(np.uint8)

    # Connect adjacent character strokes horizontally into text word blocks
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, w // 80), max(3, h // 200)))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
    detected: list[BrandingRegion] = []

    product_area = float(np.count_nonzero(mask))
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        # Text/logo criteria:
        # - Small compared to product (< 4% of product area)
        # - Plausible text glyph/line dimensions (height 8-70px, width 10-250px)
        # - Density of edge strokes within the bounding box is characteristic of text (0.15 - 0.70)
        box_area = bw * bh
        if box_area < 50 or box_area > 0.05 * product_area:
            continue
        if bh < 8 or bh > 80:
            continue
        if bw < 10 or bw > 0.70 * w:
            continue

        aspect = bw / float(bh)
        density = area / float(box_area)
        if 0.5 <= aspect <= 25.0 and 0.10 <= density <= 1.0:
            # Measure local edge density
            local_grad = grad_mag[y : y + bh, x : x + bw]
            high_edges = np.count_nonzero(local_grad > 35)
            edge_ratio = high_edges / float(box_area)
            if edge_ratio > 0.12:
                conf = float(np.clip(edge_ratio * 1.5, 0.4, 0.95))
                detected.append(
                    BrandingRegion(
                        bbox=(int(x), int(y), int(bw), int(bh)),
                        confidence=float(conf),
                        description=f"Surface text/logo candidate at ({int(x)},{int(y)})",
                    )
                )

    return detected


def clean_branding_and_text(
    contours: ContourSet,
    branding_regions: list[BrandingRegion],
    mask_shape: tuple[int, int],
) -> tuple[ContourSet, list[str]]:
    """Cautiously remove branding/text contours from holes/openings while preserving outer silhouette.

    Returns the cleaned ContourSet and any quality warning messages.
    """
    warnings: list[str] = []
    if not branding_regions:
        return contours, warnings

    # Identify any inner holes or openings that fall within a branding region
    filtered_holes = []
    removed_count = 0
    for h_pts in contours.holes:
        hx, hy, hw, hh = cv2.boundingRect(h_pts.astype(np.int32))
        hcx, hcy = hx + hw / 2.0, hy + hh / 2.0
        is_branding_hole = False
        for reg in branding_regions:
            rx, ry, rw, rh = reg.bbox
            if rx - 5 <= hcx <= rx + rw + 5 and ry - 5 <= hcy <= ry + rh + 5:
                is_branding_hole = True
                break
        if is_branding_hole:
            removed_count += 1
        else:
            filtered_holes.append(h_pts)

    if removed_count > 0:
        warnings.append(
            f"Suppressed {removed_count} surface branding/text loop(s) from technical drawing."
        )

    # If high-confidence branding regions were detected on the product
    high_conf = [r for r in branding_regions if r.confidence >= 0.70]
    if high_conf:
        warnings.append(
            f"Detected {len(high_conf)} printed marking/branding region(s) on product surface; verify geometry."
        )

    new_contours = ContourSet(outer=list(contours.outer), holes=filtered_holes)
    return new_contours, warnings
