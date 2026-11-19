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
    detected: list[BrandingRegion] = []

    # 1. Detect rectangular sticker / label contours on the solid product body
    edges = cv2.Canny(gray, 40, 120)
    cnts_e, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    product_area = float(np.count_nonzero(mask))
    for c in cnts_e:
        x, y, bw, bh = cv2.boundingRect(c)
        c_area = cv2.contourArea(c)
        box_area = bw * bh
        if box_area > 0.40 * product_area:
            continue
        # Sticker is a rectangular region in the mid/upper body (never in functional base/saucer)
        is_sticker = False
        if 1500 < c_area < 120000 and int(h * 0.20) < y < int(h * 0.60):
            ar = bw / float(bh)
            rect_ratio = c_area / float(max(1, bw * bh))
            body_coverage = float(np.mean(mask[y : y + bh, x : x + bw])) / 255.0
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if 0.35 <= ar <= 3.0 and rect_ratio > 0.45 and body_coverage > 0.70 and 4 <= len(approx) <= 8:
                is_sticker = True

        if is_sticker:
            # Verify that the interior actually contains printed content (edges/text)
            m_pad = max(6, int(min(bw, bh) * 0.10))
            inner = edges[y + m_pad : y + bh - m_pad, x + m_pad : x + bw - m_pad]
            if inner.size > 0 and float(np.count_nonzero(inner > 0)) / float(inner.size) < 0.035:
                continue
            pad = 16
            bx = max(0, x - pad)
            by = max(0, y - pad)
            bw_p = min(w - bx, bw + 2 * pad)
            bh_p = min(h - by, bh + 2 * pad)
            detected.append(
                BrandingRegion(
                    bbox=(int(bx), int(by), int(bw_p), int(bh_p)),
                    confidence=0.92,
                    description=f"Surface sticker/label candidate at ({int(bx)},{int(by)})",
                )
            )

    # 2. Focus on interior product pixels, eroded away from the outer silhouette edge
    kernel_margin = max(5, int(0.015 * max(h, w)))
    eroded_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_margin, kernel_margin)),
    )
    if not eroded_mask.any():
        return detected

    # High-gradient text detection via morphological gradient (Sobel / black-hat)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)
    grad_mag = cv2.bitwise_and(grad_mag, grad_mag, mask=eroded_mask)

    # Focus text search in upper/mid body (avoid base bristles/structures)
    grad_body = grad_mag.copy()
    grad_body[int(h * 0.72):, :] = 0
    thresh = np.where(grad_body > 35, 255, 0).astype(np.uint8)

    # Connect adjacent character strokes horizontally into text word blocks
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, w // 80), max(3, h // 200)))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
    product_area = float(np.count_nonzero(mask))
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        box_area = bw * bh
        if box_area < 50 or box_area > 0.10 * product_area:
            continue
        if bh < 8 or bh > 140:
            continue
        if bw < 10 or bw > 0.70 * w:
            continue

        aspect = bw / float(bh)
        density = area / float(box_area)
        if 0.4 <= aspect <= 25.0 and 0.10 <= density <= 1.0:
            local_grad = grad_mag[y : y + bh, x : x + bw]
            high_edges = np.count_nonzero(local_grad > 35)
            edge_ratio = high_edges / float(box_area)
            if edge_ratio > 0.20:
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

    h, w = mask_shape[:2]

    # Identify any inner holes or openings that fall within a branding region
    filtered_holes = []
    removed_count = 0
    for h_pts in contours.holes:
        hx, hy, hw, hh = cv2.boundingRect(h_pts.astype(np.int32))
        hcx, hcy = hx + hw / 2.0, hy + hh / 2.0
        is_branding_hole = False
        for reg in branding_regions:
            if reg.confidence < 0.70:
                continue
            rx, ry, rw, rh = reg.bbox
            if rx - 5 <= hcx <= rx + rw + 5 and ry - 5 <= hcy <= ry + rh + 5:
                is_branding_hole = True
                break
        if is_branding_hole:
            removed_count += 1
        else:
            filtered_holes.append(h_pts)

    # Filter internal lines that fall within a branding region while preserving structural joints
    filtered_internal = []
    for line_pts in contours.internal:
        lx, ly, lw, lh = cv2.boundingRect(line_pts.astype(np.int32))
        lcx, lcy = lx + lw / 2.0, ly + lh / 2.0
        # Continuous structural lines spanning across product or large features are never branding
        if lw > 0.20 * w or lh > 0.20 * h:
            filtered_internal.append(line_pts)
            continue

        is_branding = False
        for reg in branding_regions:
            if reg.confidence < 0.70:
                continue
            rx, ry, rw, rh = reg.bbox
            # If the line spans across most of the candidate box, it is a structural boundary
            if lw > 0.85 * rw or lh > 0.85 * rh:
                continue
            if rx - 2 <= lcx <= rx + rw + 2 and ry - 2 <= lcy <= ry + rh + 2:
                is_branding = True
                break
        if not is_branding:
            filtered_internal.append(line_pts)

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

    new_contours = ContourSet(
        outer=list(contours.outer),
        holes=filtered_holes,
        internal=filtered_internal,
    )
    return new_contours, warnings
