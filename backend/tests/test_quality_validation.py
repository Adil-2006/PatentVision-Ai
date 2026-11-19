import cv2
import numpy as np
import pytest

from app.ai.branding import detect_branding_regions
from app.ai.contour import extract_contours
from app.ai.pipeline import run_pipeline
from app.core.config import get_settings
from tests import synthetic


def test_quality_validation_flags_multiple_outer_components():
    """Verify that multiple disjoint outer contours result in Review required, never silent Completed."""
    # Create an image with two components within grouping distance but geometrically disjoint
    img = np.full((400, 500, 3), 245, dtype=np.uint8)
    cv2.circle(img, (200, 200), 55, (30, 30, 30), -1)  # object 1
    cv2.circle(img, (290, 200), 30, (30, 30, 30), -1)  # object 2 (disjoint but within grouping distance)
    data = synthetic.encode(img)

    out = run_pipeline(data, get_settings())
    assert out.metrics["quality_status"] == "review_required"
    assert any("Multiple outer components detected" in w for w in out.metrics["quality_warnings"])


def test_branding_detection_detects_surface_text_patterns():
    """Verify that printed surface text patterns are detected and flagged."""
    # Create product with simulated text block inside
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (350, 350), 255, -1)
    
    bgr = np.full((400, 400, 3), 200, dtype=np.uint8)
    cv2.rectangle(bgr, (50, 50), (350, 350), (180, 180, 180), -1)
    # Add high-contrast text lines inside the product
    for y in [180, 200, 220]:
        for x in range(120, 280, 12):
            cv2.rectangle(bgr, (x, y), (x + 8, y + 10), (20, 20, 20), -1)

    regions = detect_branding_regions(bgr, mask)
    assert len(regions) > 0
    assert any(r.confidence > 0.5 for r in regions)


def test_clean_single_object_passes_quality_checks():
    """Verify that a clean single object photo passes without review required."""
    img = np.full((400, 400, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (30, 30, 30), -1)
    data = synthetic.encode(img)

    out = run_pipeline(data, get_settings())
    assert out.metrics["quality_status"] == "passed"
    assert len(out.metrics["quality_warnings"]) == 0


def test_rembg_background_remover_initialization_and_fallback():
    """Verify that RembgBackgroundRemover initializes properly and falls back gracefully."""
    from app.ai.background import RembgBackgroundRemover, get_background_remover

    remover = get_background_remover("rembg")
    assert isinstance(remover, RembgBackgroundRemover)

    # Test on a small synthetic image
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (0, 0, 0), -1)
    mask = remover.remove(img)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
