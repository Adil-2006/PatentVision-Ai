"""Background removal. Replaceable behind the BackgroundRemover interface."""
import logging
import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.ai.errors import PipelineError
from app.ai.masks import ellipse, to_binary

log = logging.getLogger("patentvision.background")

L_WEIGHT = 0.6  # de-emphasise lightness so soft shadows are not treated as product
MIN_THRESHOLD, MAX_THRESHOLD = 14.0, 70.0

_SHARED_REMBG_LOCK = threading.Lock()
_SHARED_REMBG_SESSION = None
_SHARED_REMBG_MODEL: str | None = None


def _resolve_rembg_model() -> str:
    """Pick the configured rembg ONNX model, preferring u2netp when u2net is not pre-seeded."""
    import os
    from app.core.config import get_settings

    configured = (
        os.environ.get("PV_REMBG_MODEL")
        or os.environ.get("REMBG_MODEL")
        or get_settings().rembg_model
        or "auto"
    ).strip().lower()
    if configured and configured != "auto":
        return configured

    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    if (models_dir / "u2net.onnx").is_file():
        return "u2net"
    try:
        from rembg.sessions.u2net import U2netSession

        if U2netSession.resolve_existing("u2net.onnx") is not None:
            return "u2net"
    except Exception:
        pass
    return "u2netp"


def _shared_u2net_session(model_name: str | None = None):
    global _SHARED_REMBG_SESSION, _SHARED_REMBG_MODEL
    target = (model_name or _resolve_rembg_model()).strip().lower()
    with _SHARED_REMBG_LOCK:
        if _SHARED_REMBG_SESSION is None or _SHARED_REMBG_MODEL != target:
            import os
            import rembg

            models_dir = Path(__file__).resolve().parent.parent.parent / "models"
            if (models_dir / f"{target}.onnx").is_file():
                os.environ["U2NET_HOME"] = str(models_dir)
            _SHARED_REMBG_SESSION = rembg.new_session(target)
            _SHARED_REMBG_MODEL = target
        return _SHARED_REMBG_SESSION


class BackgroundRemover(Protocol):
    def remove(self, bgr: np.ndarray, prior_mask: np.ndarray | None = None) -> np.ndarray:
        """Return a 0/255 foreground mask the size of `bgr`."""


def background_distance(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel colour distance (Lab) from the dominant colours sampled around the image border."""
    h, w = bgr.shape[:2]
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    ring = max(4, int(0.03 * min(h, w)))
    ring_mask = np.ones((h, w), dtype=bool)
    ring_mask[ring : h - ring, ring : w - ring] = False
    samples = lab[ring_mask]
    if len(samples) > 20000:
        samples = samples[np.random.default_rng(0).choice(len(samples), 20000, replace=False)]

    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.ravel(), minlength=k)
    centers = centers[counts >= 0.08 * counts.sum()]
    if len(centers) == 0:
        centers = np.median(samples, axis=0, keepdims=True)

    weights = np.array([L_WEIGHT, 1.0, 1.0], dtype=np.float32)
    best = None
    for c in centers:
        d = np.sqrt((((lab - c) * weights) ** 2).sum(axis=2))
        best = d if best is None else np.minimum(best, d)
    return best


def otsu_threshold(dist: np.ndarray) -> float:
    d8 = np.clip(dist, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.clip(t, MIN_THRESHOLD, MAX_THRESHOLD))


def _low_contrast_white_bg_probe(bgr: np.ndarray, dist: np.ndarray) -> np.ndarray | None:
    """Rescue low-contrast products on plain/white backgrounds without triggering on blank images."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Require real interior structural gradient relative to an almost blank border
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 15, 45)
    m = max(6, int(0.06 * min(h, w)))
    interior_edges = edges[m : max(m + 1, h - m), m : max(m + 1, w - m)]
    if interior_edges.size == 0 or (interior_edges > 0).mean() < 0.003:
        return None
    ring = max(4, int(0.03 * min(h, w)))
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[ring : max(ring + 1, h - ring), ring : max(ring + 1, w - ring)] = False
    border_p95 = float(np.percentile(dist[border_mask], 95)) if border_mask.any() else 6.0
    low_thresh = float(np.clip(max(5.5, border_p95 + 1.5), 5.5, MIN_THRESHOLD))
    fg_dist = np.where(dist > low_thresh, 255, 0).astype(np.uint8)
    edge_band = cv2.dilate(edges, ellipse(max(5, min(h, w) // 80)))
    combined = cv2.bitwise_or(fg_dist, edge_band)
    combined[:3, :] = combined[-3:, :] = combined[:, :3] = combined[:, -3:] = 0
    # Flood from outside to fill the interior of closed low-contrast contours
    pad = cv2.copyMakeBorder(combined, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(pad, None, (0, 0), 128)
    filled = np.where(pad[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ellipse(3))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, ellipse(max(5, min(h, w) // 100)))
    area_ratio = float(np.count_nonzero(filled)) / max(1, filled.size)
    if 0.01 <= area_ratio <= 0.90:
        return filled
    return None


def _hysteresis_alpha_mask(alpha: np.ndarray) -> np.ndarray:
    """Binarize a soft alpha matte keeping translucent appendages connected to strong foreground core."""
    core = np.where(alpha >= 110, 255, 0).astype(np.uint8)
    weak = np.where(alpha >= 10, 255, 0).astype(np.uint8)
    if not core.any():
        core = np.where(alpha >= 30, 255, 0).astype(np.uint8)
    n_w, labels_w, _, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    mask = np.zeros_like(core)
    for i in range(1, n_w):
        comp = labels_w == i
        if np.any(core[comp]):
            mask[comp] = 255
    return mask if mask.any() else weak


class ClassicalBackgroundRemover:
    """Border-colour model + Otsu threshold with white-bg contrast probe. No model files required."""

    def __init__(self) -> None:
        self.last_method = "classical"
        self.fallback_used = False

    def remove(self, bgr: np.ndarray, prior_mask: np.ndarray | None = None) -> np.ndarray:
        self.fallback_used = False
        if prior_mask is not None:
            self.last_method = "prior_alpha"
            mask = to_binary(prior_mask)
        else:
            self.last_method = "classical"
            dist = background_distance(bgr)
            mask = np.where(dist > otsu_threshold(dist), 255, 0).astype(np.uint8)
            side = min(mask.shape)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(max(3, side // 150)))
            if np.count_nonzero(mask) < 0.004 * mask.size:
                rescued = _low_contrast_white_bg_probe(bgr, dist)
                if rescued is not None:
                    mask = rescued
        if not mask.any():
            raise PipelineError(
                "background_removal",
                "No foreground could be separated from the background.",
                "Use a photo where the product contrasts with its background.",
            )
        return mask


class RembgBackgroundRemover:
    """U-2-Net based AI background removal with transparent cutout preservation and classical fallback."""

    def __init__(self, session=None):
        self._session = session
        self._lock = threading.Lock()
        self.last_method = "rembg"
        self.fallback_used = False

    def _get_session(self):
        with self._lock:
            if self._session is None:
                self._session = _shared_u2net_session()
            return self._session

    def remove(self, bgr: np.ndarray, prior_mask: np.ndarray | None = None) -> np.ndarray:
        self.fallback_used = False
        if prior_mask is not None:
            self.last_method = "prior_alpha"
            return to_binary(prior_mask)
        try:
            import rembg

            session = self._get_session()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            output = rembg.remove(rgb, session=session)
            alpha = output[:, :, 3]
            binary_mask = _hysteresis_alpha_mask(alpha)

            # Keep primary components and drop small stray noise blobs (< 5% of primary)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            if num_labels > 2:
                max_area = float(stats[1:, cv2.CC_STAT_AREA].max())
                keep = [
                    i
                    for i in range(1, num_labels)
                    if stats[i, cv2.CC_STAT_AREA] >= 0.05 * max_area
                ]
                clean_m = np.zeros_like(binary_mask)
                clean_m[np.isin(labels, keep)] = 255
                binary_mask = clean_m

            if not binary_mask.any():
                raise PipelineError("background_removal", "No foreground detected by rembg.")
            self.last_method = "rembg"
            return binary_mask
        except Exception as exc:
            log.warning("rembg background removal failed (%s); falling back to classical remover", exc)
            fallback = ClassicalBackgroundRemover()
            mask = fallback.remove(bgr, prior_mask)
            self.last_method = "classical_fallback"
            self.fallback_used = True
            return mask


def get_background_remover(kind: str = "auto") -> BackgroundRemover:
    if kind == "classical":
        return ClassicalBackgroundRemover()
    if kind == "rembg":
        return RembgBackgroundRemover()

    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    try:
        import onnxruntime  # noqa: F401
        import rembg  # noqa: F401

        if (models_dir / "u2net.onnx").is_file() or (models_dir / "u2netp.onnx").is_file():
            return RembgBackgroundRemover()
        return RembgBackgroundRemover()
    except ImportError:
        pass
    return ClassicalBackgroundRemover()


def cutout_rgba(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Background-removed image: product pixels with the mask as alpha (transparent background)."""
    b, g, r = cv2.split(bgr)
    return cv2.merge([b, g, r, mask])


def cutout_preview_bgr(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Background-removed preview composited onto pure white (#ffffff) background."""
    preview = np.full_like(bgr, 255)
    fg = mask > 0
    preview[fg] = bgr[fg]
    return preview
