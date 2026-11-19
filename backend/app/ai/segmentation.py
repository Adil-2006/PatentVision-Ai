"""Segmentation backends. Each returns candidate masks in preference order; validation picks one."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.ai.errors import PipelineError
from app.ai.imaging import Box
from app.ai.masks import ellipse, to_binary

GRABCUT_MAX_SIDE = 640


@dataclass
class Candidate:
    name: str
    mask: np.ndarray  # 0/255, same size as the image


class Segmenter(Protocol):
    def segment(
        self,
        bgr: np.ndarray,
        box: Box,
        foreground: np.ndarray,
        view_type: str | None = None,
        foreground_name: str = "foreground-mask",
    ) -> list[Candidate]: ...


STANDARD_7_VIEWS = {
    "perspective", "perspectiveview",
    "front", "frontview",
    "rear", "rearview", "back", "backview",
    "right", "rightview", "rightside", "rightsideview",
    "left", "leftview", "leftside", "leftsideview",
    "top", "topview",
    "bottom", "bottomview",
}


def is_standard_view(view_type: str | None) -> bool:
    if not view_type:
        return False
    norm = view_type.strip().lower().replace("_", "").replace(" ", "")
    return norm in STANDARD_7_VIEWS


def _center_prior_grabcut(bgr: np.ndarray, view_type: str | None = None) -> np.ndarray | None:
    """Center-prior GrabCut with guaranteed background margin isolation and solid product filling."""
    h, w = bgr.shape[:2]
    m = max(10, int(0.04 * min(h, w)))
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc_mask[:m, :] = cv2.GC_BGD
    gc_mask[-m:, :] = cv2.GC_BGD
    gc_mask[:, :m] = cv2.GC_BGD
    gc_mask[:, -m:] = cv2.GC_BGD

    vt = (view_type or "").strip().lower().replace("_", "").replace(" ", "")
    is_circular = "top" in vt or "bottom" in vt
    if is_circular:
        gc_mask[: int(h * 0.10), :] = cv2.GC_BGD
        gc_mask[int(h * 0.90) :, :] = cv2.GC_BGD
        rx, ry, rw, rh = int(w * 0.12), int(h * 0.15), int(w * 0.76), int(h * 0.70)
    elif "perspective" in vt:
        gc_mask[: int(h * 0.12), :] = cv2.GC_BGD
        rx, ry, rw, rh = int(w * 0.08), int(h * 0.10), int(w * 0.84), int(h * 0.80)
    else:
        # Front, Rear, Right, Left: upper 18% is background room wall/ceiling
        gc_mask[: int(h * 0.18), :] = cv2.GC_BGD
        rx, ry, rw, rh = int(w * 0.08), int(h * 0.18), int(w * 0.84), int(h * 0.72)

    gc_mask[ry : ry + rh, rx : rx + rw] = cv2.GC_PR_FGD
    crx, cry, crw, crh = rx + int(rw * 0.20), ry + int(rh * 0.20), int(rw * 0.60), int(rh * 0.60)
    gc_mask[cry : cry + crh, crx : crx + crw] = cv2.GC_FGD

    scale = 0.5
    small_bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
    small_mask = cv2.resize(gc_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_bgr, small_mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg = np.where((small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)

    # Vertical close to bridge transparent/clear chambers (e.g. cup cylinder) between rim and base
    if not is_circular:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 65)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    # Keep only the largest single product component
    n_comp, labels_comp, stats_comp, _ = cv2.connectedComponentsWithStats(fg)
    if n_comp > 1:
        areas = [stats_comp[i, cv2.CC_STAT_AREA] for i in range(1, n_comp)]
        max_idx = 1 + np.argmax(areas)
        fg = np.where(labels_comp == max_idx, 255, 0).astype(np.uint8)

    return fg


def _grabcut(bgr: np.ndarray, box: Box, foreground: np.ndarray) -> np.ndarray | None:
    h, w = foreground.shape
    s = min(1.0, GRABCUT_MAX_SIDE / max(h, w))
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    img = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    fg = cv2.resize(foreground, (sw, sh), interpolation=cv2.INTER_NEAREST) if s < 1 else foreground

    k = max(3, sw // 80)
    sure_fg = cv2.erode(fg, ellipse(k))
    maybe_bg = cv2.dilate(fg, ellipse(2 * k))
    gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    gc[maybe_bg > 0] = cv2.GC_PR_BGD
    gc[fg > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD

    # Canvas border margins are definite background
    m = max(2, int(0.03 * min(sh, sw)))
    gc[:m, :] = cv2.GC_BGD
    gc[-m:, :] = cv2.GC_BGD
    gc[:, :m] = cv2.GC_BGD
    gc[:, -m:] = cv2.GC_BGD

    if not (gc == cv2.GC_FGD).any():
        ys, xs = np.nonzero(fg)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            rad = max(2, min(sw, sh) // 30)
            cv2.circle(gc, (cx, cy), rad, int(cv2.GC_FGD), -1)

    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return None

    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if s < 1:
        out = to_binary(cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR))
    x0, y0, x1, y1 = box
    clipped = np.zeros_like(out)
    clipped[y0:y1, x0:x1] = out[y0:y1, x0:x1]
    return clipped


class ClassicalSegmenter:
    """GrabCut refinement of the foreground mask, with the foreground mask itself as the fallback."""

    def segment(
        self,
        bgr: np.ndarray,
        box: Box,
        foreground: np.ndarray,
        view_type: str | None = None,
        foreground_name: str = "foreground-mask",
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        if foreground_name == "rembg":
            candidates.append(Candidate("rembg", foreground))

        refined = _grabcut(bgr, box, foreground)
        if refined is not None:
            candidates.append(Candidate("grabcut", refined))

        if view_type:
            center_cand = _center_prior_grabcut(bgr, view_type)
            if center_cand is not None:
                candidates.append(Candidate("center-grabcut", center_cand))

        if foreground_name != "rembg":
            candidates.append(Candidate("foreground-mask", foreground))
        return candidates


class SamSegmenter:
    """Segment Anything, prompted with the localized product box. Needs torch + a checkpoint."""

    def __init__(self, checkpoint: str | None, model_type: str = "vit_b") -> None:
        self.checkpoint = checkpoint
        self.model_type = model_type
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint or not Path(self.checkpoint).is_file():
            raise PipelineError(
                "segmentation",
                "The SAM model checkpoint is missing.",
                "Set PV_SAM_CHECKPOINT to a valid checkpoint file or use PV_SEGMENTER=classical.",
            )
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise PipelineError(
                "segmentation",
                "The segment-anything package is not installed.",
                "Install torch and segment-anything or use PV_SEGMENTER=classical.",
            ) from exc
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self._predictor = SamPredictor(sam)
        return self._predictor

    def segment(
        self,
        bgr: np.ndarray,
        box: Box,
        foreground: np.ndarray,
        view_type: str | None = None,
        foreground_name: str = "foreground-mask",
        **kwargs,
    ) -> list[Candidate]:
        predictor = self._load()
        predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=True)
        order = np.argsort(-scores)
        return [
            Candidate(f"sam-{i}", np.where(masks[i], 255, 0).astype(np.uint8)) for i in order
        ] + [Candidate("foreground-mask", foreground)]


def get_segmenter(kind: str, sam_checkpoint: str | None = None, sam_model_type: str = "vit_b") -> Segmenter:
    if kind == "classical":
        return ClassicalSegmenter()
    if kind == "sam":
        return SamSegmenter(sam_checkpoint, sam_model_type)
    raise ValueError(f"Unknown segmenter '{kind}'. Use 'classical' or 'sam'.")
