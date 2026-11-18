"""Holder detection: plate segmentation, sub-pixel edges, chamfer, rigid rectangle fit."""
from typing import NamedTuple

import cv2
import numpy as np

from . import constants as C
from .geometry import (holder_to_px, line_intersect, normalize, robust_line,
                       sample_profiles, similarity_fit, subpixel_peak)


class HolderPose(NamedTuple):
    H: np.ndarray      # 3x3, holder (Y, X) mm -> px
    img: np.ndarray    # normalized, smoothed image used by the later stages
    ppm: float         # holder px/mm


def find_holder_coarse(im, s=C.HOLDER_COARSE_DOWNSAMPLE):
    """Plate as the bright component best matching the 120x70 rectangle, over a threshold
    sweep on a downsampled image. Returns its minAreaRect corners in full-res px or None."""
    sm = cv2.resize(im, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(normalize(sm), (0, 0), 1.5)
    H, W = sm.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best = None
    for t in np.linspace(0.05, 0.9, 35):
        m = cv2.morphologyEx((sm > t).astype(np.uint8), cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                continue                                   # touches the image border
            area = cv2.contourArea(c)
            if area < C.HOLDER_MIN_AREA_FRAC * H * W:
                continue
            _, (rw, rh), _ = cv2.minAreaRect(c)
            fill = area / (rw * rh)
            aspect = max(rw, rh) / min(rw, rh)
            score = fill - 3 * abs(aspect - C.HOLDER_ASPECT) / C.HOLDER_ASPECT
            if fill > C.HOLDER_MIN_FILL and abs(aspect / C.HOLDER_ASPECT - 1) < C.HOLDER_ASPECT_TOL:
                if best is None or score > best[0]:
                    best = (score, c)
    if best is None:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(best[1])) * s + (s - 1) / 2
    return box.astype(np.float64)


def refine_rect(img, box, search, t_range=(0.06, 0.94), n=150, mode='step', max_tilt_deg=None):
    """Refine each side of a quadrilateral from 1-D profiles along the side normals.

    mode 'step'  : dark outside -> bright inside edge (max inward gradient)
    mode 'valley': thin dark line (groove)
    max_tilt_deg : a side tilted more than this vs. the input keeps the input direction
                   and is only shifted.
    Returns (corners, per-side quality) or (None, None)."""
    c = box.mean(0)
    lines, quals = [], []
    offs = np.arange(-search, search + 0.5, 0.5)
    for i in range(4):
        p0, p1 = box[i], box[(i + 1) % 4]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([-d[1], d[0]])
        if (c - (p0 + p1) / 2) @ nrm < 0:
            nrm = -nrm                                   # point inward
        ts = np.linspace(*t_range, n)
        pts = p0 + ts[:, None] * (p1 - p0)
        prof = sample_profiles(img, pts, nrm, offs)
        if mode == 'step':
            g = np.gradient(prof, axis=1)
        else:
            w = 8                                         # half-width in samples (4 px)
            g = np.zeros_like(prof)
            g[:, w:-w] = 0.5 * (prof[:, :-2 * w] + prof[:, 2 * w:]) - prof[:, w:-w]
        jj = np.clip(np.argmax(g, 1), 1, len(offs) - 2)
        sub, peak = subpixel_peak(g, jj)
        off = offs[jj] + 0.5 * sub
        strong = peak > 0.3 * np.median(peak) + 1e-6
        edge_pts = pts[strong] + off[strong, None] * nrm
        if len(edge_pts) < 10:
            return None, None
        ln, q = robust_line(edge_pts)
        if max_tilt_deg is not None and abs(ln[1] @ nrm) > np.sin(np.radians(max_tilt_deg)):
            sh = np.median((edge_pts - p0) @ nrm)
            ln, q = (p0 + sh * nrm, d), 0.0
        lines.append(ln); quals.append(q * strong.mean())
    corners = np.array([line_intersect(lines[i - 1], lines[i]) for i in range(4)])
    return corners, quals


def _chamfer_index(img, corners):
    """Index of the corner whose probe triangle is darkest relative to the plate, and the
    contrast to the second-darkest corner."""
    plate = np.median(img[cv2.fillConvexPoly(np.zeros(img.shape, np.uint8),
                                             corners.astype(np.int32), 1) > 0])
    long_px = max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[1] - corners[2]))
    leg = C.CHAMFER_PROBE_MM * long_px / C.HOLDER_Y
    vals = []
    for i in range(4):
        p = corners[i]
        a = p + leg * (corners[i - 1] - p) / np.linalg.norm(corners[i - 1] - p)
        b = p + leg * (corners[(i + 1) % 4] - p) / np.linalg.norm(corners[(i + 1) % 4] - p)
        m = cv2.fillConvexPoly(np.zeros(img.shape, np.uint8),
                               np.array([p, a, b]).astype(np.int32), 1) > 0
        vals.append(img[m].mean() / max(plate, 1e-6))
    order = np.argsort(vals)
    return order[0], float(vals[order[1]] - vals[order[0]])


def find_holder(im, res):
    """Detect the holder and fill the holder fields of ``res``. Returns HolderPose or None."""
    box = find_holder_coarse(im)
    if box is None:
        res.reasons.append('holder: no plate-shaped component')
        return None
    img = cv2.GaussianBlur(normalize(im), (0, 0), 2)
    corners, _ = refine_rect(img, box, search=C.HOLDER_EDGE_SEARCH_PX)
    if corners is None:
        res.reasons.append('holder: edge refinement failed')
        return None
    ci, res.chamfer_contrast = _chamfer_index(img, corners)
    if res.chamfer_contrast < C.CHAMFER_MIN_CONTRAST:
        res.reasons.append(f'holder: chamfer ambiguous (contrast {res.chamfer_contrast:.2f})')
    # origin = neighbour of the chamfer along the SHORT side
    nb = [(ci - 1) % 4, (ci + 1) % 4]
    o = min(nb, key=lambda k: np.linalg.norm(corners[k] - corners[ci]))
    # O, C (+X end), opposite of O (+X+Y), opposite of C (+Y end)
    ordered = np.array([corners[o], corners[ci], corners[(o + 2) % 4], corners[(ci + 2) % 4]])
    # rigid rectangle (rotation + uniform scale + translation) through the refined corners;
    # everything derived from the holder (window, axes) is then exactly square to it
    mm = np.array([[0, 0], [C.HOLDER_X, 0], [C.HOLDER_X, C.HOLDER_Y], [0, C.HOLDER_Y]])
    Rs, ts = similarity_fit(mm[:, ::-1], ordered)        # (Y,X) mm -> px keeps handedness
    H = np.vstack([np.c_[Rs, ts], [0, 0, 1]])
    fitted = holder_to_px(H, mm)
    res.holder_fit_rms_px = float(np.sqrt(((fitted - ordered) ** 2).sum(1).mean()))
    if res.holder_fit_rms_px > C.HOLDER_MAX_FIT_RMS_PX:
        res.reasons.append(f'holder: edges deviate from a {C.HOLDER_Y:.0f}x{C.HOLDER_X:.0f} mm '
                           f'rectangle (rms {res.holder_fit_rms_px:.1f} px)')
    ppm = float(np.sqrt(np.linalg.det(Rs)))
    sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
    res.holder_short_px = float((sides[0] + sides[2]) / 2)
    res.holder_long_px = float((sides[1] + sides[3]) / 2)
    ex_img = fitted[1] - fitted[0]
    res.holder_angle_deg = float(np.degrees(np.arctan2(ex_img[1], ex_img[0])))
    res.holder_corners = fitted.tolist()
    res.holder_mm_to_px = H.tolist()
    res.chamfer_px = fitted[1].tolist()
    res.holder_px_per_mm = ppm
    return HolderPose(H, img, ppm)
