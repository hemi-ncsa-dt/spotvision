"""Inspection overlay."""
import cv2
import numpy as np

from . import constants as C
from .geometry import holder_to_px, normalize
from .result import CLEAN, DEGRADED

_SHIFT = 4                                   # sub-pixel drawing
YELLOW, CYAN, BLUE = (0, 200, 255), (255, 220, 0), (255, 120, 40)
ORANGE, GREEN, RED = (60, 120, 255), (80, 220, 80), (60, 60, 230)
_BANNER = {CLEAN: (90, 170, 60), DEGRADED: (40, 160, 230)}
_BANNER_REFUSED = (60, 60, 200)


def _p(pt):
    return tuple(int(round(v * (1 << _SHIFT))) for v in pt)


def _fmt(v, spec='.2f'):
    return format(v, spec) if v is not None else '-'


def draw(im, res, scale=0.5):
    """BGR overlay: holder (yellow) with chamfer and axes, window (cyan), dots (blue) and
    sample axes, plus a status banner listing the reasons. Downscaled by ``scale``."""
    gray = im if im.dtype == np.uint8 else (normalize(im) * 255).astype(np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ppm = res.holder_px_per_mm or 15.0
    if res.holder_corners:
        hc = np.array(res.holder_corners)
        cv2.polylines(vis, [np.round(hc * (1 << _SHIFT)).astype(np.int32)], True, YELLOW, 3,
                      cv2.LINE_AA, _SHIFT)
        cv2.circle(vis, _p(res.chamfer_px), 30 << _SHIFT, ORANGE, 3, cv2.LINE_AA, _SHIFT)
        cv2.putText(vis, 'chamfer', tuple(int(v) + 35 for v in res.chamfer_px), 0, 1.2, ORANGE, 3)
        o, xe, ye = holder_to_px(np.array(res.holder_mm_to_px), [[0, 0], [20, 0], [0, 20]])
        for e, col, lab in ((xe, RED, 'H+X'), (ye, GREEN, 'H+Y')):
            cv2.arrowedLine(vis, _p(o), _p(e), col, 4, cv2.LINE_AA, _SHIFT, 0.15)
            cv2.putText(vis, lab, tuple(int(v) for v in e + 8), 0, 1.1, col, 3)
    if res.window_corners:
        wc = np.array(res.window_corners)
        cv2.polylines(vis, [np.round(wc * (1 << _SHIFT)).astype(np.int32)], True, CYAN, 3,
                      cv2.LINE_AA, _SHIFT)
    R = C.DOT_RADIUS_MM * ppm
    for i, d in enumerate(res.dots_px or []):
        if d is None:
            continue
        cv2.circle(vis, _p(d), int(R * (1 << _SHIFT)), BLUE, 3, cv2.LINE_AA, _SHIFT)
        cv2.drawMarker(vis, tuple(int(v) for v in d), BLUE, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, str(i), (int(d[0] + R), int(d[1] - R)), 0, 1.0, BLUE, 2)
    if res.sample_frame_px:
        o = np.array(res.sample_frame_px['origin'])
        for key, col, lab in (('x_axis', RED, 'S+X'), ('y_axis', GREEN, 'S+Y')):
            e = o + 15 * ppm * np.array(res.sample_frame_px[key])
            cv2.arrowedLine(vis, _p(o), _p(e), col, 3, cv2.LINE_AA, _SHIFT, 0.12)
            cv2.putText(vis, lab, tuple(int(v) + 8 for v in e), 0, 1.0, col, 3)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    conf = res.confidence
    lines = [f'{conf.upper()}  dots {res.n_dots}/5  rms {_fmt(res.dots_rms_px)}px  '
             f'px/mm {_fmt(res.holder_px_per_mm, ".3f")}  pol {res.dot_polarity or "-"}  '
             f'angle {_fmt(res.sample_angle_deg)}deg  perp {_fmt(res.perpendicularity_deg)}deg',
             *res.reasons]
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22 * len(lines) + 10),
                  _BANNER.get(conf, _BANNER_REFUSED), -1)
    for i, s in enumerate(lines):
        cv2.putText(vis, s, (8, 22 + 22 * i), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
