"""End-to-end analysis of one image."""
import cv2
import numpy as np

from . import constants as C
from .dots import find_dots
from .geometry import px_to_holder
from .holder import find_holder
from .result import Result
from .version import ALGORITHM_VERSION
from .window import find_window


def to_gray(im):
    """Accept grayscale or BGR(A) images of any integer/float dtype."""
    im = np.asarray(im)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY if im.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if im.ndim != 2:
        raise ValueError(f'expected a 2-D grayscale image, got shape {im.shape}')
    return im


def analyze(im):
    """Detect holder, window and fiducial dots in one image and return a ``Result``."""
    im = to_gray(im)
    res = Result()
    holder = find_holder(im, res)
    if holder is None:
        return res
    win = find_window(holder, res)
    frame = find_dots(holder.img, win, holder, res)
    if frame is not None:
        _set_pose(res, holder, *frame)
    return res


def _set_pose(res, holder, o, ex, ey):
    """Sample pose in holder mm and the COORD-style transform."""
    H, ppm = holder.H, holder.ppm
    o_h, x_h, y_h = px_to_holder(H, [o, o + ex * 10 * ppm, o + ey * 10 * ppm])
    ux, uy = (x_h - o_h) / np.linalg.norm(x_h - o_h), (y_h - o_h) / np.linalg.norm(y_h - o_h)
    res.ok = True
    res.sample_frame_px = {'origin': o.tolist(), 'x_axis': ex.tolist(), 'y_axis': ey.tolist()}
    res.sample_origin_holder_mm = o_h.tolist()
    res.sample_angle_deg = float(np.degrees(np.arctan2(ux[1], ux[0])))
    # sample (X,Y) mm -> holder (X,Y) mm; columns are the sample axes in holder mm, scaled
    # by the plane ratio and offset by the window origin, as the COORD contract does
    res.transform = {
        'algorithm_version': ALGORITHM_VERSION,
        'matrix': (res.plane_scale_ratio * np.column_stack([ux, uy])).tolist(),
        'rms_error_mm': res.dots_rms_px / ppm,
        'translation': (o_h - [C.WINDOW_X0, C.WINDOW_Y0]).tolist(),
    }
