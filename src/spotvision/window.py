"""Sample window: placed rigidly from the holder pose."""
import numpy as np

from . import constants as C
from .geometry import holder_to_px
from .holder import refine_rect


def find_window(holder, res):
    """The window is machined into the holder, so it is placed from the holder pose
    (40 x 40 mm at X=15, Y=62.5 mm), always square to it. The groove around it is
    measured only as a check on that pose (``window_groove_offset_mm``)."""
    X0, Y0, S = C.WINDOW_X0, C.WINDOW_Y0, C.WINDOW_SIZE
    win = holder_to_px(holder.H, [[X0, Y0], [X0 + S, Y0], [X0 + S, Y0 + S], [X0, Y0 + S]])
    res.window_corners = win.tolist()
    meas, _ = refine_rect(holder.img, win.astype(np.float64),
                          search=C.WINDOW_GROOVE_SEARCH_MM * holder.ppm,
                          t_range=C.WINDOW_GROOVE_SPAN, mode='valley',
                          max_tilt_deg=C.WINDOW_GROOVE_MAX_TILT_DEG)
    if meas is not None:
        res.window_groove_offset_mm = float(
            np.median(np.linalg.norm(meas - win, axis=1)) / holder.ppm)
    return win
