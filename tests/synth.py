"""Synthetic station images with known ground truth.

Scene (all layers anti-aliased): bright table, dark holder frame, bright plate with the
chamfer, the window groove, the sample square and the five dots. The holder is mapped
to the image by a rotation about the image centre; the sample sits in the holder plane.
"""
import cv2
import numpy as np

from spotvision import constants as C

SHIFT = 4


def _layer(canvas, draw, level):
    mask = np.zeros(canvas.shape, np.uint8)
    draw(mask)
    a = mask.astype(np.float32) / 255
    canvas *= 1 - a
    canvas += level * a


def _poly(pts):
    return [np.round(np.asarray(pts) * (1 << SHIFT)).astype(np.int32)]


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def render(holder_deg=0.0, sample_deg=0.0, corner_mm=(3.2, 3.2), ppm=15.55,
           polarity='bright', dot_offsets=None, missing=(), distractors=(),
           sample_level=110.0, noise=3.0, seed=0, shape=(2048, 2448)):
    """Render a scene. Returns (uint8 image, truth dict).

    dot_offsets : {dot index: (dX, dY) mm} (or pairs) moves dots, sample frame
    missing     : dot indices not drawn
    distractors : extra discs, sample-frame (X, Y) mm, same size as the dots"""
    h, w = shape
    centre = np.array([w / 2, h / 2])
    Rh = rot(holder_deg)

    def to_px(XY):                                   # holder (X, Y) mm -> px
        XY = np.atleast_2d(np.asarray(XY, float))
        YX = XY[:, ::-1] - [C.HOLDER_Y / 2, C.HOLDER_X / 2]
        return centre + ppm * YX @ Rh.T

    Rs = rot(sample_deg)                              # acts on (X, Y) vectors
    corner_h = np.array([C.WINDOW_X0, C.WINDOW_Y0]) + corner_mm

    def sample_to_holder(XY):
        return corner_h + np.atleast_2d(np.asarray(XY, float)) @ Rs.T

    canvas = np.full(shape, 150.0, np.float32)        # table
    m = 15.0
    frame = to_px([[-m, -m], [C.HOLDER_X + m, -m], [C.HOLDER_X + m, C.HOLDER_Y + m],
                   [-m, C.HOLDER_Y + m]])
    _layer(canvas, lambda k: cv2.fillPoly(k, _poly(frame), 255, cv2.LINE_AA, SHIFT), 25)
    ch = 8.0                                          # chamfer leg, mm
    plate = to_px([[0, 0], [C.HOLDER_X - ch, 0], [C.HOLDER_X, ch], [C.HOLDER_X, C.HOLDER_Y],
                   [0, C.HOLDER_Y]])
    _layer(canvas, lambda k: cv2.fillPoly(k, _poly(plate), 255, cv2.LINE_AA, SHIFT), 180)
    X0, Y0, S = C.WINDOW_X0, C.WINDOW_Y0, C.WINDOW_SIZE
    win = to_px([[X0, Y0], [X0 + S, Y0], [X0 + S, Y0 + S], [X0, Y0 + S]])
    _layer(canvas, lambda k: cv2.polylines(k, _poly(win), True, 255, 4, cv2.LINE_AA, SHIFT), 60)
    inset = 0.4
    sample = to_px([[X0 + inset, Y0 + inset], [X0 + S - inset, Y0 + inset],
                    [X0 + S - inset, Y0 + S - inset], [X0 + inset, Y0 + S - inset]])
    _layer(canvas, lambda k: cv2.fillPoly(k, _poly(sample), 255, cv2.LINE_AA, SHIFT),
           sample_level)

    dot_level = {'bright': 235.0, 'dark': 40.0}[polarity]
    R = C.DOT_RADIUS_MM * ppm
    dots_s = C.DOT_TEMPLATE.copy()
    for i, d in dict(dot_offsets or {}).items():
        dots_s[i] += d
    dots_px = to_px(sample_to_holder(dots_s))
    discs = [p for i, p in enumerate(dots_px) if i not in missing]
    discs += list(to_px(sample_to_holder(distractors))) if len(distractors) else []
    for p in discs:
        _layer(canvas, lambda k, p=p: cv2.circle(k, tuple(int(round(v * (1 << SHIFT))) for v in p),
                                                 int(round(R * (1 << SHIFT))), 255, -1,
                                                 cv2.LINE_AA, SHIFT), dot_level)

    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.0)
    canvas += np.random.default_rng(seed).normal(0, noise, shape)
    im = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    truth = {
        'ppm': ppm,
        'holder_corners': to_px([[0, 0], [C.HOLDER_X, 0], [C.HOLDER_X, C.HOLDER_Y],
                                 [0, C.HOLDER_Y]]),
        'window_corners': win,
        'dots_px': dots_px,
        'matrix': Rs,
        'translation': np.asarray(corner_mm, float),
        'sample_deg': sample_deg,
        'to_px': to_px,
    }
    return im, truth
