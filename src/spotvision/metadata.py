"""COORD-compatible ``metadata.json`` / ``transform.json`` and run-directory output."""
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone

import cv2
import numpy as np

from . import constants as C
from .overlay import draw
from .version import ALGORITHM_VERSION

# capture facts copied from the input run's metadata.json (never detection results)
ACQUISITION_KEYS = ('camera_model', 'camera_settings', 'instruction_no', 'run_id',
                    'sample_id', 'sample_igsn', 'timestamp')


def diagnostics(res):
    """The COORD ``diagnostics`` block (same keys) plus a few extra checks."""
    ppm = res.holder_px_per_mm
    hdet = -ppm ** 2 if ppm else None              # X-down/Y-right mm frame: handedness -1
    psr = res.plane_scale_ratio
    return {
        'calibrated': False,
        'chamfer_confidence': res.chamfer_contrast,
        'dot_candidates': res.dot_candidates,
        'dot_edge_blur_px': res.dot_edge_blur_px,
        'dot_polarity': res.dot_polarity,
        'holder_angle_deg': res.holder_angle_deg,
        'holder_aspect': res.holder_long_px / res.holder_short_px if res.holder_short_px else None,
        'holder_det': hdet,
        'holder_fit_rms_px': res.holder_fit_rms_px,
        'holder_long_px': res.holder_long_px,
        'holder_px_per_mm': ppm,
        'holder_short_px': res.holder_short_px,
        'mirrored_rms_px': res.mirrored_rms_px,
        'n_dots': res.n_dots,
        'perpendicularity_deg': res.perpendicularity_deg,
        'plane_scale_ratio': psr,
        'px_per_mm': ppm,
        'px_per_mm_source': 'holder_edge',
        'relative_det': psr ** 2 if psr else None,
        'sample_det': hdet * psr ** 2 if hdet and psr else None,
        'sample_rms_px': res.dots_rms_px,
        'similarity_rms_px': res.similarity_rms_px,
        'window_groove_offset_mm': res.window_groove_offset_mm,
    }


def to_jsonable(value):
    """Plain-Python copy of value that is valid strict JSON: numpy scalars and arrays become
    Python numbers and lists, tuples become lists, and NaN/inf become None."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_metadata(res, im, elapsed_s, acquisition=None, include_detection=True):
    """metadata.json in the COORD layout, as strict-JSON plain Python.

    ``timestamp`` and the other acquisition keys appear only when ``acquisition`` (the input
    run's metadata.json) provides them; ``processed_at`` is always the processing time.
    ``include_detection=False`` leaves out the full ``Result`` ('detection')."""
    acq = {k: v for k, v in (acquisition or {}).items() if k in ACQUISITION_KEYS}
    meta = {
        **acq,
        'algorithm_version': ALGORITHM_VERSION,
        'confidence': res.confidence,
        'diagnostics': diagnostics(res),
        'elapsed_s': round(elapsed_s, 3),
        'image': {'dtype': str(im.dtype), 'shape': list(im.shape)},
        'processed_at': datetime.now(timezone.utc).isoformat(),
        'reasons': list(res.reasons),
        'sample_window_origin_in_detected_holder_mm': [C.WINDOW_X0, C.WINDOW_Y0],
        'transform': res.transform,
    }
    if include_detection:
        meta['detection'] = asdict(res)
    return to_jsonable(meta)


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_run(out_dir, im, res, elapsed_s, acquisition=None, overlay_scale=0.5,
              include_detection=True):
    """Write overlay.png (unless overlay_scale is None), metadata.json and, when there is a
    pose, transform.json into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    if overlay_scale:
        cv2.imwrite(os.path.join(out_dir, 'overlay.png'), draw(im, res, overlay_scale))
    write_json(os.path.join(out_dir, 'metadata.json'),
               build_metadata(res, im, elapsed_s, acquisition, include_detection))
    if res.transform is not None:
        write_json(os.path.join(out_dir, 'transform.json'), res.transform)
