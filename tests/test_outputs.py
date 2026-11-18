"""metadata.json / transform.json layout, run-directory writing and the CLI."""
import json
import os

import cv2
import numpy as np
import pytest

from spotvision import ALGORITHM_VERSION, build_metadata, to_jsonable, write_run
from spotvision.cli import expand_inputs, main, resolve_input

# keys of the COORD station's metadata.json / diagnostics / transform.json
COORD_METADATA = {'algorithm_version', 'camera_model', 'camera_settings', 'confidence',
                  'diagnostics', 'elapsed_s', 'image', 'instruction_no', 'reasons', 'run_id',
                  'sample_id', 'sample_igsn', 'sample_window_origin_in_detected_holder_mm',
                  'timestamp', 'transform'}
COORD_DIAGNOSTICS = {'calibrated', 'chamfer_confidence', 'dot_candidates', 'dot_polarity',
                     'holder_angle_deg', 'holder_aspect', 'holder_det', 'holder_long_px',
                     'holder_px_per_mm', 'holder_short_px', 'mirrored_rms_px', 'n_dots',
                     'perpendicularity_deg', 'plane_scale_ratio', 'px_per_mm',
                     'px_per_mm_source', 'relative_det', 'sample_det', 'sample_rms_px'}
TRANSFORM = {'algorithm_version', 'matrix', 'rms_error_mm', 'translation'}
ACQ = {'camera_model': 'Blackfly S', 'camera_settings': {'exposure_us': 13500.0},
       'instruction_no': 0, 'run_id': 2071, 'sample_id': 'abc', 'sample_igsn': 'IGSN-1',
       'timestamp': '2026-09-15T16:39:46+00:00',
       'confidence': 'clean', 'transform': {'matrix': 'old result, must not be copied'}}


def test_metadata_has_coord_layout(scene):
    im, _, res = scene()
    meta = build_metadata(res, im, 0.5, ACQ)
    assert COORD_METADATA <= set(meta)
    assert COORD_DIAGNOSTICS <= set(meta['diagnostics'])
    assert set(meta['transform']) == TRANSFORM
    assert meta['algorithm_version'] == ALGORITHM_VERSION
    assert meta['confidence'] == 'clean'
    assert meta['image'] == {'dtype': 'uint8', 'shape': [2048, 2448]}
    assert meta['run_id'] == 2071 and meta['camera_settings'] == ACQ['camera_settings']
    assert meta['transform'] == to_jsonable(res.transform)    # not the input's old result
    assert meta['diagnostics']['holder_det'] < 0               # X-down/Y-right handedness
    assert abs(meta['diagnostics']['relative_det'] - 1) < 1e-2
    assert meta['timestamp'] == ACQ['timestamp'] and 'processed_at' in meta
    assert 'detection' in meta
    assert json.loads(json.dumps(meta, allow_nan=False)) == meta  # strict JSON, plain Python


def test_metadata_without_acquisition_has_no_timestamp(scene):
    im, _, res = scene()
    meta = build_metadata(res, im, 0.1)
    assert 'timestamp' not in meta and 'run_id' not in meta and 'processed_at' in meta


def test_metadata_without_detection(scene):
    im, _, res = scene()
    meta = build_metadata(res, im, 0.1, include_detection=False)
    assert 'detection' not in meta and COORD_METADATA - ACQ.keys() <= set(meta)


def test_metadata_refused_is_strict_json(scene):
    im, _, res = scene(missing=(0, 1, 2, 3, 4))
    acq = dict(ACQ, camera_settings={'gain_db': float('nan')})
    meta = build_metadata(res, im, 0.1, acq)
    json.dumps(meta, allow_nan=False)
    assert meta['camera_settings'] == {'gain_db': None}
    assert meta['diagnostics']['sample_rms_px'] is None


def test_to_jsonable():
    value = {'a': np.float32(1.5), 'b': (np.int64(2), float('inf')), 'c': np.array([[1, np.nan]]),
             'd': [np.bool_(True), None, 'x', -float('inf')], 3: {'e': np.nan}}
    assert to_jsonable(value) == {'a': 1.5, 'b': [2, None], 'c': [[1.0, None]],
                                  'd': [True, None, 'x', None], '3': {'e': None}}
    out = to_jsonable(value)
    assert type(out['a']) is float and type(out['b'][0]) is int and type(out['d'][0]) is bool


def test_write_run_with_and_without_pose(tmp_path, scene):
    im, _, res = scene()
    write_run(tmp_path / 'ok', im, res, 0.1)
    assert sorted(os.listdir(tmp_path / 'ok')) == ['metadata.json', 'overlay.png', 'transform.json']
    overlay = cv2.imread(str(tmp_path / 'ok' / 'overlay.png'))
    assert overlay.shape == (1024, 1224, 3)
    im2, _, refused = scene(missing=(0, 1, 2, 3, 4))
    write_run(tmp_path / 'refused', im2, refused, 0.1, overlay_scale=None)
    assert os.listdir(tmp_path / 'refused') == ['metadata.json']
    meta = json.loads((tmp_path / 'refused' / 'metadata.json').read_text())
    assert meta['confidence'] == 'refused' and meta['transform'] is None


@pytest.fixture
def run_dir(tmp_path, scene):
    im, _, _ = scene()
    d = tmp_path / 'RUN_0001_2026-01-01'
    d.mkdir()
    cv2.imwrite(str(d / 'raw.png'), im)
    (d / 'metadata.json').write_text(json.dumps(ACQ))
    return d


def test_resolve_input(run_dir, tmp_path):
    assert resolve_input(str(run_dir)) == (str(run_dir / 'raw.png'), run_dir.name,
                                           str(run_dir / 'metadata.json'))
    assert resolve_input(str(run_dir / 'raw.png'))[1] == run_dir.name
    other = tmp_path / 'frame7.png'
    other.write_bytes(b'')
    assert resolve_input(str(other)) == (str(other), 'frame7', None)


def test_cli_end_to_end(run_dir, tmp_path, capsys):
    out = tmp_path / 'out'
    assert main([str(run_dir), '-o', str(out)]) == 0
    files = sorted(os.listdir(out / run_dir.name))
    assert files == ['metadata.json', 'overlay.png', 'transform.json']
    meta = json.loads((out / run_dir.name / 'metadata.json').read_text())
    assert meta['run_id'] == 2071 and meta['confidence'] == 'clean'
    assert 'summary: clean 1' in capsys.readouterr().out


def test_cli_no_overlay_and_parallel(run_dir, tmp_path):
    out = tmp_path / 'out'
    assert main([str(run_dir), str(run_dir / 'raw.png'), '-o', str(out), '-j', '2', '-q',
                 '--overlay-scale', '0', '--no-detection']) == 0
    assert 'overlay.png' not in os.listdir(out / run_dir.name)
    meta = json.loads((out / run_dir.name / 'metadata.json').read_text())
    assert 'detection' not in meta


def test_cli_unreadable_input(tmp_path, capsys):
    assert main([str(tmp_path / 'missing.png'), '-o', str(tmp_path / 'out')]) == 1
    assert 'ERROR' in capsys.readouterr().out


def test_parent_directory_expands_to_runs(run_dir, tmp_path):
    (tmp_path / 'not_a_run').mkdir()
    assert expand_inputs([str(tmp_path)]) == [str(run_dir)]
    assert expand_inputs([str(run_dir)]) == [str(run_dir)]
