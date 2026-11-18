"""Non-GUI parts of spotvision-annotate: view mapping, label building, label file."""
import base64
import json

import cv2
import numpy as np
import pytest

from spotvision.annotate import (Annotation, load_labels, make_view, render, runs_to_label,
                                 save_labels)
from spotvision.result import Result


def test_view_round_trip_and_window_crop(scene):
    im, _, res = scene()
    view = make_view(im.shape, res, 800)
    win = np.asarray(res.window_corners)
    assert view.x0 < win[:, 0].min() and view.x1 > win[:, 0].max()
    assert view.y0 < win[:, 1].min() and view.y1 > win[:, 1].max()
    assert max(view.size) == 800
    u, v = view.to_view(1300.0, 800.0)
    assert view.to_full(u, v) == [1300.0, 800.0]


def test_full_view_without_holder():
    view = make_view((2048, 2448), Result(), 1000)
    assert (view.x0, view.y0, view.x1, view.y1) == (0, 0, 2448, 2048)
    assert view.size == (1000, round(2048 * 1000 / 2448))


@pytest.mark.parametrize('contrast', ['stretch', 'clahe', 'raw'])
def test_render_png(scene, contrast):
    im, _, res = scene()
    view = make_view(im.shape, res, 500)
    png = cv2.imdecode(np.frombuffer(base64.b64decode(render(im, view, contrast)), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    assert png.shape == view.size[::-1] and png.dtype == np.uint8


def test_render_16bit():
    im = (np.arange(200 * 300).reshape(200, 300) * 4).astype(np.uint16)
    view = make_view(im.shape, Result(), 150)
    for contrast in ('stretch', 'raw'):
        assert render(im, view, contrast)


def test_annotation_flow():
    a = Annotation()
    a.click([10.0, 20.0])
    a.missing()
    a.click([30.0, 20.0])
    assert a.label() is None and 'dot 3' in a.prompt()
    a.undo()
    assert a.dots == [[10.0, 20.0], None]
    a.click([30.0, 20.0]); a.click([10.0, 30.0]); a.click([10.0, 40.0])
    a.click([99.0, 99.0])                                    # a 6th click is ignored
    assert a.label() == {'dots': [[10.0, 20.0], None, [30.0, 20.0], [10.0, 30.0], [10.0, 40.0]]}


def test_annotation_holder_and_note():
    a = Annotation({'dots': [None] * 5, 'note': 'glare'})
    a.start_holder()
    for k in range(3):
        a.click([k, k])
    assert a.label() is None and 'holder corner 3' in a.prompt()
    a.click([3, 3])
    assert not a.holder_mode
    assert a.label() == {'dots': [None] * 5, 'holder': [[0, 0], [1, 1], [2, 2], [3, 3]],
                         'note': 'glare'}
    assert a.no_dots()['no_dots'] is True and 'dots' not in a.no_dots()


def test_labels_file(tmp_path):
    path = str(tmp_path / 'labels.json')
    assert load_labels(path) == {}
    labels = {'RUN_B': {'no_dots': True}, 'RUN_A': {'dots': [[1.5, 2.0], None, None, None, None]}}
    save_labels(path, labels)
    text = (tmp_path / 'labels.json').read_text()
    assert text.splitlines()[1].startswith('  "RUN_A"')         # sorted, one run per line
    assert json.loads(text) == labels == load_labels(path)
    save_labels(path, {})
    assert load_labels(path) == {}


def test_runs_to_label(tmp_path):
    for name in ('RUN_1', 'RUN_2'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'raw.png').write_bytes(b'')
    runs = runs_to_label([str(tmp_path)], {'RUN_1': {'no_dots': True}})
    assert runs == [str(tmp_path / 'RUN_2')]
    assert len(runs_to_label([str(tmp_path)], {'RUN_1': {}}, relabel=True)) == 2
