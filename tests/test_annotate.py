"""Non-GUI parts of spotvision-annotate: view mapping, label building, label file."""
import base64
import json

import cv2
import numpy as np
import pytest

from spotvision.annotate import (Annotation, View, contrast_map, load_labels, loupe, make_view,
                                 render, runs_to_label, save_labels)
from spotvision.result import Result


def test_view_round_trip_and_window_crop(scene):
    im, _, res = scene()
    view = make_view(im.shape, res, 800)
    win = np.asarray(res.window_corners)
    assert view.x0 < win[:, 0].min() and view.x1 > win[:, 0].max()
    assert view.y0 < win[:, 1].min() and view.y1 > win[:, 1].max()
    assert max(view.size) == 800
    u, v = view.to_view(1300.25, 800.5)
    assert view.to_full(u, v) == [1300.25, 800.5]


def test_view_uses_pixel_centres():
    # 2x zoom: display px 0 and 1 cover raw px 0, whose centre is at raw 0.0
    view = View(10, 20, 110, 120, 2.0)
    assert view.to_full(0, 0) == [9.75, 19.75] and view.to_full(1, 1) == [10.25, 20.25]
    assert view.to_view(10.0, 20.0) == (0.5, 0.5)


def test_full_view_without_holder():
    view = make_view((2048, 2448), Result(), 1000)
    assert (view.x0, view.y0, view.x1, view.y1) == (0, 0, 2448, 2048)
    assert view.size == (1000, round(2048 * 1000 / 2448))


@pytest.mark.parametrize('contrast', ['stretch', 'clahe', 'raw'])
def test_render_png(scene, contrast):
    im, _, res = scene()
    view = make_view(im.shape, res, 500)
    g = contrast_map(im, view, contrast)
    assert g.shape == (view.y1 - view.y0, view.x1 - view.x0) and g.dtype == np.uint8
    png = cv2.imdecode(np.frombuffer(base64.b64decode(render(g, view)), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    assert png.shape == view.size[::-1] and png.dtype == np.uint8


def test_render_16bit():
    im = (np.arange(200 * 300).reshape(200, 300) * 4).astype(np.uint16)
    view = make_view(im.shape, Result(), 150)
    for contrast in ('stretch', 'raw'):
        assert render(contrast_map(im, view, contrast), view)


def test_loupe_centres_subpixel_point():
    # linear ramp: the loupe centre pixel must show the value at the (sub-pixel) point
    yy, xx = np.mgrid[:100, :120]
    g = (xx + 0.5 * yy).astype(np.float32)                   # float: no quantisation
    view = View(20, 10, 140, 110, 1.0)
    for pt in ([60.0, 40.0], [60.4, 40.0], [75.25, 52.75]):
        out = loupe(g, view, pt, zoom=8, size=101)
        assert out.shape == (101, 101)
        expect = (pt[0] - view.x0) + 0.5 * (pt[1] - view.y0)
        assert abs(float(out[50, 50]) - expect) < 0.05         # warpAffine: 1/32 px steps
        assert out[50, 58] > out[50, 50] > out[50, 42]           # +x is right


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


def test_annotation_press_drag_nudge():
    a = Annotation()
    assert a.press([10.0, 10.0], tol=5) == ('dots', 0)        # empty spot: new dot
    assert a.press([12.0, 11.0], tol=5) == ('dots', 0)        # near dot 0: grabs it
    assert len(a.dots) == 1
    a.move(('dots', 0), (10.123, 9.876))
    assert a.dots == [[10.12, 9.88]]
    a.nudge(('dots', 0), 0.25, -1.0)
    assert a.get(('dots', 0)) == [10.37, 8.88]
    assert a.press([50.0, 10.0], tol=5) == ('dots', 1)
    a.missing()
    a.nudge(('dots', 2), 1, 1)                                # a not-visible dot stays None
    assert a.dots[2] is None and a.get(('dots', 7)) is None
    a.dots += [[0.0, 0.0], [1.0, 1.0]]
    assert a.press([200.0, 200.0], tol=5) is None             # all placed, nothing near
    assert a.hit([0.5, 0.4], tol=5) == ('dots', 3)            # nearest of two within tol


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
