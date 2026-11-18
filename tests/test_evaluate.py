"""spotvision-evaluate on synthetic scenes with known dots."""
import json

import cv2
import numpy as np

from spotvision.evaluate import label_frame, main, score_run, summarize


def truth_label(truth, drop=()):
    return {'dots': [None if i in drop else [float(v) for v in p]
                     for i, p in enumerate(truth['dots_px'])]}


def test_score_run_perfect_label(scene):
    _, truth, res = scene()
    row = score_run(res, truth_label(truth))
    assert row['confidence'] == 'clean' and row['n_label'] == 5
    assert max(row['dot_err_px'].values()) < 0.5
    assert row['missed'] == row['extra'] == row['wrong'] == []
    assert row['origin_err_mm'] < 0.03 and abs(row['angle_err_deg']) < 0.05


def test_score_run_counts_missed_extra_wrong(scene):
    _, truth, res = scene()
    label = truth_label(truth, drop=(2,))                    # labeller did not see dot 2
    label['dots'][4] = [label['dots'][4][0] + 40, label['dots'][4][1]]  # and misplaced dot 4
    row = score_run(res, label)
    assert row['extra'] == [2] and row['wrong'] == [4] and row['missed'] == []
    _, truth, partial = scene(missing=(3,))
    assert score_run(partial, truth_label(truth))['missed'] == [3]


def test_uncertain_label_counts_presence_only(scene):
    _, truth, res = scene()
    label = truth_label(truth, drop=(2,))
    label['dots'][4] = [0.0, 0.0]                             # positions not to be trusted
    label['uncertain'] = True
    row = score_run(res, label)
    assert row['uncertain'] and row['extra'] == [2]
    assert row['dot_err_px'] == {} and row['wrong'] == [] and 'origin_err_mm' not in row
    assert summarize({'R': row})['with_dots']['uncertain'] == 1


def test_no_dots_label(scene):
    _, _, res = scene(missing=(0, 1, 2, 3, 4))
    assert score_run(res, {'no_dots': True}) == {'confidence': 'refused', 'no_dots': True}


def test_label_frame(scene):
    _, truth, res = scene(sample_deg=3.0)
    o, ex, _ = label_frame(truth_label(truth)['dots'], res.holder_mm_to_px)
    np.testing.assert_allclose(o, truth['dots_px'][0], atol=1e-6)
    assert abs(np.degrees(np.arctan2(ex[1], ex[0])) - np.degrees(np.arctan2(
        *(np.subtract(truth['dots_px'][3], truth['dots_px'][0])[::-1])))) < 1e-6
    assert label_frame(truth_label(truth, drop=(0, 1, 2))['dots'], res.holder_mm_to_px) is None


def test_summarize_and_cli(tmp_path, scene, capsys):
    im, truth, _ = scene()
    for name in ('RUN_A', 'RUN_B'):
        (tmp_path / name).mkdir()
        cv2.imwrite(str(tmp_path / name / 'raw.png'), im)
    labels = {'RUN_A': truth_label(truth), 'RUN_B': {'no_dots': True},
              'RUN_GONE': {'no_dots': True}}                 # RUN_GONE has no folder
    (tmp_path / 'labels.json').write_text(json.dumps(labels))
    out = tmp_path / 'eval.json'
    assert main([str(tmp_path / 'labels.json'), '--root', str(tmp_path), '-v',
                 '--json', str(out)]) == 0
    text = capsys.readouterr().out
    assert 'runs: 2' in text and 'FALSE POSE 1' in text      # RUN_B has dots after all
    s = json.loads(out.read_text())['summary']
    assert s == summarize(json.loads(out.read_text())['runs'])
    assert s['with_dots'] == {'n': 1, 'clean': 1, 'degraded': 0, 'refused': 0, 'uncertain': 0}
    assert s['dots']['found'] == 5 and s['dots']['wrong'] == 0
