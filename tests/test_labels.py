"""The detector against the hand labels in tests/data/labels.json (spotvision-annotate).

Skipped when the captured run folders are absent. The bounds sit a little outside the
current results, so a change that makes things worse fails here."""
import json
import os

import pytest

from spotvision.evaluate import evaluate, summarize
from conftest import REPO

pytestmark = pytest.mark.realdata
LABELS = os.path.join(REPO, 'tests', 'data', 'labels.json')


@pytest.fixture(scope='module')
def summary():
    with open(LABELS, encoding='utf-8') as f:
        labels = json.load(f)
    rows = evaluate(labels, REPO, jobs=min(8, os.cpu_count() or 1))
    if len(rows) < len(labels):
        pytest.skip(f'only {len(rows)} of {len(labels)} labelled runs available')
    return summarize(rows)


def test_no_false_poses(summary):
    assert summary['no_dots']['n'] >= 60
    assert summary['no_dots']['false_pose'] == 0


def test_posed_runs_have_no_wrong_or_extra_dots(summary):
    assert summary['dots']['wrong'] == 0 and summary['dots']['extra'] == 0


def test_coverage(summary):
    wd = summary['with_dots']
    assert wd['n'] - wd['refused'] >= 22 and wd['clean'] >= 7      # 10-32-49: out of focus


def test_accuracy(summary):
    o, a, d = (summary['pose']['origin_err_mm'], summary['pose']['angle_err_deg'],
               summary['dots']['err_px'])
    assert o['median'] < 0.06 and o['p90'] < 0.08 and o['max'] < 0.15
    assert a['median'] < 0.05 and a['p90'] < 0.4 and a['max'] < 2.2
    assert d['median'] < 0.7 and d['p90'] < 1.3
