"""Regression checks on captured runs; skipped when the run folders are not present."""
import os

import cv2
import numpy as np
import pytest

from spotvision import analyze
from conftest import REPO

pytestmark = pytest.mark.realdata


def load(run):
    path = os.path.join(REPO, run, 'raw.png')
    if not os.path.isfile(path):
        pytest.skip(f'{run} not available')
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def angle(res):
    m = res.transform['matrix']
    return np.degrees(np.arctan2(m[1][0], m[0][0]))


@pytest.mark.parametrize('run,translation,deg', [
    # agrees with the old COORD pipeline's clean result (2.43, 3.06) mm / -0.69 deg
    ('JHAMAL00018-028_69d6b39ef0d5d4b7f27fe46e_0_2071_2026-09-15_16-39-46', (2.50, 3.07), -0.75),
    # middle Y dot at ~9.3 mm on this sample family
    ('JHAMAA00008-005_6aaace8e3415e897e0786536_0_2088_2026-09-24_18-53-15', None, None),
    # very dark exposure
    ('TEST00_000000000000000000000000_0_2183_2026-10-15_12-55-08', None, None),
    # dots ~2.4 mm from the window edges: NCC must ignore the edge gap and bevel
    ('APLMAA00001-002_6ac8ed51b13375b643372e44_0_2239_2026-11-02_16-02-01', None, None),
])
def test_clean_runs(run, translation, deg):
    res = analyze(load(run))
    assert res.confidence == 'clean', res.reasons
    assert res.n_dots == 5
    if translation:
        np.testing.assert_allclose(res.transform['translation'], translation, atol=0.05)
        assert abs(angle(res) - deg) < 0.1


def test_uneven_middle_dot_measured():
    res = analyze(load('JHAMAA00008-005_6aaace8e3415e897e0786536_0_2088_2026-09-24_18-53-15'))
    assert abs(res.arm_spacing_mm[1] - 9.34) < 0.05


@pytest.mark.parametrize('run', [
    'JHBMAI00005-003_6aaafd0eade031549e2d4688_0_2076_2026-09-17_19-01-00',    # no visible dots
    'JHAMAL00020-S8R0C1_6a428ee4529e4df6c516a1c8_0_2075_2026-09-17_18-47-48',  # rock texture
])
def test_refused_runs(run):
    res = analyze(load(run))
    assert res.confidence == 'refused'
    assert res.transform is None
    assert res.holder_corners is not None


def test_glare_run_degraded_with_four_dots():
    # camera reflection over the dots; dot 1 is missed, the other four are right
    res = analyze(load('APLMAA00001-008_6ac8ed6b6ed7698c1288dc37_0_2239_2026-11-02_16-05-02'))
    assert res.confidence == 'degraded'
    assert [d is not None for d in res.dots_px] == [True, False, True, True, True]
