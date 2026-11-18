"""End-to-end tests on synthetic scenes with known ground truth."""
import cv2
import numpy as np
import pytest

from spotvision import CLEAN, DEGRADED, REFUSED, analyze
from spotvision import constants as C

CORNER_PX, PPM_REL, T_MM, M_ABS, DOT_PX = 1.5, 3e-3, 0.05, 2e-3, 0.2


def assert_pose(res, truth):
    np.testing.assert_allclose(res.transform['translation'], truth['translation'], atol=T_MM)
    np.testing.assert_allclose(res.transform['matrix'], truth['matrix'], atol=M_ABS)
    assert abs(res.sample_angle_deg - truth['sample_deg']) < 0.1


def assert_holder(res, truth):
    assert np.abs(np.array(res.holder_corners) - truth['holder_corners']).max() < CORNER_PX
    assert abs(res.holder_px_per_mm / truth['ppm'] - 1) < PPM_REL
    np.testing.assert_allclose(res.chamfer_px, truth['holder_corners'][1], atol=CORNER_PX)


def test_clean_scene_full_accuracy(scene):
    _, truth, res = scene()
    assert res.confidence == CLEAN and res.reasons == []
    assert res.n_dots == 5
    assert_holder(res, truth)
    np.testing.assert_allclose(res.window_corners, truth['window_corners'], atol=CORNER_PX)
    for d, p in zip(res.dots_px, truth['dots_px']):
        assert np.linalg.norm(np.array(d) - p) < DOT_PX
    assert_pose(res, truth)
    assert res.dots_rms_px < 0.2
    assert res.transform['rms_error_mm'] < 0.02
    assert abs(res.plane_scale_ratio - 1) < 3e-3
    assert res.perpendicularity_deg < 0.1
    assert res.mirrored_rms_px > 20 * res.similarity_rms_px
    np.testing.assert_allclose(res.arm_spacing_mm, [0, 10, 20, 6, 12], atol=0.03)


# at 8 deg the far Y dot rises 2.8 mm: the corner sits lower so that it stays in the window
@pytest.mark.parametrize('holder_deg,sample_deg,corner_mm', [
    (1.5, 0.0, (3.2, 3.2)), (-2.5, 1.0, (3.2, 3.2)), (180.0, -3.0, (3.2, 3.2)),
    (0.0, 8.0, (4.5, 3.2))])
def test_rotations(scene, holder_deg, sample_deg, corner_mm):
    _, truth, res = scene(holder_deg=holder_deg, sample_deg=sample_deg, corner_mm=corner_mm)
    assert res.confidence == CLEAN
    assert_holder(res, truth)
    assert_pose(res, truth)


def test_holder_frame_follows_chamfer_not_image(scene):
    """Rotating the holder 180 deg moves the origin to the other side of the image."""
    _, _, upright = scene()
    _, _, flipped = scene(holder_deg=180.0)
    assert upright.holder_corners[0][0] < 1224 < flipped.holder_corners[0][0]
    assert abs(abs(flipped.holder_angle_deg - upright.holder_angle_deg) - 180) < 0.1


def test_dark_dots(scene):
    _, truth, res = scene(polarity='dark')
    assert res.confidence == CLEAN
    assert res.dot_polarity in ('dark', 'mixed')
    assert_pose(res, truth)


def test_uneven_middle_dot_is_still_clean(scene):
    """Some samples have the middle Y dot at ~9.3 mm; only the arms define the frame."""
    _, truth, res = scene(dot_offsets=((1, (0.0, -0.7)),))
    assert res.confidence == CLEAN
    assert abs(res.arm_spacing_mm[1] - 9.3) < 0.05
    assert_pose(res, truth)


def test_missing_dot_degrades(scene):
    _, truth, res = scene(missing=(2,))
    assert res.confidence == DEGRADED
    assert res.n_dots == 4 and res.dots_px[2] is None
    assert any('4 of 5' in r for r in res.reasons)
    assert_pose(res, truth)


def test_no_dots_refused(scene):
    _, _, res = scene(missing=(0, 1, 2, 3, 4))
    assert res.confidence == REFUSED
    assert res.transform is None
    assert res.holder_corners is not None                    # holder still reported
    assert any('no hypothesis' in r for r in res.reasons)


def test_dots_away_from_window_corner_refused(scene):
    _, _, res = scene(corner_mm=(15.0, 15.0))
    assert res.confidence == REFUSED


def test_mirrored_pattern_refused(scene):
    """Arms swapped (6 mm pitch along +Y, 10 mm along +X) is a handedness fault."""
    swapped = tuple((i, (Y - X, X - Y)) for i, (X, Y) in enumerate(C.DOT_TEMPLATE))
    _, _, res = scene(dot_offsets=swapped)
    assert res.confidence == REFUSED


def test_distractor_grid_ignored(scene):
    grid = tuple((10.0 + 7 * i, 8.0 + 7 * j) for i in range(4) for j in range(4))
    _, truth, res = scene(distractors=grid)
    assert res.confidence == CLEAN
    assert_pose(res, truth)


def test_no_holder_refused():
    im = np.random.default_rng(0).normal(128, 20, (2048, 2448)).clip(0, 255).astype(np.uint8)
    res = analyze(im)
    assert res.confidence == REFUSED
    assert res.holder_corners is None
    assert res.reasons == ['holder: no plate-shaped component']


def test_color_and_16bit_input(scene):
    im, truth, ref = scene()
    res = analyze(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
    assert res.confidence == CLEAN
    np.testing.assert_allclose(res.transform['translation'], ref.transform['translation'], atol=1e-9)
    res16 = analyze(im.astype(np.uint16) * 257)
    assert res16.confidence == CLEAN
    np.testing.assert_allclose(res16.transform['translation'], truth['translation'], atol=T_MM)


def test_rejects_non_image():
    with pytest.raises(ValueError):
        analyze(np.zeros((4, 4, 4, 4)))
