import numpy as np
import pytest

from spotvision.geometry import (holder_to_px, inside_quad, line_intersect, normalize,
                                 px_to_holder, robust_line, similarity_fit, subpixel_peak)


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def test_similarity_fit_recovers_transform():
    rng = np.random.default_rng(1)
    src = rng.uniform(-50, 50, (12, 2))
    R_true, t_true = 2.5 * _rot(30), np.array([100.0, -20.0])
    R, t = similarity_fit(src, src @ R_true.T + t_true)
    np.testing.assert_allclose(R, R_true, atol=1e-10)
    np.testing.assert_allclose(t, t_true, atol=1e-9)


def test_similarity_fit_never_reflects():
    src = np.array([[0, 0], [10, 0], [0, 5]], float)
    R, _ = similarity_fit(src, src * [1, -1])                 # mirrored target
    assert np.linalg.det(R) >= 0


def test_robust_line_ignores_outliers():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 100, 80)
    pts = np.c_[x, 0.5 * x + 10 + rng.normal(0, 0.1, x.size)]
    pts[::5, 1] += rng.uniform(20, 40, 16)                    # 20 % gross outliers
    (p, d), frac = robust_line(pts)
    assert abs(d[1] / d[0] - 0.5) < 0.01
    assert abs(p[1] - (0.5 * p[0] + 10)) < 0.2
    assert 0.75 < frac < 0.85


def test_line_intersect():
    pt = line_intersect((np.array([0.0, 1.0]), np.array([1.0, 0.0])),
                        (np.array([3.0, 0.0]), np.array([0.0, 1.0])))
    np.testing.assert_allclose(pt, [3, 1])


def test_inside_quad_margin():
    quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    pts = np.array([[5, 5], [0.5, 5], [-0.5, 5]])
    assert inside_quad(quad, pts).tolist() == [True, True, False]
    assert inside_quad(quad, pts, margin=1).tolist() == [True, False, False]
    assert inside_quad(quad, pts, margin=-1).tolist() == [True, True, True]


def test_holder_px_round_trip_and_axis_convention():
    H = np.eye(3)
    H[:2, :2] = 15.5 * _rot(7)
    H[:2, 2] = [300, 400]
    XY = np.array([[0, 0], [70, 0], [10, 60]], float)
    px = holder_to_px(H, XY)
    np.testing.assert_allclose(px_to_holder(H, px), XY, atol=1e-9)
    # with no rotation, holder +Y is image +x and holder +X is image +y
    H0 = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]])
    np.testing.assert_allclose(holder_to_px(H0, [[1, 0], [0, 1]]), [[0, 2], [2, 0]])


@pytest.mark.parametrize('vertex', [10.0, 10.3, 9.6])
def test_subpixel_peak_on_parabola(vertex):
    x = np.arange(20, dtype=float)
    g = -(x - vertex) ** 2
    j = np.array([int(round(vertex))])
    off, peak = subpixel_peak(g[None], j)
    assert abs(j[0] + off[0] - vertex) < 1e-9


def test_normalize_range():
    im = np.random.default_rng(3).integers(0, 4096, (50, 60)).astype(np.uint16)
    n = normalize(im)
    assert np.issubdtype(n.dtype, np.floating) and n.min() == 0 and n.max() == 1
