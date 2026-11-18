import cv2
import numpy as np
import pytest

from spotvision import constants as C
from spotvision.dots import dot_candidates, dot_ncc, edge_blur, fit_l_frame, refine_dot

R = 23.0


def disc_image(centers, level=200.0, bg=100.0, shape=(300, 400), noise=2.0, seed=0, r=R):
    canvas = np.full(shape, bg, np.float32)
    for c in centers:
        m = np.zeros(shape, np.uint8)
        cv2.circle(m, tuple(int(round(v * 16)) for v in c), int(round(r * 16)), 255, -1,
                   cv2.LINE_AA, 4)
        a = m / 255.0
        canvas = canvas * (1 - a) + level * a
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.0)
    canvas += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(canvas, 0, 255) / 255.0


@pytest.mark.parametrize('level,pol', [(200, 'bright'), (30, 'dark'), (200, 'mixed'), (30, 'mixed')])
def test_refine_dot_subpixel(level, pol):
    c = np.array([150.3, 120.7])
    img = disc_image([c], level=level)
    out = refine_dot(img, c + [4, -3], R, pol)              # start off-centre
    assert out is not None
    center, rad, frac = out
    assert np.linalg.norm(center - c) < 0.1
    assert abs(rad - R) < 1.0             # max-gradient radius; only the centre feeds the pose
    assert frac > 0.9


@pytest.mark.parametrize('r', [0.4 * R, 2.2 * R])
def test_wrong_size_discs_are_not_dots(r):
    # the radius prior makes the circle fit look near the nominal radius, so a wrong-size
    # disc may still give a circle there; the disc NCC then rejects it
    img = disc_image([(150, 120)], r=r)
    out = refine_dot(img, np.array([150.0, 120.0]), R, 'bright')
    assert out is None or dot_ncc(img, out[0], out[1], 'bright') < C.DOT_WEAK_NCC


def test_refine_dot_ignores_stronger_edge_outside_window():
    # faint dot 2.0 mm from a strong window edge: without the window the edge pulls the
    # circle ~1 px
    c = np.array([150.0, 150.0])
    img = disc_image([c], level=125, bg=110, shape=(300, 300), noise=1.0)
    edge = c[0] - 2.0 / C.DOT_RADIUS_MM * R
    img[:, :int(edge) + 6] = 0.0
    win = np.array([[edge, 0], [299, 0], [299, 299], [edge, 299]])
    inset = C.DOT_WINDOW_INSET_MM / C.DOT_RADIUS_MM * R
    out = refine_dot(img, c + [3, 2], R, 'bright', win=win, inset=inset)
    assert out is not None and np.linalg.norm(out[0] - c) < 0.4 and abs(out[1] - R) < 1.0


def test_dot_ncc_polarity_and_noise():
    c = (150.0, 120.0)
    img = disc_image([c])
    assert dot_ncc(img, c, R, 'bright') > 0.95
    assert dot_ncc(img, c, R, 'dark') < -0.95
    assert dot_ncc(img, c, R, 'mixed') > 0.95
    noise = np.random.default_rng(5).normal(0.5, 0.05, (300, 400))
    assert abs(dot_ncc(noise, c, R, 'mixed')) < 0.2


def test_dot_ncc_ignores_background_gradient():
    c = (150.0, 120.0)
    img = disc_image([c], noise=0.5)
    yy, xx = np.mgrid[:300, :400]
    assert dot_ncc(img + 0.004 * xx, c, R, 'bright') > 0.95  # strong ramp, removed by the plane


def test_dot_ncc_outside_image_is_zero():
    assert dot_ncc(np.zeros((50, 50)), (5.0, 5.0), R, 'bright') == 0.0


def test_dot_candidates_finds_all_dots():
    centers = [(80, 60), (235, 60), (390, 60), (80, 153), (80, 246)]
    img = disc_image(centers, shape=(320, 480))
    win = np.array([[40, 30], [440, 30], [440, 290], [40, 290]], float)
    cands = dot_candidates(img, win, R)
    pts = cands['bright'][0][:5]
    for c in centers:
        assert np.min(np.linalg.norm(pts - c, axis=1)) < 2
    assert len(cands['mixed'][0]) >= 5


def _l_points(origin, deg, ppm=15.5, offsets=None):
    t = np.radians(deg)
    ex = np.array([np.cos(t), np.sin(t)]); ey = np.array([-ex[1], ex[0]])
    pts = []
    for i, (X, Y) in enumerate(C.DOT_TEMPLATE):
        dX, dY = (offsets or {}).get(i, (0, 0))
        pts.append(origin + ppm * ((X + dX) * ex + (Y + dY) * ey))
    return pts, ex, ey


def test_fit_l_frame_exact_and_free_along_arms():
    o = np.array([500.0, 400.0])
    # middle Y dot moved along its arm: the L-frame must not care
    found, ex, ey = _l_points(o, 97.0, offsets={1: (0, -0.66)})
    o_fit, ex_fit, ey_fit, e = fit_l_frame(found, ex + [0.02, 0], ey, o + [3, -2])
    np.testing.assert_allclose(o_fit, o, atol=1e-6)
    np.testing.assert_allclose(ex_fit, ex, atol=1e-8)
    np.testing.assert_allclose(ey_fit, ey, atol=1e-8)
    assert np.abs(e).max() < 1e-6
    assert abs(ex_fit @ ey_fit) < 1e-12                      # orthogonal by construction


def test_fit_l_frame_with_missing_corner_and_residual():
    o = np.array([500.0, 400.0])
    found, ex, ey = _l_points(o, 90.0, offsets={3: (0, 0.1)})  # dot 3 off its arm by 0.1 mm
    found = list(found)
    found[0] = None
    o_fit, _, _, e = fit_l_frame(found, ex, ey, o)
    assert np.linalg.norm(o_fit - o) < 2.0
    assert np.abs(e).max() > 0.3


def test_dot_ncc_ignores_window_edge():
    # dark gap + bright bevel 2.2 mm from a dot centre (APLMAA00001-002): the background
    # ring crosses them unless the NCC is restricted to the sample surface
    c = (150.0, 150.0)
    img = disc_image([c], level=60, bg=110, shape=(300, 300), noise=1.0)
    edge = c[0] - 2.2 / C.DOT_RADIUS_MM * R
    img[:, :int(edge) + 6] = 25 / 255
    img[:, int(edge) + 6:int(edge) + 10] = 160 / 255
    win = np.array([[edge, 0], [299, 0], [299, 299], [edge, 299]])
    inset = C.DOT_WINDOW_INSET_MM / C.DOT_RADIUS_MM * R
    assert dot_ncc(img, c, R, 'dark') < C.DOT_STRONG_NCC
    assert dot_ncc(img, c, R, 'dark', win, inset) > 0.9


@pytest.mark.parametrize('sigma', [2.0, 3.0, 5.0])
def test_edge_blur_measures_gaussian_sigma(sigma):
    img = disc_image([(150, 150)], shape=(300, 300), noise=1.0)      # already blurred by 1
    img = cv2.GaussianBlur(img, (0, 0), np.sqrt(sigma ** 2 - 1.0))
    assert abs(edge_blur(img, (150.0, 150.0), R) - sigma) < 0.2


def test_edge_blur_ignores_rays_leaving_the_window():
    c = np.array([150.0, 150.0])
    img = disc_image([c], shape=(300, 300), noise=1.0)
    edge = c[0] - 2.2 / C.DOT_RADIUS_MM * R
    img[:, :int(edge) + 6] = 0.0                                     # window gap
    win = np.array([[edge, 0], [299, 0], [299, 299], [edge, 299]])
    inset = C.DOT_WINDOW_INSET_MM / C.DOT_RADIUS_MM * R
    assert abs(edge_blur(img, c, R, win, inset) - 1.17) < 0.2      # same as without the gap
