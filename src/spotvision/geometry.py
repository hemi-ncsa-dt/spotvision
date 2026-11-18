"""Small geometric and image-sampling helpers."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def normalize(im):
    """Float image stretched to [0, 1] between the 1st and 99.5th percentiles."""
    im = im.astype(np.float32)
    lo, hi = np.percentile(im, [1, 99.5])
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)


def line_intersect(l1, l2):
    """Intersection of two lines given as (point, direction)."""
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def robust_line(pts, iters=3, tol=1.5):
    """Huber line fit with iterative outlier trimming. Returns ((point, dir), inlier_frac)."""
    pts = np.asarray(pts, np.float32)
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        vx, vy, x0, y0 = cv2.fitLine(pts[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy]); p = np.array([x0, y0])
        n = np.array([-vy, vx])
        res = np.abs((pts - p) @ n)
        keep = res < max(tol, 2.5 * np.median(res[keep]))
    return (p, d), keep.mean()


def sample_profiles(img, pts, normal, offsets):
    """img sampled at pts[:,None] + offsets[None,:]*normal -> (n_pts, n_off)."""
    xy = pts[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    return ndi.map_coordinates(img, [xy[..., 1], xy[..., 0]], order=1, mode='nearest')


def subpixel_peak(g, j):
    """Parabolic refinement of the per-row peak at column j of g.

    Returns (offset in samples, clipped to [-1, 1], peak value)."""
    r = np.arange(len(j))
    y0, y1, y2 = g[r, j - 1], g[r, j], g[r, j + 1]
    den = y0 - 2 * y1 + y2
    sub = 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, np.inf)
    return np.clip(sub, -1, 1), y1


def similarity_fit(src, dst):
    """Least-squares similarity (rotation + uniform scale + translation, no reflection).

    Returns (R, t) with dst ~ R @ src + t, where R already includes the scale."""
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    a = (s * d).sum()
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum()
    n = (s ** 2).sum()
    c, sn = a / n, b / n                         # scale*cos, scale*sin
    R = np.array([[c, -sn], [sn, c]])
    t = md - R @ ms
    return R, t


def inside_quad(quad, pts, margin=0.0):
    """pts (..., 2) inside convex quad by at least margin px (negative = may stick out)."""
    c = quad.mean(0)
    ok = np.ones(pts.shape[:-1], bool)
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        n = np.array([-(p1 - p0)[1], (p1 - p0)[0]]); n /= np.linalg.norm(n)
        if (c - p0) @ n < 0:
            n = -n
        ok &= (pts - p0) @ n >= margin
    return ok


def holder_to_px(H, XY):
    """Holder (X, Y) mm, shape (n, 2) -> image px (n, 2). H maps (Y, X) mm to px."""
    pts = np.asarray(XY, np.float64)[:, ::-1].reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def px_to_holder(H, px):
    """Image px, shape (n, 2) -> holder (X, Y) mm."""
    pts = np.asarray(px, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)[:, ::-1]
