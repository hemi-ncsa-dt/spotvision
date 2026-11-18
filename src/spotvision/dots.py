"""Fiducial dots: candidates, template hypotheses, verification, L-frame fit."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from . import constants as C
from .geometry import inside_quad, px_to_holder, similarity_fit, subpixel_peak

# template (X, Y) mm is fitted as (Y, X) so that an unrotated sample maps with the
# holder's +Y image direction as rotation angle and keeps the holder's handedness
_TMPL_YX = C.DOT_TEMPLATE[:, ::-1]


def dot_candidates(img, win, R, max_n=C.DOT_MAX_CANDIDATES):
    """Blob candidates around the window from a multi-scale, scale-normalised LoG.

    Returns {'bright' | 'dark' | 'mixed': (points px (n, 2), responses (n,))}; 'mixed'
    merges both polarities (dots can flip polarity within one image under glare)."""
    x0, y0 = np.floor(win.min(0) - R).astype(int)       # pad so the LoG sees edge dots whole
    x1, y1 = np.ceil(win.max(0) + R).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    roi = img[y0:y1, x0:x1]
    sigmas = [f * R / np.sqrt(2) for f in C.DOT_LOG_SCALES]
    logs = [s ** 2 * ndi.gaussian_laplace(roi, s) for s in sigmas]
    out = {}
    for pol, resp in (('bright', np.max([-L for L in logs], 0)),
                      ('dark', np.max(logs, 0))):
        mx = ndi.maximum_filter(resp, size=int(R))
        pk = np.argwhere((resp == mx) & (resp > C.DOT_LOG_MIN_RESPONSE))
        v = resp[pk[:, 0], pk[:, 1]]
        idx = np.argsort(-v)[:max_n]
        out[pol] = (pk[idx][:, ::-1] + [x0, y0]).astype(np.float64), v[idx]
    p = np.r_[out['bright'][0], out['dark'][0]]
    v = np.r_[out['bright'][1], out['dark'][1]]
    keep = []
    for k in np.argsort(-v):
        if all(np.linalg.norm(p[k] - p[j]) > R for j in keep):
            keep.append(k)
    out['mixed'] = p[keep[:max_n]], v[keep[:max_n]]
    return out


def fit_template(cands, scores, tmpl, ppm, R, win, exp_ang, corner_ok, scale_tol=C.DOT_SCALE_TOL):
    """Similarity hypotheses tmpl (mm) -> px from every candidate pair, under the priors:
    scale ~ holder px/mm, rotation within MAX_SAMPLE_ROT_DEG of exp_ang, all dots inside
    the window and the corner dot near the window's top-left (corner_ok).

    Returns up to N_HYPOTHESES distinct [(score, Rm, t, match)], match[i] = candidate or -1."""
    n = len(cands)
    if n < 2:
        return []
    hyps = []
    tol = C.DOT_MATCH_TOL_R * R
    I, J = np.where(~np.eye(n, dtype=bool))
    dc = cands[J] - cands[I]
    for a in range(len(tmpl)):
        for b in range(len(tmpl)):
            if a == b:
                continue
            dt = tmpl[b] - tmpl[a]
            s = np.linalg.norm(dc, axis=1) / np.linalg.norm(dt)
            z = (dc[:, 0] + 1j * dc[:, 1]) / (dt[0] + 1j * dt[1])   # rotation dt -> dc
            dang = np.angle(z * np.exp(-1j * exp_ang), deg=True)
            ok = (np.abs(s / ppm - 1) < scale_tol) & (np.abs(dang) < C.MAX_SAMPLE_ROT_DEG)
            if not ok.any():
                continue
            z = z[ok]
            Rm = np.stack([np.stack([z.real, -z.imag], -1), np.stack([z.imag, z.real], -1)], -2)
            t = cands[I[ok]] - Rm @ tmpl[a]
            proj = np.einsum('kij,pj->kpi', Rm, tmpl) + t[:, None, :]     # (k, 5, 2)
            valid = inside_quad(win, proj, -C.DOT_WINDOW_OVERHANG_R * R).all(1)
            valid &= corner_ok(t)                              # tmpl[0] = corner dot = origin
            d = np.linalg.norm(proj[:, :, None, :] - cands[None, None], axis=-1)  # (k, 5, n)
            nn = d.argmin(-1); dmin = d.min(-1)
            inl = dmin < tol
            strength = (inl * scores[nn] * (1 - dmin / tol)).sum(1)
            sc = (inl.sum(1) + C.DOT_MATCH_STRENGTH_WEIGHT * strength) * valid
            for k in np.argsort(-sc)[:C.N_HYPOTHESES]:
                if sc[k] > 0:
                    hyps.append((sc[k], Rm[k], t[k], np.where(inl[k], nn[k], -1)))
    hyps.sort(key=lambda h: -h[0])
    out = []
    for h in hyps:                                   # NMS on the predicted origin
        if all(np.linalg.norm(h[2] - o[2]) > R for o in out):
            out.append(h)
        if len(out) == C.N_HYPOTHESES:
            break
    return out


def refine_dot(img, c, R, polarity, n_rays=48):
    """Sub-pixel circle fit to the radial edge around c (two passes, trimmed Kasa fit).

    polarity 'bright' / 'dark' uses the signed edge, anything else |gradient|.
    Returns (center, radius, edge inlier fraction) or None."""
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    c = np.asarray(c, float)
    step = 0.5                                       # px along each ray
    for _ in range(2):
        rs = np.arange(C.DOT_RAY_RANGE_R[0] * R, C.DOT_RAY_RANGE_R[1] * R, step)
        xy = c[None, None] + rs[None, :, None] * dirs[:, None]
        prof = ndi.map_coordinates(img, [xy[..., 1], xy[..., 0]], order=1, mode='nearest')
        g = np.gradient(prof, axis=1)
        g = {'bright': -g, 'dark': g}.get(polarity, np.abs(g))
        j = np.clip(g.argmax(1), 1, len(rs) - 2)
        sub, peak = subpixel_peak(g, j)
        rr = rs[j] + step * sub
        good = peak > C.DOT_EDGE_MIN_PEAK * np.median(peak) + 1e-6
        P = c + rr[:, None] * dirs
        keep = good.copy()
        for _ in range(3):
            if keep.sum() < 8:
                return None
            A = np.c_[2 * P[keep], np.ones(keep.sum())]
            bvec = (P[keep] ** 2).sum(1)
            cx, cy, k0 = np.linalg.lstsq(A, bvec, rcond=None)[0]
            rad = np.sqrt(k0 + cx ** 2 + cy ** 2)
            resid = np.abs(np.linalg.norm(P - [cx, cy], axis=1) - rad)
            trim = max(C.DOT_FIT_TRIM_PX, C.DOT_FIT_TRIM_FACTOR * np.median(resid[keep]))
            keep = good & (resid < trim)
        c = np.array([cx, cy])
    lo, hi = C.DOT_RADIUS_RANGE
    if keep.mean() < C.DOT_MIN_EDGE_FRAC or not lo < rad / R < hi:
        return None
    return c, rad, keep.mean()


def dot_ncc(img, c, rad, polarity, win=None, inset=0.0):
    """Contrast-independent disc evidence: NCC between the patch (planar background,
    estimated from a ring around the dot, removed) and an ideal disc of radius rad.
    Signed: > 0 means the expected polarity ('mixed' -> absolute value).

    With win, only pixels at least inset px inside that quad are used: dots near the
    window edge would otherwise fit their background to the dark gap and bright bevel."""
    r_in, r_out = C.DOT_NCC_RING_R[0] * rad, C.DOT_NCC_RING_R[1] * rad
    r = int(np.ceil(r_out))
    x0, y0 = int(round(c[0])) - r, int(round(c[1])) - r
    if x0 < 0 or y0 < 0 or x0 + 2 * r + 1 > img.shape[1] or y0 + 2 * r + 1 > img.shape[0]:
        return 0.0
    patch = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(np.float64)
    yy, xx = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
    dist = np.hypot(xx - c[0], yy - c[1])
    m = dist < r_out
    if win is not None:
        m &= inside_quad(win, np.stack([xx, yy], -1).astype(np.float64), inset)
    ring = m & (dist > r_in)
    full_ring_area = np.pi * (r_out ** 2 - r_in ** 2)
    if ring.sum() < C.DOT_NCC_MIN_RING_FRAC * full_ring_area:
        return 0.0                                                # too little background
    A = np.c_[xx[ring], yy[ring], np.ones(ring.sum())]
    coef = np.linalg.lstsq(A, patch[ring], rcond=None)[0]
    v = (patch - (coef[0] * xx + coef[1] * yy + coef[2]))[m]
    model = np.clip(rad + 0.5 - dist, 0, 1)[m]                    # anti-aliased disc
    v, model = v - v.mean(), model - model.mean()
    ncc = (v * model).sum() / (np.sqrt((v ** 2).sum() * (model ** 2).sum()) + 1e-12)
    return float({'bright': ncc, 'dark': -ncc}.get(polarity, abs(ncc)))


def fit_l_frame(found, ex0, ey0, o0):
    """Orthogonal L-frame (origin + perpendicular unit axes) minimising each dot's
    distance to its arm line; spacing along the arms is free. Handedness from (ex0, ey0).

    Returns (origin, ex, ey, residuals px)."""
    hand = np.sign(ex0[0] * ey0[1] - ex0[1] * ey0[0])
    xs = [found[i] for i in C.X_ARM if found[i] is not None]
    ys = [found[i] for i in C.Y_ARM if found[i] is not None]

    def axes(th):
        ex = np.array([np.cos(th), np.sin(th)])
        return ex, hand * np.array([-ex[1], ex[0]])

    def resid(p):
        o, (ex, ey) = p[:2], axes(p[2])
        return np.r_[[(q - o) @ ey for q in xs], [(q - o) @ ex for q in ys]]

    sol = least_squares(resid, np.r_[o0, np.arctan2(ex0[1], ex0[0])])
    ex, ey = axes(sol.x[2])
    return sol.x[:2], ex, ey, resid(sol.x)


def _arm_counts(found):
    return (sum(found[i] is not None for i in C.X_ARM),
            sum(found[i] is not None for i in C.Y_ARM))


def _determined(found):
    """Enough dots on both arms for the 3-parameter L-frame plus >= 1 redundant equation."""
    nx, ny = _arm_counts(found)
    return nx >= 1 and ny >= 1 and nx + ny >= 4


def _accept(img, r, R, pol, ncc, found, i, win):
    ncc[i] = dot_ncc(img, r[0], r[1], pol, win, C.DOT_NCC_WINDOW_INSET_MM / C.DOT_RADIUS_MM * R)
    if ncc[i] >= C.DOT_WEAK_NCC and r[2] >= C.DOT_WEAK_FRAC:
        found[i] = r[0]


def _best_hypothesis(img, cands, ppm, R, win, exp_ang, corner_ok):
    """Verify template hypotheses by refining every predicted dot; keep the one with the
    largest summed NCC that has enough strong / consensus evidence."""
    best = None
    for pol, (pts, sc) in cands.items():
        for _, Rm, t, match in fit_template(pts, sc / (sc.max() + 1e-9), _TMPL_YX, ppm, R,
                                            win, exp_ang, corner_ok):
            pred = (Rm @ _TMPL_YX.T).T + t
            found, ncc = [None] * 5, [None] * 5
            for i, p in enumerate(pred):
                r = refine_dot(img, pts[match[i]] if match[i] >= 0 else p, R, pol)
                if r is not None and np.linalg.norm(r[0] - p) <= C.DOT_MAX_SHIFT_R * R:
                    _accept(img, r, R, pol, ncc, found, i, win)
            acc = [v for v, f in zip(ncc, found) if f is not None]
            n_strong = sum(v >= C.DOT_STRONG_NCC for v in acc)
            if n_strong < 1 or (n_strong < C.MIN_STRONG and len(acc) < C.MIN_TOTAL_IF_WEAK):
                continue
            if best is None or sum(acc) > best[0]:
                best = (sum(acc), Rm, t, pol, found, ncc)
    return best


def _dots_list(found):
    return [None if f is None else f.tolist() for f in found]


def find_dots(img, win, holder, res):
    """Find the 5 fiducial dots and the sample L-frame; fill the dot fields of ``res``.

    Returns (origin, ex, ey) in px or None when no pose survives."""
    ppm, H = holder.ppm, holder.H
    R = C.DOT_RADIUS_MM * ppm
    hy = H[:2, :2] @ [1.0, 0.0]                           # holder +Y direction in the image
    exp_ang = np.arctan2(hy[1], hy[0])
    nominal = np.array([C.WINDOW_X0, C.WINDOW_Y0]) + C.CORNER_DOT_IN_WINDOW_MM

    def corner_ok(px):
        return (np.abs(px_to_holder(H, np.atleast_2d(px)) - nominal) <= C.CORNER_DOT_TOL_MM).all(1)

    cands = dot_candidates(img, win, R)
    res.dot_candidates = len(cands['mixed'][0])
    best = _best_hypothesis(img, cands, ppm, R, win, exp_ang, corner_ok)
    if best is None:
        res.reasons.append('dots: no hypothesis with enough dot evidence')
        return None
    _, Rm, t, pol, found, ncc = best
    res.dot_polarity = pol
    res.dots_ncc = [None if v is None else round(v, 2) for v in ncc]
    ex0, ey0 = Rm @ _TMPL_YX[3], Rm @ _TMPL_YX[1]           # template axes in px
    ex0, ey0 = ex0 / np.linalg.norm(ex0), ey0 / np.linalg.norm(ey0)

    if not _determined(found):
        res.n_dots = sum(f is not None for f in found)
        res.dots_px = _dots_list(found)
        res.reasons.append(f'dots: only {res.n_dots} of 5 resolved, pose underdetermined')
        return None

    # guided pass: weaker dots accepted where the fitted frame predicts them
    o, ex, ey, _ = fit_l_frame(found, ex0, ey0, t)
    far = [(np.linalg.norm(found[i] - o), np.linalg.norm(C.DOT_TEMPLATE[i]))
           for i in (2, 4, 1, 3) if found[i] is not None]
    s_ppm = far[0][0] / far[0][1] if far else ppm          # sample plane may sit higher
    for i in range(5):
        if found[i] is None:
            p = o + s_ppm * (C.DOT_TEMPLATE[i, 0] * ex + C.DOT_TEMPLATE[i, 1] * ey)
            r = refine_dot(img, p, R, pol)
            if r is not None and np.linalg.norm(r[0] - p) <= C.DOT_GUIDED_MAX_SHIFT_R * R:
                _accept(img, r, R, pol, ncc, found, i, win)
    res.dots_ncc = [None if v is None else round(v, 2) for v in ncc]

    # prune: dots off nominal spacing, then dots far off their arm line
    while True:
        o, ex, ey, e = fit_l_frame(found, ex0, ey0, o)
        idx = [i for i in C.X_ARM if found[i] is not None] + \
              [i for i in C.Y_ARM if found[i] is not None]
        sp_err = {i: abs((found[i] - o) @ (ex if i in C.X_ARM else ey) / ppm
                         / np.linalg.norm(C.DOT_TEMPLATE[i]) - 1) - C.SPACING_TOL[i]
                  for i in range(1, 5) if found[i] is not None}
        k = int(np.argmax(np.abs(e)))
        if sp_err and max(sp_err.values()) > 0:
            drop, why = max(sp_err, key=sp_err.get), 'spacing off nominal'
        elif abs(e[k]) > C.DOT_DROP_RESIDUAL_PX and idx[k] != 0:
            drop, why = idx[k], f'residual {abs(e[k]):.1f} px'
        else:
            break
        found[drop] = None
        res.reasons.append(f'dots: dropped dot {drop} ({why})')
        if not _determined(found):
            res.n_dots = sum(f is not None for f in found)
            res.dots_px = _dots_list(found)
            res.reasons.append('dots: too few consistent dots left, no pose')
            return None

    res.n_dots = sum(f is not None for f in found)
    res.dots_px = _dots_list(found)
    res.dots_rms_px = float(np.sqrt((e ** 2).mean()))
    if not corner_ok(o)[0]:
        res.reasons.append('dots: fitted corner dot too far from window top-left, no pose')
        return None
    if res.dots_rms_px > C.MAX_RMS_PX:
        res.reasons.append(f'dots: L-frame RMS {res.dots_rms_px:.2f} px > {C.MAX_RMS_PX}, no pose')
        return None

    _describe(res, found, ncc, o, ex, ey, ppm)
    return o, ex, ey


def _describe(res, found, ncc, o, ex, ey, ppm):
    """Diagnostics and degraded-confidence reasons for an accepted L-frame."""
    res.arm_spacing_mm = [None if f is None else
                          round(float((f - o) @ (ex if i in C.X_ARM else ey) / ppm), 3)
                          for i, f in enumerate(found)]
    xs = np.array([found[i] for i in C.X_ARM if found[i] is not None])
    ys = np.array([found[i] for i in C.Y_ARM if found[i] is not None])
    if len(xs) >= 2 and len(ys) >= 2:
        def axis(P):
            return np.linalg.svd(P - P.mean(0))[2][0]
        cosang = abs(axis(xs) @ axis(ys))
        res.perpendicularity_deg = float(np.degrees(np.arcsin(min(cosang, 1.0))))
    if res.n_dots < 5:
        res.reasons.append(f'dots: {res.n_dots} of 5 resolved')
    weak = [i for i, f in enumerate(found) if f is not None and ncc[i] < C.DOT_STRONG_NCC]
    if weak:
        res.reasons.append(f'dots: {weak} weak (ncc < {C.DOT_STRONG_NCC}), accepted by geometry')
    if res.dots_rms_px > C.CLEAN_RMS_PX:
        res.reasons.append(f'dots: L-frame RMS {res.dots_rms_px:.2f} px > {C.CLEAN_RMS_PX}')
    if res.perpendicularity_deg is not None and res.perpendicularity_deg > C.CLEAN_PERP_DEG:
        res.reasons.append(f'dots: arms {res.perpendicularity_deg:.2f} deg off perpendicular')
    # comparison fits on the same dots: plain similarity at both handednesses
    idx = [i for i, f in enumerate(found) if f is not None]
    src, dst = _TMPL_YX[idx], np.array([found[i] for i in idx])
    for flip, attr in ((1, 'similarity_rms_px'), (-1, 'mirrored_rms_px')):
        Rf, tf = similarity_fit(src * [1, flip], dst)
        err = np.linalg.norm((Rf @ (src * [1, flip]).T).T + tf - dst, axis=1)
        setattr(res, attr, float(np.sqrt((err ** 2).mean())))
    # plane scale from the far dots (middle-dot pitch varies between samples)
    far = [i for i in (2, 4) if found[i] is not None] or \
          [i for i in (1, 3) if found[i] is not None]
    res.plane_scale_ratio = float(np.median(
        [(found[i] - o) @ (ex if i in C.X_ARM else ey) / (np.linalg.norm(C.DOT_TEMPLATE[i]) * ppm)
         for i in far]))
