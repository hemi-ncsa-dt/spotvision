"""``spotvision-evaluate``: score the detector against hand labels (``spotvision-annotate``).

Per run: a label of "no dots" should be refused; a label with dots should get a pose whose
dots match the labelled ones. Poses are compared as L-frames fitted to the labelled dots
(origin in mm, X-axis angle in degrees), so a dot the detector missed still counts if the
pose it found is right.
"""
import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from . import constants as C
from .annotate import load_labels
from .dots import _determined, fit_l_frame
from .pipeline import analyze

WRONG_DOT_R = 0.5          # a detected dot this many radii from the label is a wrong dot


def label_frame(dots, H):
    """L-frame (origin, ex, ey) in px fitted to labelled dots, or None if underdetermined.
    The holder axes (H maps holder (Y, X) mm to px) seed the fit."""
    found = [None if p is None else np.asarray(p, float) for p in dots]
    if not _determined(found):
        return None
    H = np.asarray(H)
    ey0, ex0 = H[:2, 0] / np.linalg.norm(H[:2, 0]), H[:2, 1] / np.linalg.norm(H[:2, 1])
    o0 = found[0] if found[0] is not None else np.mean([f for f in found if f is not None], 0)
    o, ex, ey, _ = fit_l_frame(found, ex0, ey0, o0)
    return o, ex, ey


def score_run(res, label):
    """One run's comparison of a ``Result`` with its label (a dict, see annotate.py)."""
    row = {'confidence': res.confidence, 'no_dots': bool(label.get('no_dots'))}
    if row['no_dots']:
        return row
    ppm = res.holder_px_per_mm
    lab = label['dots']
    det = res.dots_px or [None] * 5
    R = C.DOT_RADIUS_MM * ppm if ppm else None
    err = {i: float(np.hypot(*np.subtract(det[i], lab[i])))
           for i in range(5) if lab[i] is not None and det[i] is not None}
    row.update(
        n_label=sum(p is not None for p in lab),
        dot_err_px=err,
        missed=[i for i in range(5) if lab[i] is not None and det[i] is None],
        extra=[i for i in range(5) if lab[i] is None and det[i] is not None],
        wrong=[i for i, e in err.items() if R and e > WRONG_DOT_R * R])
    if res.sample_frame_px and res.holder_mm_to_px:
        fr = label_frame(lab, res.holder_mm_to_px)
        if fr is not None:
            o, ex, _ = fr
            sf = res.sample_frame_px
            dx = np.asarray(sf['x_axis'])
            d_ang = np.degrees(np.arctan2(ex[0] * dx[1] - ex[1] * dx[0], ex @ dx))
            row['origin_err_mm'] = float(np.linalg.norm(np.subtract(sf['origin'], o)) / ppm)
            row['angle_err_deg'] = float(d_ang)
    return row


def _evaluate_one(args):
    name, root, label = args
    im = cv2.imread(os.path.join(root, name, 'raw.png'), cv2.IMREAD_UNCHANGED)
    if im is None:
        return name, None
    return name, score_run(analyze(im), label)


def evaluate(labels, root='.', jobs=1):
    """{run name: row} for every labelled run whose raw.png exists under root."""
    work = [(n, root, labels[n]) for n in sorted(labels)
            if os.path.isfile(os.path.join(root, n, 'raw.png'))]
    if jobs > 1:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
            out = list(ex.map(_evaluate_one, work))
    else:
        out = [_evaluate_one(w) for w in work]
    return {n: r for n, r in out if r is not None}


def _stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {'n': len(v), 'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)),
            'max': float(v.max())}


def summarize(rows):
    nd = [r for r in rows.values() if r['no_dots']]
    wd = [r for r in rows.values() if not r['no_dots']]
    posed = [r for r in wd if r['confidence'] != 'refused']   # dot stats: runs with a pose
    pose = [r for r in posed if 'origin_err_mm' in r]
    return {
        'runs': len(rows),
        'no_dots': {'n': len(nd), 'refused': sum(r['confidence'] == 'refused' for r in nd),
                    'false_pose': sum(r['confidence'] != 'refused' for r in nd)},
        'with_dots': {'n': len(wd), 'clean': sum(r['confidence'] == 'clean' for r in wd),
                      'degraded': sum(r['confidence'] == 'degraded' for r in wd),
                      'refused': len(wd) - len(posed)},
        'dots': {'labelled': sum(r['n_label'] for r in posed),
                 'found': sum(len(r['dot_err_px']) for r in posed),
                 'missed': sum(len(r['missed']) for r in posed),
                 'extra': sum(len(r['extra']) for r in posed),
                 'wrong': sum(len(r['wrong']) for r in posed),
                 'err_px': _stats([e for r in posed for e in r['dot_err_px'].values()])},
        'pose': {'origin_err_mm': _stats([r['origin_err_mm'] for r in pose]),
                 'angle_err_deg': _stats([abs(r['angle_err_deg']) for r in pose])},
    }


def _fmt(s):
    return '-' if s is None else (f"median {s['median']:.2f}, p90 {s['p90']:.2f}, "
                                  f"max {s['max']:.2f} (n={s['n']})")


def report(rows, verbose=False):
    s = summarize(rows)
    lines = [f"runs: {s['runs']}",
             f"labelled no dots: {s['no_dots']['n']}: refused {s['no_dots']['refused']}, "
             f"FALSE POSE {s['no_dots']['false_pose']}",
             f"labelled with dots: {s['with_dots']['n']}: clean {s['with_dots']['clean']}, "
             f"degraded {s['with_dots']['degraded']}, refused {s['with_dots']['refused']}",
             f"dots in runs with a pose: labelled {s['dots']['labelled']}, "
             f"found {s['dots']['found']}, "
             f"missed {s['dots']['missed']}, extra {s['dots']['extra']}, "
             f"wrong {s['dots']['wrong']}",
             f"dot error px: {_fmt(s['dots']['err_px'])}",
             f"pose origin error mm: {_fmt(s['pose']['origin_err_mm'])}",
             f"pose angle error deg: {_fmt(s['pose']['angle_err_deg'])}"]
    if verbose:
        lines.append('')
        for n, r in sorted(rows.items()):
            if r['no_dots']:
                flag = '' if r['confidence'] == 'refused' else '  FALSE POSE'
                lines.append(f"{n[:60]:60s} no dots  {r['confidence']}{flag}")
                continue
            errs = ' '.join(f'{i}:{e:.1f}' for i, e in sorted(r['dot_err_px'].items()))
            pose = (f" origin {r['origin_err_mm']:.2f}mm angle {r['angle_err_deg']:+.2f}deg"
                    if 'origin_err_mm' in r else '')
            lines.append(f"{n[:60]:60s} {r['n_label']} dots {r['confidence']:8s} "
                         f"err[{errs}] missed{r['missed']} extra{r['extra']} "
                         f"wrong{r['wrong']}{pose}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='spotvision-evaluate',
                                 description='Score the detector against a labels file.')
    ap.add_argument('labels', help='labels JSON from spotvision-annotate')
    ap.add_argument('--root', default='.', help='directory holding the run directories')
    ap.add_argument('-j', '--jobs', type=int, default=1, help='parallel worker processes')
    ap.add_argument('-v', '--verbose', action='store_true', help='one line per run')
    ap.add_argument('--json', help='also write the per-run rows and summary to this file')
    args = ap.parse_args(argv)
    rows = evaluate(load_labels(args.labels), args.root, args.jobs)
    if not rows:
        print('no labelled run found under', args.root)
        return 1
    print(report(rows, args.verbose))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summarize(rows), 'runs': rows}, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
