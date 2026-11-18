"""Command line: ``spotvision -o OUT INPUT...``"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import cv2

from .metadata import write_run
from .pipeline import analyze
from .version import __version__


def resolve_input(path):
    """(image path, run name, input metadata.json path or None) for a run dir or image."""
    if os.path.isdir(path):
        image = os.path.join(path, 'raw.png')
        run_dir = path
    else:
        image = path
        run_dir = os.path.dirname(os.path.abspath(path))
    if os.path.basename(image) == 'raw.png':
        name = os.path.basename(os.path.abspath(run_dir))
    else:
        name = os.path.splitext(os.path.basename(image))[0]
    meta = os.path.join(run_dir, 'metadata.json')
    return image, name, meta if os.path.isfile(meta) else None


def expand_inputs(paths):
    """A directory without raw.png stands for the run directories directly inside it."""
    out = []
    for p in paths:
        if os.path.isdir(p) and not os.path.isfile(os.path.join(p, 'raw.png')):
            out += sorted(os.path.join(p, d) for d in os.listdir(p)
                          if os.path.isfile(os.path.join(p, d, 'raw.png')))
        else:
            out.append(p)
    return out


def process(path, out_root, overlay_scale=0.5, include_detection=True):
    """Analyze one input and write its outputs. Returns a summary dict."""
    image, name, meta = resolve_input(path)
    im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if im is None:
        return {'name': name, 'error': f'cannot read image {image}'}
    t0 = time.monotonic()
    res = analyze(im)
    elapsed = time.monotonic() - t0
    acquisition = None
    if meta:
        with open(meta, encoding='utf-8') as f:
            acquisition = json.load(f)
    write_run(os.path.join(out_root, name), im, res, elapsed, acquisition, overlay_scale,
              include_detection)
    return {'name': name, 'elapsed_s': elapsed, 'confidence': res.confidence,
            'n_dots': res.n_dots, 'rms_px': res.dots_rms_px, 'reasons': res.reasons}


def _line(s):
    if 'error' in s:
        return f'{s["name"]:60s} ERROR    {s["error"]}'
    rms = f'{s["rms_px"]:.2f}' if s['rms_px'] is not None else '-'
    return (f'{s["name"]:60s} {s["elapsed_s"]:4.2f}s {s["confidence"]:8s} '
            f'dots={s["n_dots"]} rms={rms} {"; ".join(s["reasons"])}')


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='spotvision',
        description='Detect sample holder, window and fiducial dots; write overlay.png, '
                    'metadata.json and transform.json per input into OUT/<run name>/.')
    ap.add_argument('inputs', nargs='+',
                    help='run directories (containing raw.png), image files, or a directory '
                         'of run directories')
    ap.add_argument('-o', '--out', required=True, help='output root directory')
    ap.add_argument('-j', '--jobs', type=int, default=1, help='parallel worker processes')
    ap.add_argument('--overlay-scale', type=float, default=0.5,
                    help='overlay downscale factor; 0 disables the overlay')
    ap.add_argument('--no-detection', dest='detection', action='store_false',
                    help="leave the full Result ('detection') out of metadata.json")
    ap.add_argument('-q', '--quiet', action='store_true', help='only print the summary')
    ap.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = ap.parse_args(argv)
    args.inputs = expand_inputs(args.inputs)
    if not args.inputs:
        ap.error('no runs found in the given inputs')

    if args.jobs > 1:
        # spawn: forking a process that already runs OpenCV/BLAS threads can deadlock
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn')) as ex:
            results = ex.map(process, args.inputs, [args.out] * len(args.inputs),
                             [args.overlay_scale] * len(args.inputs),
                             [args.detection] * len(args.inputs))
            summaries = []
            for s in results:
                summaries.append(s)
                if not args.quiet:
                    print(_line(s), flush=True)
    else:
        summaries = []
        for path in args.inputs:
            s = process(path, args.out, args.overlay_scale, args.detection)
            summaries.append(s)
            if not args.quiet:
                print(_line(s), flush=True)

    counts = Counter(s.get('confidence', 'error') for s in summaries)
    print('summary: ' + ', '.join(f'{k} {counts[k]}' for k in
                                  ('clean', 'degraded', 'refused', 'error') if counts[k]))
    return 1 if counts['error'] else 0


if __name__ == '__main__':
    sys.exit(main())
