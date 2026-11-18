"""``spotvision-annotate``: click the fiducial dots on runs the detector gets wrong.

Labels go to one JSON file keyed by run name, one run per line:

  "<run>": {"dots": [[x, y] or null, ...5]}   dot centres in raw.png px, template order
  "<run>": {"no_dots": true}                  no dot visible to a human either

optionally with "holder" (the 4 plate corners, starting at the chamfer corner) and "note".
The file is rewritten after every saved run, so quitting never loses work.
"""
import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np

from . import constants as C
from .cli import expand_inputs, resolve_input
from .geometry import normalize
from .pipeline import analyze, to_gray

DOT_NAMES = ('corner dot', 'wide arm (10 mm), 1st', 'wide arm (10 mm), 2nd',
             'narrow arm (6 mm), 1st', 'narrow arm (6 mm), 2nd')
DOT_COLORS = ('#ff4040', '#40c0ff', '#40c0ff', '#40ff60', '#40ff60')
DETECTOR_COLOR = '#ffa000'              # the detector's dots, shown for reference
CONTRASTS = ('stretch', 'clahe', 'raw')
VIEW_MARGIN = 0.25                    # around the window crop, fraction of the window size
PANEL_CHARS = 34                      # width of the side panel, in average characters
HELP = '\n'.join(f'{k:<9}{v}' for k, v in (
    ('click', 'place dot'), ('x, right', 'not visible'), ('u', 'undo'),
    ('a', 'accept detector'), ('n', 'no dots visible'), ('h', 'holder corners'),
    ('f', 'full image / window'), ('c', 'contrast'), ('t', 'note'),
    ('Enter', 'save, next'), ('s', 'skip'), ('b', 'back'), ('q', 'quit')))


def load_labels(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_labels(path, labels):
    """One run per line (readable diffs); written to a temp file and renamed into place."""
    body = ',\n'.join(f'  {json.dumps(k)}: {json.dumps(labels[k])}' for k in sorted(labels))
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('{\n' + body + ('\n' if body else '') + '}\n')
    os.replace(tmp, path)


class View:
    """The image region [x0, x1) x [y0, y1) shown at ``scale``: display px <-> raw.png px."""

    def __init__(self, x0, y0, x1, y1, scale):
        self.x0, self.y0, self.x1, self.y1, self.scale = x0, y0, x1, y1, scale
        self.size = (round((x1 - x0) * scale), round((y1 - y0) * scale))

    def to_full(self, u, v):
        x, y = self.x0 + u / self.scale, self.y0 + v / self.scale
        return [round(float(x), 1), round(float(y), 1)]

    def to_view(self, x, y):
        return (x - self.x0) * self.scale, (y - self.y0) * self.scale


def make_view(shape, res, max_px, full=False):
    """Window region plus a margin when the window is known, else the whole image."""
    H, W = shape[:2]
    if res.window_corners is not None and not full:
        win = np.asarray(res.window_corners)
        lo, hi = win.min(0), win.max(0)
        pad = VIEW_MARGIN * (hi - lo)
        x0, y0 = np.maximum(np.floor(lo - pad), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(hi + pad), [W, H]).astype(int)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    return View(int(x0), int(y0), int(x1), int(y1), min(max_px / (x1 - x0), max_px / (y1 - y0)))


def render(im, view, contrast='stretch'):
    """The view as base64 PNG (what tkinter.PhotoImage takes as ``data``)."""
    crop = im[view.y0:view.y1, view.x0:view.x1]
    if contrast == 'raw':
        top = np.iinfo(crop.dtype).max if np.issubdtype(crop.dtype, np.integer) else 1.0
        g = np.clip(crop.astype(np.float64) / top * 255, 0, 255).astype(np.uint8)
    else:
        g = (normalize(crop) * 255).astype(np.uint8)
        if contrast == 'clahe':
            g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    interp = cv2.INTER_AREA if view.scale < 1 else cv2.INTER_LINEAR
    g = cv2.resize(g, view.size, interpolation=interp)
    return base64.b64encode(cv2.imencode('.png', g)[1].tobytes()).decode('ascii')


class Annotation:
    """The label being built for one run."""

    def __init__(self, label=None):
        label = label or {}
        self.dots = list(label.get('dots', []))
        self.holder = list(label.get('holder', []))
        self.note = label.get('note')
        self.holder_mode = False

    def click(self, pt):
        if self.holder_mode:
            self.holder.append(pt)
            self.holder_mode = len(self.holder) < 4
        elif len(self.dots) < 5:
            self.dots.append(pt)

    def missing(self):
        if not self.holder_mode and len(self.dots) < 5:
            self.dots.append(None)

    def undo(self):
        if self.holder_mode and self.holder:
            self.holder.pop()
        elif self.dots:
            self.dots.pop()

    def start_holder(self):
        self.holder, self.holder_mode = [], True

    def prompt(self):
        if self.holder_mode:
            return f'click holder corner {len(self.holder)} (0 = chamfer corner, then around)'
        if len(self.dots) < 5:
            i = len(self.dots)
            return f'click dot {i}: {DOT_NAMES[i]}  (x if not visible)'
        return 'all 5 dots placed: Enter to save'

    def label(self):
        """The finished label, or None while dots are still unplaced."""
        if self.holder_mode or len(self.dots) < 5:
            return None
        out = {'dots': self.dots}
        if len(self.holder) == 4:
            out['holder'] = self.holder
        if self.note:
            out['note'] = self.note
        return out

    def no_dots(self):
        out = {'no_dots': True}
        if len(self.holder) == 4:
            out['holder'] = self.holder
        if self.note:
            out['note'] = self.note
        return out


def runs_to_label(inputs, labels, relabel=False):
    """Run paths from the CLI inputs, minus already-labelled runs unless relabel."""
    runs = expand_inputs(inputs)
    return [p for p in runs if relabel or resolve_input(p)[1] not in labels]


class App:
    """The Tk window. Keeps everything except drawing in View / Annotation."""

    def __init__(self, runs, labels_path, include_clean=False, size=None):
        import tkinter as tk
        import tkinter.font as tkfont
        self.tk = tk
        self.runs, self.path, self.include_clean = runs, labels_path, include_clean
        self.labels = load_labels(labels_path)
        self.root = tk.Tk()
        self.root.title('spotvision-annotate')
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        # image on the left, fixed-width text panel on the right (screens are landscape)
        panel_w = tkfont.nametofont('TkDefaultFont').measure('n' * PANEL_CHARS)
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(side='left', anchor='n')
        panel = tk.Frame(self.root, width=panel_w)
        panel.pack(side='left', fill='y', padx=8, pady=8)
        self.status = tk.Label(panel, anchor='nw', justify='left', wraplength=panel_w)
        self.status.pack(fill='x')
        self.help = tk.Label(panel, text=HELP, anchor='nw', justify='left', fg='#666',
                             font='TkFixedFont')
        self.help.pack(fill='x', side='bottom')
        fit = min(sh - 100, sw - panel_w - 60)                  # title bar, panels, padding
        self.max_px = max(400, min(size or fit, fit))
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Button-3>', lambda e: self.act(self.ann.missing))
        self.root.bind('<Key>', self.on_key)
        self.root.protocol('WM_DELETE_WINDOW', self.root.destroy)
        self.root.focus_force()                                  # keys work without a click
        self.history, self.i, self.message = [], -1, ''
        self.next_run()

    # ---- run navigation -----------------------------------------------------------
    def next_run(self):
        while True:
            self.i += 1
            if self.i >= len(self.runs):
                self.root.destroy()
                return
            if self.load(self.runs[self.i]):
                self.history.append(self.i)
                return

    def previous_run(self):
        if len(self.history) < 2:
            self.message = 'no earlier run'
            return self.draw()
        self.history.pop()
        self.i = self.history[-1]
        self.load(self.runs[self.i], force=True)

    def load(self, path, force=False):
        image, self.name, _ = resolve_input(path)
        self.root.title(f'spotvision-annotate  [{self.i + 1}/{len(self.runs)}]  {self.name}')
        self.root.update()
        im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f'{self.name}: cannot read {image}, skipped', file=sys.stderr)
            return False
        self.im = to_gray(im)
        self.res = analyze(self.im)
        if self.res.confidence == 'clean' and not (self.include_clean or force):
            return False
        self.ann = Annotation(self.labels.get(self.name))
        self.full, self.contrast, self.message = False, 0, ''
        self.set_view()
        return True

    def set_view(self):
        self.view = make_view(self.im.shape, self.res, self.max_px, self.full)
        self.canvas.config(width=self.view.size[0], height=self.view.size[1])
        self.photo = self.tk.PhotoImage(
            data=render(self.im, self.view, CONTRASTS[self.contrast]))
        self.draw()

    # ---- drawing ------------------------------------------------------------------
    def draw(self):
        cv = self.canvas
        cv.delete('all')
        cv.create_image(0, 0, image=self.photo, anchor='nw')
        ppm = self.res.holder_px_per_mm
        r = max(6.0, (C.DOT_RADIUS_MM * ppm if ppm else 20) * self.view.scale)
        for i, p in enumerate(self.res.dots_px or []):           # detector, for reference
            if p is not None:
                u, v = self.view.to_view(*p)
                cv.create_oval(u - r, v - r, u + r, v + r, outline=DETECTOR_COLOR, dash=(4, 3))
                cv.create_text(u - r, v - r, text=f'd{i}', fill=DETECTOR_COLOR, anchor='se')
        for i, p in enumerate(self.ann.dots):
            if p is not None:
                u, v = self.view.to_view(*p)
                cv.create_oval(u - r, v - r, u + r, v + r, outline=DOT_COLORS[i], width=2)
                cv.create_line(u - 4, v, u + 4, v, fill=DOT_COLORS[i])
                cv.create_line(u, v - 4, u, v + 4, fill=DOT_COLORS[i])
                cv.create_text(u + r + 2, v, text=str(i), fill=DOT_COLORS[i], anchor='w',
                               font=('TkDefaultFont', 12, 'bold'))
        for i, p in enumerate(self.ann.holder):
            u, v = self.view.to_view(*p)
            cv.create_line(u - 8, v - 8, u + 8, v + 8, fill='yellow', width=2)
            cv.create_line(u - 8, v + 8, u + 8, v - 8, fill='yellow', width=2)
            cv.create_text(u + 10, v, text=f'H{i}', fill='yellow', anchor='w')
        dots = ', '.join('-' if p is None else str(i) for i, p in enumerate(self.ann.dots))
        reasons = '; '.join(self.res.reasons) or '-'
        saved = self.labels.get(self.name)
        saved = ('-' if saved is None else 'no dots' if saved.get('no_dots')
                 else f'{sum(p is not None for p in saved["dots"])} dots')
        lines = [f'detector: {self.res.confidence}  ({reasons})', f'saved: {saved}',
                 f'labelled dots: [{dots}]   view: {"full" if self.full else "window"}, '
                 f'contrast: {CONTRASTS[self.contrast]}'
                 + (f'   note: {self.ann.note}' if self.ann.note else ''),
                 f'>> {self.ann.prompt()}' + (f'   ({self.message})' if self.message else '')]
        self.status.config(text='\n'.join(lines))

    # ---- input --------------------------------------------------------------------
    def act(self, fn):
        self.message = ''
        fn()
        self.draw()

    def on_click(self, e):
        self.act(lambda: self.ann.click(self.view.to_full(e.x, e.y)))

    def save(self, label):
        self.labels[self.name] = label
        save_labels(self.path, self.labels)
        self.next_run()

    def on_key(self, e):
        k = e.keysym
        if k in ('q', 'Escape'):
            self.root.destroy()
        elif k in ('Return', 'KP_Enter'):
            label = self.ann.label()
            if label is None:
                self.message = 'place or x every dot first (or n = no dots)'
                self.draw()
            else:
                self.save(label)
        elif k == 'n':
            self.save(self.ann.no_dots())
        elif k == 's':
            self.next_run()
        elif k == 'b':
            self.previous_run()
        elif k == 'x':
            self.act(self.ann.missing)
        elif k in ('u', 'BackSpace'):
            self.act(self.ann.undo)
        elif k == 'h':
            self.full = True
            self.act(self.ann.start_holder)
            self.set_view()
        elif k == 'a':
            if self.res.dots_px:
                self.ann.dots = [None if p is None else [round(float(c), 1) for c in p]
                                 for p in self.res.dots_px]
                self.draw()
            else:
                self.message = 'detector found no dots'
                self.draw()
        elif k == 'f':
            self.full = not self.full
            self.set_view()
        elif k == 'c':
            self.contrast = (self.contrast + 1) % len(CONTRASTS)
            self.set_view()
        elif k == 't':
            from tkinter import simpledialog
            note = simpledialog.askstring('note', f'note for {self.name}',
                                          initialvalue=self.ann.note or '', parent=self.root)
            if note is not None:
                self.ann.note = note.strip() or None
            self.draw()

    def run(self):
        if self.i < len(self.runs):
            self.root.mainloop()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='spotvision-annotate',
        description='Click the 5 fiducial dots of each run (by default only runs the detector '
                    'does not get clean) and save them to a labels JSON file.')
    ap.add_argument('inputs', nargs='+',
                    help='run directories (containing raw.png) or a directory of them')
    ap.add_argument('-o', '--labels', default='labels.json',
                    help='labels file, created or extended (default: labels.json)')
    ap.add_argument('--all', action='store_true', help='also show runs the detector gets clean')
    ap.add_argument('--relabel', action='store_true', help='also show already-labelled runs')
    ap.add_argument('--size', type=int, help='largest image size in screen px '
                                             '(default: as large as the screen allows)')
    args = ap.parse_args(argv)
    try:
        import tkinter  # noqa: F401
    except ImportError:
        ap.error('needs tkinter (e.g. apt install python3-tk)')
    runs = runs_to_label(args.inputs, load_labels(args.labels), args.relabel)
    if not runs:
        print('nothing to label')
        return 0
    app = App(runs, args.labels, include_clean=args.all, size=args.size)
    app.run()
    labels = load_labels(args.labels)
    print(f'{len(labels)} runs labelled in {args.labels}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
