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
import math
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
# magnifier shown while a point is held or nudged
LOUPE_HALF_R = 2.0                    # it shows +- this many dot radii around the point
LOUPE_FRAC, LOUPE_MIN_PX = 0.4, 300   # its size on screen, fraction of the image size
FALLBACK_DOT_R_PX = 23.0              # dot radius when the holder (px/mm) is unknown
NUDGE_PX, NUDGE_FAST_PX = 0.25, 1.0   # arrow keys, without / with Shift
RADIUS_STEP_PX = 0.5                  # mouse wheel: circle radius
HELP = '\n'.join(f'{k:<9}{v}' for k, v in (
    ('hold', 'place dot, magnified'), ('drag dot', 'move it'),
    ('arrows', f'nudge {NUDGE_PX} px (Shift {NUDGE_FAST_PX:g})'), ('wheel', 'circle size'),
    ('x, right', 'not visible'), ('u', 'undo'), ('a', 'accept detector'),
    ('n', 'no dots visible'), ('h', 'holder corners'), ('f', 'full image / window'),
    ('c', 'contrast'), ('t', 'note'), ('Esc', 'hide magnifier'), ('Enter', 'save, next'),
    ('s', 'skip'), ('b', 'back'), ('q', 'quit')))


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
    """The image region [x0, x1) x [y0, y1) shown at ``scale``: display px <-> raw.png px.

    Pixel centres are at integer coordinates in both (cv2.resize's convention)."""

    def __init__(self, x0, y0, x1, y1, scale):
        self.x0, self.y0, self.x1, self.y1, self.scale = x0, y0, x1, y1, scale
        self.size = (round((x1 - x0) * scale), round((y1 - y0) * scale))

    def to_full(self, u, v):
        x = self.x0 + (u + 0.5) / self.scale - 0.5
        y = self.y0 + (v + 0.5) / self.scale - 0.5
        return [round(float(x), 2), round(float(y), 2)]

    def to_view(self, x, y):
        return ((x - self.x0 + 0.5) * self.scale - 0.5,
                (y - self.y0 + 0.5) * self.scale - 0.5)


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


def contrast_map(im, view, contrast='stretch'):
    """The view's region of im as uint8 at full resolution, with the chosen contrast."""
    crop = im[view.y0:view.y1, view.x0:view.x1]
    if contrast == 'raw':
        top = np.iinfo(crop.dtype).max if np.issubdtype(crop.dtype, np.integer) else 1.0
        return np.clip(crop.astype(np.float64) / top * 255, 0, 255).astype(np.uint8)
    g = (normalize(crop) * 255).astype(np.uint8)
    if contrast == 'clahe':
        g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    return g


def _png(g):
    """base64 PNG, what tkinter.PhotoImage takes as ``data``."""
    return base64.b64encode(cv2.imencode('.png', g)[1].tobytes()).decode('ascii')


def render(g, view):
    """The contrast-mapped region g at the view's scale, as base64 PNG."""
    interp = cv2.INTER_AREA if view.scale < 1 else cv2.INTER_LINEAR
    return _png(cv2.resize(g, view.size, interpolation=interp))


def loupe(g, view, pt, zoom, size):
    """size x size patch of g magnified zoom x (vs. raw px), with raw point pt (sub-pixel)
    exactly at pixel ((size - 1) / 2, (size - 1) / 2). Returns uint8."""
    c = (size - 1) / 2
    M = np.array([[zoom, 0, c - zoom * (pt[0] - view.x0)],
                  [0, zoom, c - zoom * (pt[1] - view.y0)]])
    return cv2.warpAffine(g, M, (size, size), flags=cv2.INTER_CUBIC)


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

    def points(self, kind):
        return self.holder if kind == 'holder' else self.dots

    def get(self, target):
        """The point of target = (kind, index), or None if it is gone or not visible."""
        pts = self.points(target[0])
        return pts[target[1]] if target[1] < len(pts) else None

    def hit(self, pt, tol):
        """The placed point nearest pt within tol, as (kind, index), or None."""
        best = None
        for kind in ('dots', 'holder'):
            for i, p in enumerate(self.points(kind)):
                if p is not None:
                    d = math.hypot(p[0] - pt[0], p[1] - pt[1])
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, (kind, i))
        return best and best[1]

    def press(self, pt, tol):
        """A press at pt grabs the placed point within tol, else places the next one.
        Returns the grabbed or new point as (kind, index), or None."""
        target = self.hit(pt, tol)
        if target is None and (self.holder_mode or len(self.dots) < 5):
            kind = 'holder' if self.holder_mode else 'dots'
            self.click(pt)
            target = (kind, len(self.points(kind)) - 1)
        return target

    def move(self, target, pt):
        self.points(target[0])[target[1]] = [round(float(pt[0]), 2), round(float(pt[1]), 2)]

    def nudge(self, target, dx, dy):
        p = self.get(target)
        if p is not None:
            self.move(target, (p[0] + dx, p[1] + dy))

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
        self.loupe_px = max(LOUPE_MIN_PX, int(LOUPE_FRAC * self.max_px))
        for widget, event, handler in (
                (self.canvas, '<ButtonPress-1>', self.on_press),
                (self.canvas, '<B1-Motion>', self.on_drag),
                (self.canvas, '<ButtonRelease-1>', self.on_release),
                (self.canvas, '<Button-3>', lambda e: self.act(self.ann.missing)),
                (self.canvas, '<Button-4>', lambda e: self.resize_circle(1)),    # X11 wheel
                (self.canvas, '<Button-5>', lambda e: self.resize_circle(-1)),
                (self.canvas, '<MouseWheel>',
                 lambda e: self.resize_circle(1 if e.delta > 0 else -1)),
                (self.root, '<Key>', self.on_key)):
            widget.bind(event, self.unless_busy(handler))
        self.root.protocol('WM_DELETE_WINDOW', self.root.destroy)
        self.root.focus_force()                                  # keys work without a click
        self.history, self.i, self.message, self.busy = [], -1, '', False
        self.r_adjust = 0.0               # wheel change to the circle radius, kept across runs
        self.next_run()

    def unless_busy(self, handler):
        """Input is dropped while a run loads: it was meant for the previous run."""
        return lambda e: None if self.busy else handler(e)

    def loading(self, fn):
        """Run fn with input disabled, then discard the input queued meanwhile."""
        self.busy = True
        try:
            fn()
        finally:
            try:
                self.root.update()                       # handlers see busy and drop it
            except self.tk.TclError:                     # window closed after the last run
                pass
            self.busy = False

    # ---- run navigation -----------------------------------------------------------
    def next_run(self):
        self.loading(self._next_run)

    def _next_run(self):
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
        self.loading(lambda: self.load(self.runs[self.i], force=True))

    def load(self, path, force=False):
        image, self.name, _ = resolve_input(path)
        self.root.title(f'spotvision-annotate  [{self.i + 1}/{len(self.runs)}]  {self.name}')
        self.root.update_idletasks()                     # redraw only; never handle input here
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
        self.selected = self.drag = None    # (kind, index) of the point being worked on
        self.magnify = False
        ppm = self.res.holder_px_per_mm
        self.dot_r = C.DOT_RADIUS_MM * ppm if ppm else FALLBACK_DOT_R_PX
        self.set_view()
        return True

    def set_view(self):
        self.view = make_view(self.im.shape, self.res, self.max_px, self.full)
        self.canvas.config(width=self.view.size[0], height=self.view.size[1])
        self.view8 = contrast_map(self.im, self.view, CONTRASTS[self.contrast])
        self.photo = self.tk.PhotoImage(data=render(self.view8, self.view))
        self.draw()

    # ---- drawing ------------------------------------------------------------------
    def draw(self):
        cv = self.canvas
        cv.delete('all')
        cv.create_image(0, 0, image=self.photo, anchor='nw')
        r = max(6.0, self.dot_r * self.view.scale)
        if self.selected is not None and self.ann.get(self.selected) is None:
            self.selected, self.magnify = None, False      # undone / replaced meanwhile
        for i, p in enumerate(self.res.dots_px or []):           # detector, for reference
            if p is not None:
                u, v = self.view.to_view(*p)
                cv.create_oval(u - r, v - r, u + r, v + r, outline=DETECTOR_COLOR, dash=(4, 3))
                cv.create_text(u - r, v - r, text=f'd{i}', fill=DETECTOR_COLOR, anchor='se')
        for i, p in enumerate(self.ann.dots):
            if p is not None:
                u, v = self.view.to_view(*p)
                width = 3 if self.selected == ('dots', i) else 2
                cv.create_oval(u - r, v - r, u + r, v + r, outline=DOT_COLORS[i], width=width)
                cv.create_line(u - 4, v, u + 4, v, fill=DOT_COLORS[i])
                cv.create_line(u, v - 4, u, v + 4, fill=DOT_COLORS[i])
                cv.create_text(u + r + 2, v, text=str(i), fill=DOT_COLORS[i], anchor='w',
                               font=('TkDefaultFont', 12, 'bold'))
        for i, p in enumerate(self.ann.holder):
            u, v = self.view.to_view(*p)
            cv.create_line(u - 8, v - 8, u + 8, v + 8, fill='yellow', width=2)
            cv.create_line(u - 8, v + 8, u + 8, v - 8, fill='yellow', width=2)
            cv.create_text(u + 10, v, text=f'H{i}', fill='yellow', anchor='w')
        if self.magnify and self.selected is not None:
            self.draw_loupe()
        dots = ', '.join('-' if p is None else str(i) for i, p in enumerate(self.ann.dots))
        reasons = '; '.join(self.res.reasons) or '-'
        saved = self.labels.get(self.name)
        saved = ('-' if saved is None else 'no dots' if saved.get('no_dots')
                 else f'{sum(p is not None for p in saved["dots"])} dots')
        lines = [f'detector: {self.res.confidence}  ({reasons})', f'saved: {saved}',
                 f'labelled dots: [{dots}]   view: {"full" if self.full else "window"}, '
                 f'contrast: {CONTRASTS[self.contrast]}   circle r: {self.circle_r():.1f} px'
                 + (f'   note: {self.ann.note}' if self.ann.note else ''),
                 f'>> {self.ann.prompt()}' + (f'   ({self.message})' if self.message else '')]
        self.status.config(text='\n'.join(lines))

    def circle_r(self):
        return max(1.0, self.dot_r + self.r_adjust)

    def draw_loupe(self):
        """Magnified patch centred on the selected point, with the dot-sized circle."""
        kind, i = self.selected
        p = self.ann.get(self.selected)
        L = self.loupe_px
        zoom = L / (2 * LOUPE_HALF_R * self.dot_r)
        self.loupe_photo = self.tk.PhotoImage(data=_png(loupe(self.view8, self.view, p, zoom, L)))
        u, v = self.view.to_view(*p)
        W, H = self.view.size                                  # keep it on the canvas
        ox = min(max(round(u - (L - 1) / 2), 0), max(W - L, 0))
        oy = min(max(round(v - (L - 1) / 2), 0), max(H - L, 0))
        cx, cy = ox + (L - 1) / 2, oy + (L - 1) / 2           # where p is in the patch
        cv, color = self.canvas, DOT_COLORS[i] if kind == 'dots' else 'yellow'
        cv.create_image(ox, oy, image=self.loupe_photo, anchor='nw')
        cv.create_rectangle(ox, oy, ox + L, oy + L, outline='white')
        if kind == 'dots':
            rr = self.circle_r() * zoom
            cv.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=color, width=2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):       # crosshair, open centre
            cv.create_line(cx + 4 * dx, cy + 4 * dy, cx + 14 * dx, cy + 14 * dy, fill=color)
        name = f'dot {i}' if kind == 'dots' else f'H{i}'
        cv.create_text(ox + 6, oy + 4, anchor='nw', fill=color,
                       text=f'{name}  ({p[0]:.2f}, {p[1]:.2f})')

    # ---- input --------------------------------------------------------------------
    def act(self, fn):
        self.message = ''
        fn()
        self.draw()

    def on_press(self, e):
        pt = self.view.to_full(e.x, e.y)
        self.message = ''
        target = self.ann.press(pt, tol=self.dot_r)
        if target is None:
            return self.draw()
        p = self.ann.get(target)
        self.grab = (p[0] - pt[0], p[1] - pt[1])       # an existing point does not jump
        self.drag = self.selected = target
        self.magnify = True
        self.canvas.config(cursor='none')
        self.draw()

    def on_drag(self, e):
        if self.drag is None:
            return
        x, y = self.view.to_full(e.x, e.y)
        self.ann.move(self.drag, (x + self.grab[0], y + self.grab[1]))
        self.draw()

    def on_release(self, e):
        if self.drag is None:
            return
        self.drag, self.magnify = None, False
        self.canvas.config(cursor='crosshair')
        self.draw()

    def nudge(self, dx, dy, fast):
        if self.selected is None:
            self.message = 'no dot selected: click or drag one first'
        else:
            step = NUDGE_FAST_PX if fast else NUDGE_PX
            self.ann.nudge(self.selected, dx * step, dy * step)
            self.magnify = True
        self.draw()

    def resize_circle(self, sign):
        self.r_adjust += sign * RADIUS_STEP_PX
        self.draw()

    def save(self, label):
        self.labels[self.name] = label
        save_labels(self.path, self.labels)
        self.next_run()

    def on_key(self, e):
        k = e.keysym
        arrows = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}
        if k in arrows:
            return self.nudge(*arrows[k], fast=bool(e.state & 0x1))   # 0x1: Shift
        if self.magnify:                                    # any other key closes it
            self.magnify = False
            self.draw()
        if k == 'Escape':
            return
        if k == 'q':
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
