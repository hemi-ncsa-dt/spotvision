# spotvision

[![tests](https://github.com/hemi-ncsa-dt/spotvision/actions/workflows/tests.yml/badge.svg)](https://github.com/hemi-ncsa-dt/spotvision/actions/workflows/tests.yml)

Detects, in one overhead camera image of a sample station:

1. **the sample holder**: a rigid 120 × 70 mm rectangle, with the chamfered corner that fixes its orientation;
2. **the holder window**: the 40 × 40 mm pocket the sample sits in;
3. **the fiducial dots**: 5 etched dots forming an L that defines the sample's X/Y axes;

and reports the **sample → holder transform** in the same `metadata.json` / `transform.json` layout as the station's earlier COORD pipeline. The algorithm is a from-scratch reimplementation; only COORD's output format and physical nominals were reused.

```
spotvision -o results  RUN_DIR [RUN_DIR ...]
```

[frames](#coordinate-frames) · [install](#install) · [usage](#usage) · [outputs](#outputs) · [algorithm](#algorithm) · [confidence](#confidence-and-reasons) · [tuning](#tuning) · [validation](#validation) · [limitations](#limitations) · [development](#development)

---

## Install

Python ≥ 3.10.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'      # numpy, scipy, opencv-python-headless (+ pytest)
```

## Usage

### Command line

```bash
# one or more run directories (each containing raw.png, optionally metadata.json)
spotvision -o results TEST26_000000000000000000000026_0_2261_2026-11-10_13-55-22

# a directory of run directories, 8 worker processes, summary only
spotvision -o results /data/coord_runs -j 8 -q

# plain image files work too (output folder named after the file stem)
spotvision -o results frame_0001.png --overlay-scale 0     # no overlay
```

| option | meaning |
|---|---|
| `INPUT…` | run directory (contains `raw.png`), image file, or a directory whose sub-directories are runs |
| `-o/--out` | output root; each input writes to `OUT/<run name>/` |
| `-j/--jobs N` | parallel worker processes (default 1) |
| `--overlay-scale F` | overlay downscale factor, default `0.5`; `0` disables the overlay |
| `--no-detection` | leave the full `Result` (`detection`) out of `metadata.json` |
| `-q/--quiet` | print only the final summary |

One line is printed per run (`name  time  confidence  dots  rms  reasons`), then a summary such as `summary: clean 125, degraded 18, refused 73`. The exit code is 1 if any input could not be read. `python -m spotvision` is equivalent.

### Python

```python
import cv2, spotvision

im = cv2.imread('RUN/raw.png', cv2.IMREAD_GRAYSCALE)   # BGR, uint16 also accepted
res = spotvision.analyze(im)                            # -> spotvision.Result

res.confidence            # 'clean' | 'degraded' | 'refused'
res.reasons               # why it is not clean (empty when clean)
res.holder_corners        # 4 x (x, y) px: origin, chamfer (+X end), opposite, +Y end
res.window_corners        # 4 x (x, y) px, same order
res.dots_px               # 5 x (x, y) px or None: corner, +Y mid, +Y far, +X mid, +X far
res.transform             # COORD transform dict, or None when refused

cv2.imwrite('overlay.png', spotvision.draw(im, res))
spotvision.write_run('out/RUN', im, res, elapsed_s=0.4, acquisition=None)
```

---

## Coordinate frames

```
image (x right, y down), holder upright:

  H origin ────────── +Y (long edge, 120 mm) ──────────┐
  │                               ┌─ window ─────────┐ │
  │                               │ ●───●───● +Y     │ │   window: 40 x 40 mm at
  +X (short edge, 70 mm)          │ ●  S frame       │ │   X = 15, Y = 62.5 mm
  │                               │ ●                │ │
  │                               │ +X               │ │
  │                               └──────────────────┘ │
  ╲ chamfer                                            │
   └───────────────────────────────────────────────────┘
```

* **Holder frame (mm)**: anchored to the chamfer, never to the image. The origin is the corner across the *short* edge from the chamfer. +X runs along the short edge toward the chamfer, +Y along the long edge. If the holder is rotated 180° in the image, the frame rotates with it.
* **Sample frame (mm)**: the origin is the corner dot. +X runs along the closer-spaced arm (dots at 6 and 12 mm), +Y along the wider arm (10 and 20 mm).
* Both frames are X-down / Y-right, so they are mirrored relative to the image raster (determinant < 0). Their relative transform is a proper rotation. A pattern with the opposite handedness is treated as a fault, never as a valid variant.

## Outputs

`OUT/<run>/` contains:

| file | when | content |
|---|---|---|
| `overlay.png` | unless `--overlay-scale 0` | holder (yellow) + chamfer + holder axes, window (cyan), dots (blue, numbered) + sample axes, status banner with reasons |
| `metadata.json` | always | COORD layout (below) |
| `transform.json` | only with a pose | the `transform` object on its own |

### `transform.json`

```json
{
  "algorithm_version": "spotvision-0.3.1",
  "matrix": [[0.99680, 0.01313], [-0.01313, 0.99680]],
  "rms_error_mm": 0.0036,
  "translation": [2.502, 3.073]
}
```

`holder_mm = matrix @ sample_mm + translation + [15.0, 62.5]`, with both vectors in (X, Y) mm.

* `translation`: the corner dot's position relative to the window's top-left corner (holder frame).
* `matrix`: its columns are the sample's +X and +Y axes expressed in holder mm. Following COORD, it is scaled by `plane_scale_ratio` (the sample plane's apparent px/mm over the holder's, from the far dots; typically 0.99–1.03 because the sample surface sits at a different height). Divide by `sqrt(det(matrix))` for a pure rotation.
* `rms_error_mm`: the L-frame residual (`sample_rms_px`) divided by px/mm.

### `metadata.json`

| key | source |
|---|---|
| `camera_model`, `camera_settings`, `instruction_no`, `run_id`, `sample_id`, `sample_igsn`, `timestamp` | copied from the input run's `metadata.json` if present; these are capture facts, and any old detection results in that file are ignored. Keys the input doesn't have are left out (`timestamp` is never invented). |
| `algorithm_version`, `elapsed_s`, `processed_at`, `image {dtype, shape}` | this run (`processed_at`: UTC ISO 8601 processing time) |
| `confidence`, `reasons` | see [confidence](#confidence-and-reasons) |
| `sample_window_origin_in_detected_holder_mm` | `[15.0, 62.5]` |
| `transform` | as above, `null` when refused |
| `diagnostics` | COORD's keys: `calibrated` (always false), `chamfer_confidence`, `dot_candidates`, `dot_polarity`, `holder_angle_deg`, `holder_aspect`, `holder_det`, `holder_long_px`, `holder_short_px`, `holder_px_per_mm`, `px_per_mm`, `px_per_mm_source` (`holder_edge`), `mirrored_rms_px`, `n_dots`, `perpendicularity_deg`, `plane_scale_ratio`, `relative_det`, `sample_det`, `sample_rms_px`; plus `holder_fit_rms_px`, `similarity_rms_px`, `window_groove_offset_mm` |
| `detection` | the full `Result`: every pixel coordinate, per-dot NCC, arm spacings, the holder mm→px matrix, the sample frame in px, … Left out with `--no-detection` / `include_detection=False`. |

Both files are strict JSON: values that are not finite (e.g. `sample_rms_px` when fewer than three dots are found) are written as `null`, never `NaN`. `build_metadata()` returns the same thing as plain Python (no numpy types), so it can go straight to an API such as Girder's item metadata:

```python
from spotvision import analyze, build_metadata
meta = build_metadata(analyze(im), im, elapsed_s, acquisition, include_detection=False)
gc.addMetadataToItem(item_id, meta)
```

`to_jsonable(value)` does the same conversion for any other value (e.g. `Result.transform`).

COORD's `photo_only` mode is not reproduced: every image is measured.

---

## Algorithm

All stages work on a percentile-normalised (1–99.5 %) image, so absolute exposure does not matter. Each stage writes its diagnostics into `Result`.

### 1. Holder (`holder.py`)

1. **Coarse:** the image is downsampled 4×. A sweep of 35 thresholds keeps the bright component that does not touch the image border and best matches the plate (fill > 0.85 of its min-area rectangle, aspect within 8 % of 120/70). The plate is always surrounded by the dark holder frame, so the bright table outside cannot merge with it.
2. **Sub-pixel edges:** along each side of the coarse box, 150 intensity profiles across the edge (±12 px, 0.5 px steps) give the strongest dark→bright step with parabolic peak refinement. A Huber line fit with outlier trimming ignores the notches and chamfer. The four line intersections are the corners.
3. **Chamfer:** a 4 mm triangle is probed at each corner. The corner darkest relative to the plate median is the chamfer, and the contrast to the next-darkest corner is reported as `chamfer_confidence`.
4. **Rigid fit:** a rotation + uniform scale + translation maps the nominal 120 × 70 mm rectangle onto the refined corners. This gives the holder frame, `px_per_mm`, and an exactly rectangular yellow box. `holder_fit_rms_px` is the corner residual.

### 2. Window (`window.py`)

The window is machined into the holder, so it is **placed** from the holder pose: 40 × 40 mm at (X 15, Y 62.5), always square to the holder. The dark groove around it is measured too, but only as a check on the holder pose (`window_groove_offset_mm`, median 0.26 mm on the dataset).

### 3. Dots (`dots.py`)

1. **Candidates:** a scale-normalised Laplacian-of-Gaussian at 3 scales (dot size varies between samples), computed separately for bright and dark blobs. The search covers the window padded by one dot radius, so dots half hidden under the window lip still count. A third, `mixed` set merges both polarities, because under glare some dots can be brighter and others darker than their surroundings.
2. **Hypotheses:** every ordered candidate pair, matched against every ordered template pair, defines a similarity transform. A hypothesis is kept only if it satisfies all of these priors:
   * scale within 12 % of the holder px/mm;
   * rotation within ±20° of the holder axes;
   * all five dots inside the window, allowing up to ¼ radius under the lip;
   * the corner dot within ±2 mm of (3.2, 3.2) mm from the window's top-left corner (measured on 125 clean runs: 1.75–4.3 mm).

   Hypotheses are scored mainly by the number of matched dots. The top 10 per polarity are kept, with non-maximum suppression.
3. **Verification:** each predicted dot is refined by a sub-pixel circle fit (48 radial edge rays, trimmed Kasa fit, radius within 0.65–1.5 × nominal). It is then scored by its **disc NCC**: the normalised cross-correlation of the patch with an ideal disc, after removing a planar background estimated from a surrounding ring (1.2–1.7 × the radius). This score does not depend on contrast or illumination gradients. Only sample surface is used: pixels within 0.8 mm of the window edge are masked out, because the dark gap and bright bevel there would otherwise corrupt the background of dots that sit close to the edge. On this data, random texture scores ≤ 0.45 (95th percentile) and real dots score 0.7–0.97.
   * A dot is **strong** if NCC ≥ 0.7, and **weak** if NCC ≥ 0.45 with ≥ 60 % of its edge rays on the circle.
   * A hypothesis needs **≥ 1 strong dot**, and either ≥ 2 strong dots or ≥ 4 dots in total. Weak dots therefore count only through geometric consensus.
   * Of the hypotheses that qualify, the one with the highest summed NCC wins.
4. **L-frame fit:** the origin and one angle, with the two arms orthogonal by construction. It minimises each dot's perpendicular distance to its own arm line. Spacing along the arms is deliberately free, because some samples are etched with the middle +Y dot at about 9.3 mm instead of 10.
5. **Guided pass:** missing dots are searched for where the fitted frame predicts them (within 0.4 radius), using the sample-plane scale.
6. **Pruning:** a dot whose distance from the corner is off nominal (beyond 12 % for the middle dots, 8 % for the far dots) is dropped first. Then any dot more than 3 px off its arm line is dropped. If the pose becomes underdetermined, there is no pose.
7. **Final checks:** the fitted corner must still satisfy the corner prior, and the RMS must be ≤ 3 px.

The pose needs at least one dot on each arm and at least 4 arm equations: for example the corner plus 2 dots, or 2 dots per arm without the corner.

### 4. Pose (`pipeline.py`)

The sample frame in pixels is mapped into holder mm through the holder pose. This produces the transform, `sample_origin_holder_mm` and `sample_angle_deg`.

## Confidence and reasons

| confidence | meaning |
|---|---|
| `clean` | pose found and nothing flagged |
| `degraded` | pose found, with at least one reason below; usable, but check it |
| `refused` | no pose; `transform` is `null` and `transform.json` is not written |

**Reasons that refuse:**

* `holder: no plate-shaped component`, `holder: edge refinement failed`
* `dots: no hypothesis with enough dot evidence`
* `dots: only N of 5 resolved, pose underdetermined`
* `dots: too few consistent dots left, no pose` (after pruning)
* `dots: fitted corner dot too far from window top-left, no pose`
* `dots: L-frame RMS … px > 3.0, no pose`

**Reasons that degrade:**

* `dots: N of 5 resolved`
* `dots: [i, …] weak (ncc < 0.7), accepted by geometry`
* `dots: dropped dot i (spacing off nominal | residual … px)`
* `dots: L-frame RMS … px > 1.0`
* `dots: arms … deg off perpendicular` (> 0.5°; only measured with ≥ 2 dots on each arm)
* `holder: chamfer ambiguous (contrast …)` (< 0.25)
* `holder: edges deviate from a 120x70 mm rectangle (rms … px)` (> 3 px)

## Tuning

Every nominal and threshold lives in `src/spotvision/constants.py`, with a comment on each. The most important ones:

| constant | value | effect |
|---|---|---|
| `HOLDER_X, HOLDER_Y` | 70, 120 mm | plate size |
| `WINDOW_X0, WINDOW_Y0, WINDOW_SIZE` | 15, 62.5, 40 mm | window placement |
| `DOT_TEMPLATE`, `DOT_RADIUS_MM` | 6/12 mm and 10/20 mm arms, r = 1.5 mm | dot pattern |
| `CORNER_DOT_IN_WINDOW_MM`, `CORNER_DOT_TOL_MM` | (3.2, 3.2), ±2 mm | where the corner dot may be |
| `MAX_SAMPLE_ROT_DEG` | 20° | sample rotation prior |
| `DOT_STRONG_NCC`, `DOT_WEAK_NCC` | 0.7, 0.45 | dot evidence; lowering these trades false refusals for false poses |
| `DOT_NCC_WINDOW_INSET_MM` | 0.8 mm | window-edge band excluded from the dot NCC |
| `MIN_STRONG`, `MIN_TOTAL_IF_WEAK` | 2, 4 | consensus rule |
| `SPACING_TOL` | 0 / 12 % / 8 % | allowed along-arm deviation per dot |
| `CLEAN_RMS_PX`, `CLEAN_PERP_DEG`, `MAX_RMS_PX` | 1.0 px, 0.5°, 3.0 px | clean vs. degraded vs. refused |

The thresholds were tuned by eye on the 216 captured runs. Re-check them against a labelled set before relying on them for new sample families.

---

## Validation

**Synthetic scenes** (`tests/synth.py`: anti-aliased, blurred, noisy, with known truth):

* holder corners < 1 px;
* px/mm within 0.1 %;
* dot centres < 0.1 px;
* transform within 0.015 mm and 0.06°.

These hold across holder rotations (including 180°), sample rotations up to 8°, bright and dark dots, an uneven middle dot, and a distractor grid of same-size discs.

**Captured runs (216):**

| | count |
|---|---|
| clean | 125 |
| degraded | 18 |
| refused | 73 |

* The holder was found in 213 of 216 runs.
* Speed is about 0.4 s per image on one core; all 216 take about 21 s with `-j 8`.
* All 143 poses were inspected on contact sheets. Every clean pose looked correct. No hand-labelled ground truth exists yet.
* Most refusals have no visible dots: dark JHBMAI samples, round JHACRA samples, rock-textured JHAMAL00020 samples, and dots washed out by glare.

**Comparison with the old COORD pipeline.** This is a cross-check, not ground truth.

* **Both clean (61 runs):** they agree within 0.11 mm, 0.16° and 0.003 in scale.
* **Better here, clean (17 runs):**
  * 16 runs where COORD gave no pose, mostly dark images where it lost the holder;
  * 1 run where COORD's pose was off by 45 mm.
* **Old pose wrong, correct here but flagged:** 2 runs (COORD reported 142° and 160° rotations).
* **Old pose wrong, refused here:** 39 runs of COORD "degraded" poses on rock texture, dotless or glare images.
* **Same pose, clean here instead of degraded:** 46 runs. These are the uneven-middle-dot samples (COORD residual ≈ 3.8 px).
* **Refused here, but COORD's pose looks right:** 1 run (TEST_0000_0_1_2026-11-03_15-17-16), with glare.
* **Degraded here, but the pose is correct:** runs where a visible dot is missed or scores just under NCC 0.7, mostly under glare (e.g. APLMAA00001-008, TEST20).

## Limitations

* **Glare:** on mirror-like samples the camera's own reflection appears as a dark disc, and other areas wash out to white. This hides dots and is the main cause of refusals. Better lighting fixes more than software can: cross-polarised or diffuse dome light, low-angle ring light for the etched dots, 12-bit capture, and more consistent exposure.
* **Dots partly hidden under the window lip** get biased centres (seen as an RMS of 1.9 px and arms 3.9° off perpendicular on TEST_0021) or are refused.
* **Uncalibrated:** px/mm comes from the holder edges. The plate measures 0.17 % off the nominal aspect (about 1.6 px corner residual), and the sample plane differs in scale by up to about 3 %. A calibration target would pin down both, as well as lens distortion.
* **Thresholds were tuned on this dataset only.** See [Tuning](#tuning).

## Annotating failing runs

`spotvision-annotate` shows each run the detector does not get clean, zoomed on the window, and records where the dots really are. The labels are the ground truth for tuning thresholds and for regression tests.

```bash
spotvision-annotate /data/runs -o labels.json   # refused + degraded runs not yet labelled
spotvision-annotate RUN_DIR ... --all           # include clean runs (to measure accuracy)
spotvision-annotate */ --relabel                # revisit runs that already have a label
```

Inputs are expanded like `spotvision`'s: a directory without `raw.png` stands for the run directories in it. It needs a display and tkinter (part of Python; on Debian/Ubuntu `apt install python3-tk`). No other dependencies.

**Per run, either:**

* place the 5 dot centres in template order: **0** corner dot, **1, 2** along the wider-spaced (10 mm) arm, **3, 4** along the closer-spaced (6 mm) arm. A dot you can't see: `x` or right-click. Then `Enter`.
* or press `n` if no dot is visible to you either; that confirms a refusal is right.

**Placing a dot accurately:** press and hold. A magnifier opens on the point with a circle of the nominal dot size and a crosshair; drag until the circle sits on the dot's edge all round, then release. Pressing on a placed dot picks it up again (it moves with the cursor without jumping to it). The arrow keys nudge the selected dot by 0.25 px (Shift: 1 px) with the magnifier shown, and the mouse wheel changes the circle radius to match the dots in front of you (it carries over to the next runs). Positions are stored to 0.01 px in `raw.png` pixel-centre coordinates (the convention the detector uses).

The detector's dots are shown as dashed orange `d0`–`d4` circles for reference; `a` copies them as a starting point (`u` removes dots from the end so they can be re-clicked). Other keys: `u` undo, `c` cycle contrast (stretch / CLAHE / raw, helps on dark or washed-out images), `f` full image, `h` click the 4 plate corners starting at the chamfer corner (only when the yellow holder box is wrong), `t` note, `Esc` close the magnifier, `s` skip, `b` back, `q` quit. `--size` caps the image size on screen.

**`labels.json`** is rewritten after every saved run, one run per line, with coordinates in full-resolution `raw.png` px (x, y):

```json
{
  "APLMAA00001-008_…": {"dots": [[1287.4, 722.9], null, [1596.9, 718.4], [1282.9, 817.1], [1281.4, 909.8]]},
  "JHBMAI00005-003_…": {"no_dots": true, "note": "dark sample"}
}
```

Optional keys: `holder` (4 corners) and `note`.

---

## Development

```bash
.venv/bin/python -m pytest -q            # 70 tests, ~10 s
.venv/bin/python -m pytest -m realdata   # only the captured-run regressions
```

| test file | covers |
|---|---|
| `test_geometry.py` | similarity fit, robust line, intersection, quad containment, frame conversions, sub-pixel peak |
| `test_dots.py` | circle refinement accuracy (both polarities), disc NCC (polarity, gradients, noise, window-edge masking), candidates, L-frame fit |
| `test_pipeline.py` | synthetic end-to-end accuracy; rotations; holder frame following the chamfer; dark dots; uneven middle dot; missing dot → degraded; no dots, dots away from the corner, mirrored pattern, no holder → refused; distractor grid; colour and 16-bit input |
| `test_outputs.py` | COORD key sets for metadata, diagnostics and transform; acquisition whitelist; `write_run`; CLI end-to-end, parallel, no overlay, unreadable input, input expansion |
| `test_annotate.py` | annotation tool without the GUI: pixel-centre view mapping, rendering, sub-pixel magnifier, press / drag / nudge, labels file |
| `test_realdata.py` | regression on captured runs; skipped when the run folders are absent |

**Layout:**

```
src/spotvision/
  constants.py   nominals and thresholds
  geometry.py    fitting / sampling helpers, holder mm <-> px
  holder.py      plate, edges, chamfer, rigid fit
  window.py      rigid window placement + groove check
  dots.py        candidates, hypotheses, verification, L-frame
  pipeline.py    analyze() and the transform
  result.py      Result dataclass, confidence
  overlay.py     draw()
  metadata.py    metadata.json / transform.json, write_run()
  cli.py         command line
  annotate.py    spotvision-annotate (Tk click tool)
tests/           pytest suite + synthetic scene renderer (synth.py)
```
