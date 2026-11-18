"""Physical nominals (mm) and detector thresholds.

Frames
  Holder (mm): origin at the corner across the SHORT edge from the chamfer, +X along
               the short edge toward the chamfer, +Y along the long edge.
  Sample (mm): origin at the corner dot, +X along the closer-spaced (6 mm) arm,
               +Y along the wider-spaced (10 mm) arm.
Both are X-down / Y-right when the holder is upright with the chamfer bottom-left.
"""
import numpy as np

# ---- holder ------------------------------------------------------------------------
HOLDER_X, HOLDER_Y = 70.0, 120.0          # short side, long side
HOLDER_ASPECT = HOLDER_Y / HOLDER_X
HOLDER_ASPECT_TOL = 0.08                  # coarse search, fractional
HOLDER_MIN_FILL = 0.85                    # contour area / minAreaRect area
HOLDER_MIN_AREA_FRAC = 0.08               # of the downsampled image
HOLDER_COARSE_DOWNSAMPLE = 4
HOLDER_COARSE_BLUR_PX = 1.5               # Gaussian sigma on the downsampled image
HOLDER_THRESHOLDS = np.linspace(0.05, 0.9, 35)  # normalised-intensity sweep for the plate
HOLDER_ASPECT_WEIGHT = 3.0                # coarse score = fill - weight * rel. aspect error
HOLDER_REFINE_BLUR_PX = 2.0               # Gaussian sigma before sub-pixel edge refinement
HOLDER_EDGE_SEARCH_PX = 12                # +- around the coarse edge
HOLDER_MAX_FIT_RMS_PX = 3.0               # refined corners vs. rigid 120x70 rectangle
CHAMFER_PROBE_MM = 4.0                    # leg of the corner triangle probed for the chamfer
CHAMFER_MIN_CONTRAST = 0.25               # darkest vs. second-darkest corner probe

# ---- window (machined into the holder; placed rigidly from the holder pose) ---------
WINDOW_X0, WINDOW_Y0, WINDOW_SIZE = 15.0, 62.5, 40.0
# groove check only (window_groove_offset_mm); never moves the window
WINDOW_GROOVE_SEARCH_MM = 1.5             # +- around the placed window edge
WINDOW_GROOVE_SPAN = (0.12, 0.88)         # part of each side profiled (skips round corners)
WINDOW_GROOVE_MAX_TILT_DEG = 1.0

# ---- fiducial dots -----------------------------------------------------------------
DOT_RADIUS_MM = 1.5
# sample frame (X, Y) mm; 0 = corner, 1,2 along +Y (10 mm pitch), 3,4 along +X (6 mm)
DOT_TEMPLATE = np.array([[0, 0], [0, 10], [0, 20], [6, 0], [12, 0]], float)
X_ARM, Y_ARM = (0, 3, 4), (0, 1, 2)       # dot indices on each arm (0 = corner, on both)
# corner dot sits near the window's top-left corner (measured on 125 clean runs: 1.75-4.3 mm)
CORNER_DOT_IN_WINDOW_MM, CORNER_DOT_TOL_MM = (3.2, 3.2), 2.0
MAX_SAMPLE_ROT_DEG = 20.0                 # sample rotation prior relative to holder axes
DOT_SCALE_TOL = 0.12                      # dot pitch vs. holder px/mm, fractional
DOT_MAX_CANDIDATES = 60                   # per polarity
DOT_LOG_SCALES = (0.8, 1.0, 1.25)         # LoG sigma = f * R / sqrt(2) (dot size varies)
DOT_LOG_MIN_RESPONSE = 0.02               # scale-normalised LoG peak on the [0, 1] image
# hypotheses (lengths below in units of the nominal dot radius R)
DOT_MATCH_TOL_R = 0.5                     # candidate this close to a predicted dot = inlier
DOT_MATCH_STRENGTH_WEIGHT = 0.5           # score = inliers + weight * their LoG strength
DOT_WINDOW_OVERHANG_R = 0.25              # predicted dots may stick out under the lip
DOT_MAX_SHIFT_R = 0.8                     # refined dot vs. hypothesis prediction
DOT_GUIDED_MAX_SHIFT_R = 0.4              # same, in the guided pass (frame already fitted)
# sub-pixel circle refinement
DOT_RAY_RANGE_R = (0.4, 1.7)              # radial edge search along each ray
DOT_EDGE_MIN_PEAK = 0.25                  # ray edge kept if gradient > this * median peak
DOT_EDGE_PRIOR_R = 0.15                   # ray edge weighted by a Gaussian of this sigma
                                          # around the radius estimate (nominal, then fitted)
DOT_FIT_TRIM_PX, DOT_FIT_TRIM_FACTOR = 1.5, 2.5  # circle-fit inliers: resid < max(px, f * median)
DOT_MIN_EDGE_FRAC = 0.5                   # rays on the fitted circle, else no dot
DOT_RADIUS_RANGE = (0.65, 1.5)            # accepted fitted radius relative to nominal
N_HYPOTHESES = 10                         # per polarity, verified by refinement
# per-dot evidence: NCC against an ideal disc and circle-edge inlier fraction
DOT_NCC_RING_R = (1.2, 1.7)               # background ring (plane fit) around the disc
DOT_NCC_MIN_RING_FRAC = 0.25              # of that ring left after masking, else NCC = 0
DOT_STRONG_NCC = 0.7                      # random texture: p95 ~0.45, max ~0.78
DOT_WEAK_NCC, DOT_WEAK_FRAC = 0.45, 0.6   # weak dots only count via geometric consensus:
MIN_STRONG, MIN_TOTAL_IF_WEAK = 2, 4      # need >=2 strong dots, or >=4 dots (>=1 strong)
DOT_GUIDED_WEAK_NCC = 0.5                 # weak dot found by searching at a predicted spot,
                                          # where geometry no longer vouches for it
DOT_WEAK_MAX_OFF_PX = 1.5                 # weak dot vs. the strong dots' frame, when they fix it
SPACING_TOL = (0.0, 0.12, 0.08, 0.12, 0.08)  # per dot, fraction of nominal distance from corner
DOT_WINDOW_INSET_MM = 0.8                 # circle fit + NCC ignore the window-edge gap + bevel
DOT_DROP_RESIDUAL_PX = 3.0                # a dot this far off its arm line is dropped
DOT_EDGE_MAX_BLUR_PX = 1.5                # Gaussian sigma of the dot edges; in focus <= 1.3
                                          # (median 0.8), focus sweep 1.7-4.6 -> degraded
MAX_RMS_PX = 3.0                          # L-frame residual beyond this -> no pose

# ---- confidence --------------------------------------------------------------------
CLEAN_RMS_PX, CLEAN_PERP_DEG = 1.0, 0.5
