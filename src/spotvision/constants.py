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
HOLDER_EDGE_SEARCH_PX = 12                # +- around the coarse edge
HOLDER_MAX_FIT_RMS_PX = 3.0               # refined corners vs. rigid 120x70 rectangle
CHAMFER_PROBE_MM = 4.0                    # leg of the corner triangle probed for the chamfer
CHAMFER_MIN_CONTRAST = 0.25               # darkest vs. second-darkest corner probe

# ---- window (machined into the holder; placed rigidly from the holder pose) ---------
WINDOW_X0, WINDOW_Y0, WINDOW_SIZE = 15.0, 62.5, 40.0

# ---- fiducial dots -----------------------------------------------------------------
DOT_RADIUS_MM = 1.5
# sample frame (X, Y) mm; 0 = corner, 1,2 along +Y (10 mm pitch), 3,4 along +X (6 mm)
DOT_TEMPLATE = np.array([[0, 0], [0, 10], [0, 20], [6, 0], [12, 0]], float)
X_ARM, Y_ARM = (0, 3, 4), (0, 1, 2)       # dot indices on each arm (0 = corner, on both)
# corner dot sits near the window's top-left corner (measured on 117 clean runs: 2.0-4.3 mm)
CORNER_DOT_IN_WINDOW_MM, CORNER_DOT_TOL_MM = (3.2, 3.2), 2.0
MAX_SAMPLE_ROT_DEG = 20.0                 # sample rotation prior relative to holder axes
DOT_SCALE_TOL = 0.12                      # dot pitch vs. holder px/mm, fractional
DOT_MAX_CANDIDATES = 60                   # per polarity
DOT_LOG_SCALES = (0.8, 1.0, 1.25)         # LoG sigma = f * R / sqrt(2) (dot size varies)
DOT_RADIUS_RANGE = (0.65, 1.5)            # accepted fitted radius relative to nominal
N_HYPOTHESES = 10                         # per polarity, verified by refinement
# per-dot evidence: NCC against an ideal disc and circle-edge inlier fraction
DOT_STRONG_NCC = 0.7                      # random texture: p95 ~0.45, max ~0.78
DOT_WEAK_NCC, DOT_WEAK_FRAC = 0.45, 0.6   # weak dots only count via geometric consensus:
MIN_STRONG, MIN_TOTAL_IF_WEAK = 2, 4      # need >=2 strong dots, or >=4 dots (>=1 strong)
SPACING_TOL = (0.0, 0.12, 0.08, 0.12, 0.08)  # per dot, fraction of nominal distance from corner
DOT_DROP_RESIDUAL_PX = 3.0                # a dot this far off its arm line is dropped
MAX_RMS_PX = 3.0                          # L-frame residual beyond this -> no pose

# ---- confidence --------------------------------------------------------------------
CLEAN_RMS_PX, CLEAN_PERP_DEG = 1.0, 0.5
