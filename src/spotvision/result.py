"""The per-image detection result."""
from dataclasses import dataclass, field

CLEAN, DEGRADED, REFUSED = 'clean', 'degraded', 'refused'


@dataclass
class Result:
    """Everything one image yielded. Pixel coordinates are (x, y) in the full image.

    A pose exists when ``ok`` is true. ``confidence`` is ``clean`` when it exists and
    nothing was flagged, ``degraded`` when it exists with reasons, else ``refused``."""
    ok: bool = False
    reasons: list = field(default_factory=list)
    # holder
    holder_corners: list = None      # px; order: origin, +X end (chamfer), opposite, +Y end
    holder_mm_to_px: list = None     # 3x3, maps holder (Y, X) mm -> px (rigid + scale)
    holder_px_per_mm: float = None
    holder_fit_rms_px: float = None  # refined corners vs. the fitted rigid rectangle
    holder_long_px: float = None     # refined edges, before the rigid-rectangle fit
    holder_short_px: float = None
    holder_angle_deg: float = None   # holder +X direction in the image, deg from image +x
    chamfer_px: list = None
    chamfer_contrast: float = None
    # window
    window_corners: list = None      # px; order matches holder_corners
    window_groove_offset_mm: float = None  # diagnostic: measured groove vs. placed window
    # dots (index 0 corner, 1-2 along +Y, 3-4 along +X; None if unresolved)
    dot_candidates: int = 0
    dot_polarity: str = None         # bright | dark | mixed
    dots_px: list = None
    dots_ncc: list = None            # disc NCC of each refined dot
    n_dots: int = 0
    dots_rms_px: float = None        # perpendicular residual of dots to the fitted L-frame
    perpendicularity_deg: float = None  # free-fit arm angle minus 90 (needs >=2 dots per arm)
    dot_edge_blur_px: float = None   # median Gaussian sigma of the dot edges (focus)
    arm_spacing_mm: list = None      # distance of each dot from the corner along its arm
    similarity_rms_px: float = None  # dots vs nominal template, best similarity
    mirrored_rms_px: float = None    # same with the opposite handedness (should be far worse)
    plane_scale_ratio: float = None  # sample-plane px/mm over holder px/mm (from far dots)
    # sample pose
    sample_frame_px: dict = None     # origin, x_axis, y_axis (unit vectors) in px
    sample_origin_holder_mm: list = None
    sample_angle_deg: float = None   # sample +X relative to holder +X
    transform: dict = None           # sample mm -> holder mm, translation rel. window origin

    @property
    def confidence(self):
        if not self.ok:
            return REFUSED
        return DEGRADED if self.reasons else CLEAN
