import functools
import os

import pytest

from spotvision import analyze
from synth import render

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=None)
def _scene(key):
    im, truth = render(**dict(key))
    return im, truth, analyze(im)


@pytest.fixture
def scene():
    """scene(**render_kwargs) -> (image, truth, Result); cached across tests."""
    return lambda **kw: _scene(tuple(sorted(kw.items())))
