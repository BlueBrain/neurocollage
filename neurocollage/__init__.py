"""neurocollage package.

A tool to create 2D morphology collage plots based on matplotlib.
"""
import importlib.metadata

from .collage import plot_2d_collage  # noqa: F401
from .collage import plot_3d_collage  # noqa: F401
from .planes import create_planes  # noqa: F401
from .planes import get_layer_annotation  # noqa: F401

__version__ = importlib.metadata.version("neurocollage")
