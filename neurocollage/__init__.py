"""NeuroCollage package."""
import pkg_resources

from .collage import plot_collage  # noqa: F401
from .planes import create_planes  # noqa: F401
from .planes import get_layer_annotation  # noqa: F401

__version__ = pkg_resources.get_distribution("NeuroCollage").version
