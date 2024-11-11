"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import shutil
import tempfile
from importlib import metadata
from pathlib import Path

# -- Project information -----------------------------------------------------

project_name = "NeuroCollage"
package_name = "neurocollage"

# The short X.Y version
version = metadata.version(package_name)

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_click",
    "m2r2",
]

todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx-bluebrain-theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_theme_options = {
    "metadata_distribution": package_name,
}

html_title = project_name

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# autosummary settings
autosummary_generate = True

# autodoc settings
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


_README = str(Path(__file__).parent.parent.parent / "README.md")
_README_TMP = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
self_readme_copy_path = _README_TMP.name


def fix_readme(app, _):
    """Change links in the README that point to a file in the `doc/source` directory."""
    readme_file = _README
    shutil.copyfile(readme_file, app.config.self_readme_copy_path)
    with open(readme_file, "r", encoding="utf-8") as fp:
        content = fp.read()
    new_content = content.replace("doc/source/", "")
    with open(readme_file, "w", encoding="utf-8") as fp:
        fp.write(new_content)


def restore_readme(app, _):
    """Restore the original README file."""
    shutil.copyfile(app.config.self_readme_copy_path, _README)


def setup(app):
    """Setup sphinx hooks."""
    app.add_config_value("self_readme_copy_path", _README_TMP.name, "", ())
    app.connect("config-inited", fix_readme)
    app.connect("build-finished", restore_readme)
