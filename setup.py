"""Setup for the neurocollage package."""
import importlib.util
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "neurocollage.version",
    "neurocollage/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

reqs = [
    "atlas_analysis>=0.0.4",
    "brainbuilder>=0.17",
    "bluepysnap>=0.13.1,<1",
    "click>=8",
    "joblib>=0.14",
    "matplotlib>=3.4",
    "morph_tool>=2.9",
    "neurom>=3.2",
    "numpy>=1.23",
    "pandas>=1.5",
    "pyglet>=2",
    "pyquaternion>=0.9.5",
    "region_grower>=1.0.0",
    "scipy>=1.8",
    "tqdm>=4.60",
    "trimesh>=3.6",
    "voxcell>=3.1.2",
]

doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-click",
]

test_reqs = [
    "mock>=3",
    "coverage>=6.5,<7",
    "pytest>=6",
    "pytest-click>=1",
    "pytest-cov>=3,<4",
    "pytest-html>=2",
]

setup(
    name="neurocollage",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    description="A tool to create 2D morphology collage plots based on matplotlib.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://bbpteam.epfl.ch/documentation/projects/neurocollage",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/neurocollage",
    },
    license="BBP-internal-confidential",
    packages=find_namespace_packages(include=["neurocollage*"]),
    python_requires=">=3.8",
    version=VERSION,
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "neurocollage=neurocollage.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
