"""Setup for the NeuroCollage package."""
import importlib.util
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "neurocollage.version",
    "neurocollage/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

setup(
    name="neurocollage",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    version=VERSION,
    description="A tool to create 2D morphology collage plots based on matplotlib.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://bbpteam.epfl.ch/documentation/projects/NeuroCollage",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "git@bbpgitlab.epfl.ch:neuromath/NeuroCollage.git",
    },
    license="BBP-internal-confidential",
    install_requires=[
        "atlas_analysis>=0.0.3",
        "brainbuilder",
        "click>=8.1.3",
        "joblib",
        "matplotlib",
        "morph_tool",
        "neurom>=3.2",
        "NeuroTS>=3.1.1",
        "numpy>=1.23",
        "pandas>=1.4",
        "pyquaternion",
        "region_grower>=0.4.0",
        "scipy>=1.8",
        "tqdm>=4.6",
        "voxcell>=3.1.2",
        "docutils<0.19",  # related to https://github.com/CrossNox/m2r2/issues/52
    ],
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    extras_require={
        "docs": ["m2r2", "sphinx", "sphinx-bluebrain-theme", "sphinx-click"],
        "test": [
            "mock",
            "pytest",
            "pytest-click",
            "pytest-cov",
            "pytest-html",
        ],
    },
    entry_points={
        "console_scripts": ["neuro-collage=neurocollage.cli:main"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
