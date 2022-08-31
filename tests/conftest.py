"""Configuration for the pytest test suite."""
# pylint: disable=redefined-outer-name
from copy import deepcopy
from itertools import cycle
from itertools import islice
from os import devnull
from pathlib import Path
from subprocess import call

import numpy as np
import pytest
from voxcell import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from neurocollage.planes import create_planes
from neurocollage.planes import get_layer_annotation

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_path():
    """The path to the directory containing the tested data."""
    return DATA


def generate_small_O1(directory):
    """Dump a small O1 atlas in folder path."""
    # fmt: off
    with open(devnull, "w", encoding="utf-8") as f:
        call(
            [
                "brainbuilder", "atlases",
                "-n", "6,5,4,3,2,1",
                "-t", "200,100,100,100,100,200",
                "-d", "100",
                "-o", str(directory),
                "column",
                "-a", "1000",
            ],
            stdout=f,
            stderr=f,
        )
    # fmt: on
    return str(directory)


@pytest.fixture(scope="session")
def small_O1_path(tmpdir_factory):
    """Generate the atlas."""
    atlas_dir = str(tmpdir_factory.mktemp("atlas_small_O1"))
    generate_small_O1(atlas_dir)
    return {"atlas": atlas_dir, "structure": DATA / "region_structure.yaml"}


@pytest.fixture(scope="session")
def small_O1(small_O1_path):
    """Open the atlas."""
    return Atlas.open(small_O1_path)


@pytest.fixture(scope="function")
def cells_df(tmpdir, small_O1_path, layer_annotation):
    """Raw data for the cell collection."""
    return generate_cells_df(tmpdir, small_O1_path, layer_annotation)


def generate_cell_collection(cells_df):
    """The cell collection."""
    return CellCollection.from_dataframe(cells_df)


@pytest.fixture(scope="function")
def cell_collection(cells_df):
    """The cell collection."""
    return generate_cell_collection(cells_df)


@pytest.fixture(scope="function")
def layer_annotation(small_O1_path):
    """Layer annotation path."""
    return get_layer_annotation(small_O1_path)


@pytest.fixture(scope="function")
def planes(layer_annotation):
    """Planes."""
    return create_planes(layer_annotation, plane_count=5)[0]


def make_cell_density(small_O1_path, layer_annotation):
    """Create cell density file in atlas."""
    layer = "2"
    mtype = "L2_TPC:A"
    keys = [k + 1 for k, d in layer_annotation["mapping"].items() if d.endswith(layer)]
    density_annotation = deepcopy(layer_annotation["annotation"])
    density_annotation.raw = np.array(density_annotation.raw, dtype=float)
    density_annotation.raw[layer_annotation["annotation"].raw == keys[0]] = 1000
    density_annotation.raw[layer_annotation["annotation"].raw != keys[0]] = 0
    density_annotation.save_nrrd(f"{small_O1_path['atlas']}/[cell_density]{mtype}.nrrd")


def generate_cells_df(tmpdir, small_O1_path, layer_annotation):
    """Raw data for the cell collection."""
    make_cell_density(small_O1_path, layer_annotation)

    # fmt: off
    call(
        [
            "brainbuilder", "cells", "place",
            "--composition", str(DATA / "cell_composition.yaml"),
            "--mtype-taxonomy", str(DATA / "mtype_taxonomy.tsv"),
            "--atlas", small_O1_path['atlas'],
            "--output", tmpdir / "cells.h5",
        ],
    )
    # fmt: on

    df = CellCollection.load(tmpdir / "cells.h5").as_dataframe()
    morphs = ["C170797A-P1", "C280199C-P3", "C280998A-P3"]
    df["morphology"] = list(islice(cycle(morphs), len(df.index)))
    df["path"] = str(DATA / "input-cells") + "/" + df["morphology"] + ".h5"
    return df
