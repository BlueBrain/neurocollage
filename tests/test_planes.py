"""Tes planes module."""
from pathlib import Path

from atlas_analysis.planes.planes import load_planes_centerline

# from atlas_analysis.planes.planes import save_planes_centerline
from numpy.testing import assert_array_equal
from voxcell.nexus.voxelbrain import VoxelData

from neurocollage import planes as tested

DATA = Path(__file__).parent / "data"


def test_layer_annotation(small_O1_path):
    """Test layer_annotation."""
    layer_annotation = tested.get_layer_annotation(small_O1_path)

    assert layer_annotation["mapping"] == {
        0: "layer 1",
        1: "layer 2",
        2: "layer 3",
        3: "layer 4",
        4: "layer 5",
        5: "layer 6",
    }
    # layer_annotation.save_nrrd(str(DATA / "expected_layer_annotation.nrrd"))
    expected_layer_annotation = VoxelData.load_nrrd(str(DATA / "expected_layer_annotation.nrrd"))
    assert_array_equal(expected_layer_annotation.raw, layer_annotation["annotation"].raw)


def test_create_planes(layer_annotation):
    """Test create_planes."""
    planes, centerline = tested.create_planes(layer_annotation)

    # save_planes_centerline(DATA / "expected_planes", planes, centerline)
    expected_planes = load_planes_centerline(DATA / "expected_planes.npz")
    assert_array_equal(expected_planes["centerline"], centerline)
    for expected_plane, plane in zip(expected_planes["planes"], planes):
        assert_array_equal(expected_plane.to_numpy(), plane.to_numpy())

    # test other parameters here
