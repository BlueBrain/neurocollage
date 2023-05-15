"""Tes planes module."""
from pathlib import Path

from atlas_analysis.planes.planes import load_planes_centerline
from numpy.testing import assert_array_equal
from voxcell.nexus.voxelbrain import VoxelData

from neurocollage import planes as tested

DATA = Path(__file__).parent / "data"


def test_layer_annotation(small_O1_path):
    """Test layer_annotation."""
    layer_annotation = tested.get_layer_annotation(small_O1_path, "O0")

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

    expected_left_planes = load_planes_centerline(DATA / "expected_left_planes.npz")
    expected_center_planes = load_planes_centerline(DATA / "expected_center_planes.npz")
    expected_right_planes = load_planes_centerline(DATA / "expected_right_planes.npz")
    assert_array_equal(expected_left_planes["centerline"], centerline)
    for expected_left_plane, expected_center_plane, expected_right_plane, plane in zip(
        expected_left_planes["planes"],
        expected_center_planes["planes"],
        expected_right_planes["planes"],
        planes,
    ):
        assert_array_equal(expected_left_plane.to_numpy(), plane["left"].to_numpy())
        assert_array_equal(expected_center_plane.to_numpy(), plane["center"].to_numpy())
        assert_array_equal(expected_right_plane.to_numpy(), plane["right"].to_numpy())

    # test other parameters here
