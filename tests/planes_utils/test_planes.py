"""
Unit tests of the planes module
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import networkx
import nose.tools as nt
import numpy as np
import numpy.testing as npt
import voxcell

import neurocollage.planes_utils.planes as tested
from neurocollage.planes_utils.maths import Plane

# pylint: disable=missing-function-docstring,protected-access


def create_rectangular_shape(length, width):
    raw = np.zeros((length, width, width))
    raw[1:-1, 1:-1, 1:-1] = 12
    return voxcell.VoxelData(raw, (10, 10, 10), (0, 0, 0))


def test_add_interpolated_planes():
    qv = 0.5
    planes = [
        Plane.from_quaternion([0, 0, 0], [qv, qv, qv, qv]),
        Plane.from_quaternion([1, 1, 1], [qv, qv, qv, qv]),
        Plane.from_quaternion([2, 2, 2], [qv, qv, qv, qv]),
    ]

    interplane_count = 2
    extended_planes = tested.add_interpolated_planes(planes, interplane_count)
    expected_length = interplane_count * (len(planes) - 1) + len(planes)
    nt.assert_equal(len(extended_planes), expected_length)

    scalars = np.diag(np.linspace(0, 2, num=expected_length))
    expected_points = np.matmul(scalars, np.ones((expected_length, 3)))
    quaternion = planes[0].get_quaternion().elements
    expected_quaternions = [quaternion] * expected_length

    npt.assert_allclose([plane.point for plane in extended_planes], expected_points)
    npt.assert_allclose(
        [plane.get_quaternion().elements for plane in extended_planes],
        expected_quaternions,
    )

    actual = tested.add_interpolated_planes(planes, 0)
    actual = [plane.to_numpy() for plane in actual]
    npt.assert_allclose(actual, [plane.to_numpy() for plane in planes])


def test_save_planes_centerline():
    planes = [
        Plane.from_quaternion([0, 0, 0], [1, 0, 1, 0]),
        Plane.from_quaternion([0, 0, 0], [1, 1, 0, 0]),
        Plane.from_quaternion([0, 0, 0], [1, 0, 0, 1]),
    ]
    centerline = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    with TemporaryDirectory() as directory:
        tested.save_planes_centerline(Path(directory, "test.npz"), planes, centerline)
        nt.ok_(Path(directory, "test.npz").exists())
        data = np.load(Path(directory, "test.npz"))
        assert "centerline" in data
        assert "planes" in data
        nt.assert_equal(data["plane_format"], "quaternion")

    with TemporaryDirectory() as directory:
        tested.save_planes_centerline(
            Path(directory, "test.npz"), planes, centerline, plane_format="point_normal"
        )
        nt.ok_(Path(directory, "test.npz").exists())
        data = np.load(Path(directory, "test.npz"))
        assert "centerline" in data
        assert "planes" in data
        nt.assert_equal(data["plane_format"], "point_normal")

    with nt.assert_raises(Exception):
        tested.save_planes_centerline(
            Path(directory, "test.npz"),
            planes,
            centerline,
            plane_format="standard_format",
        )


def test_load_planes_centerline():
    expected_planes = [
        Plane.from_quaternion([0, 0, 0], [1, 0, 1, 0]),
        Plane.from_quaternion([0, 0, 0], [1, 1, 0, 0]),
        Plane.from_quaternion([0, 0, 0], [1, 0, 0, 1]),
    ]
    expected_centerline = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    # Implicit quaternionic format (for backward compatibility)
    with TemporaryDirectory() as directory:
        filepath = tested.save_planes_centerline(
            Path(directory, "test.npz"), expected_planes, expected_centerline
        )
        res_planes = tested.load_planes_centerline(filepath)
        npt.assert_almost_equal(
            [plane.to_numpy() for plane in res_planes["planes"]],
            [plane.to_numpy() for plane in expected_planes],
        )
        npt.assert_almost_equal(res_planes["centerline"], expected_centerline)
        nt.assert_equal(res_planes["plane_format"], "quaternion")

    # Explicit quaternionic format
    with TemporaryDirectory() as directory:
        filepath = tested.save_planes_centerline(
            Path(directory, "test.npz"),
            expected_planes,
            expected_centerline,
            plane_format="quaternion",
        )
        res_planes = tested.load_planes_centerline(filepath)
        npt.assert_almost_equal(
            [plane.to_numpy() for plane in res_planes["planes"]],
            [plane.to_numpy() for plane in expected_planes],
        )
        npt.assert_almost_equal(res_planes["centerline"], expected_centerline)
        nt.assert_equal(res_planes["plane_format"], "quaternion")

    # Explicit equation format
    with TemporaryDirectory() as directory:
        filepath = tested.save_planes_centerline(
            Path(directory, "test.npz"),
            expected_planes,
            expected_centerline,
            plane_format="point_normal",
        )
        res_planes = tested.load_planes_centerline(filepath)
        npt.assert_almost_equal(
            [plane.to_numpy() for plane in res_planes["planes"]],
            [plane.to_numpy() for plane in expected_planes],
        )
        npt.assert_almost_equal(res_planes["centerline"], expected_centerline)
        nt.assert_equal(res_planes["plane_format"], "point_normal")

    # Wrong format
    with TemporaryDirectory() as directory:
        filepath = str(Path(directory, "test.npz"))
        np.savez(
            filepath,
            planes=expected_planes,
            centerline=expected_centerline,
            plane_format=np.array("mixed", dtype=str),
        )
        with nt.assert_raises(Exception):
            tested.load_planes_centerline(filepath)


def test__distance_transform():
    volume = create_rectangular_shape(7, 7)
    res = tested._distance_transform(volume)
    nt.assert_equal(res[3, 3, 3], 3)
    nt.assert_equal(res[2, 3, 3], 2)
    nt.assert_equal(res[1, 3, 3], 1)


def test__explore_valley():
    volume = create_rectangular_shape(30, 15)
    dist = tested._distance_transform(volume)

    for _ in range(10):
        res_points = tested._explore_ridge(
            dist,
            [[1, 7, 7], [29, 7, 7]],
            chain_length=10000,
            chain_count=2,
            proposal_step=1,
        )

        # not point outside the volume
        nt.assert_equal(len(res_points[res_points[:, 0] < 1]), 0)
        nt.assert_equal(len(res_points[res_points[:, 0] > 29]), 0)

        nt.assert_equal(len(res_points[res_points[:, 1] < 1]), 0)
        nt.assert_equal(len(res_points[res_points[:, 1] > 14]), 0)

        nt.assert_equal(len(res_points[res_points[:, 2] < 1]), 0)
        nt.assert_equal(len(res_points[res_points[:, 2] > 14]), 0)

        # when the chain reach stability the mean of y and z should be close to the
        # center of the volume 7 and 7 here
        y_mean_res = res_points.mean(axis=0)[1]
        z_mean_res = res_points.mean(axis=0)[2]
        tol = 1
        nt.ok_(abs(y_mean_res - 7) < tol)
        nt.ok_(abs(z_mean_res - 7) < tol)


def test__clusterize_cloud():
    points = [
        [250.0, 250.0, 250.0],
        [251.0, 251.0, 251.0],
        [150.0, 150.0, 150.0],
        [151.0, 151.0, 151.0],
        [152.0, 152.0, 152.0],
        [162.0, 152.0, 152.0],  # should be skipped
    ]

    res = tested._clusterize_cloud(points, max_length=10)
    nt.assert_equal(len(res), 2)
    expected = [[250.5, 250.5, 250.5], [151.0, 151.0, 151.0]]
    npt.assert_allclose(res, expected)


def test__create_graph():
    cloud = [
        [250.0, 250.0, 250.0],
        [250.0, 250.0, 250.0],
        [251.0, 251.0, 251.0],
        [1240.0, 1240.0, 1240.0],
        [1250.0, 1250.0, 1250.0],
        [2250.0, 2250.0, 2250.0],
        [3250.0, 3250.0, 3250.0],
    ]

    graph = tested._create_graph(cloud)
    nt.assert_equal(len(graph.nodes), len(cloud))
    connected_comp = list(networkx.connected_components(graph))
    nt.assert_equal(len(connected_comp), 1)


def test_create_centerline():
    volume = create_rectangular_shape(1000, 15)
    res = tested.create_centerline(
        volume, [[1, 7, 7], [999, 7, 7]], link_distance=2, chain_length=10000
    )
    # when the chain reach stability the mean of y and z should be close to the
    # center of the volume 75 and 75 here
    y_mean_res = res.mean(axis=0)[1]
    z_mean_res = res.mean(axis=0)[2]
    tol = 5
    nt.ok_(abs(y_mean_res - 75) < tol)
    nt.ok_(abs(z_mean_res - 75) < tol)
    with nt.assert_raises(Exception):
        tested.create_centerline(volume, [[1, 7, 7], [999, 7, 7], [1, 1, 1]])


def test_split_path():
    path = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [6, 0, 0], [10, 0, 0]]
    npt.assert_allclose(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        tested._split_path(path, 11),
    )

    np.random.seed(42)
    path = np.random.random((50, 3)) + np.repeat(np.arange(50), 3).reshape((50, 3))
    npt.assert_allclose(
        [
            [0.3745401203632355, 0.9507142901420593, 0.7319939136505127],
            [5.709218502044678, 5.71990442276001, 5.982579231262207],
            [11.592016220092773, 11.488036155700684, 11.344573020935059],
            [16.80758285522461, 16.557477951049805, 17.166072845458984],
            [22.063697814941406, 22.652379989624023, 22.022172927856445],
            [27.551259994506836, 27.261695861816406, 27.052141189575195],
            [32.786991119384766, 32.70025634765625, 32.75294494628906],
            [38.08820724487305, 38.5226936340332, 38.43498611450195],
            [43.70819091796875, 43.71569061279297, 43.77989959716797],
            [49.50267791748047, 49.05147933959961, 49.27864456176758],
        ],
        tested._split_path(path, 10),
    )


def test__smoothing():
    volume = create_rectangular_shape(1000, 15)
    centerline = tested.create_centerline(
        volume, [[1, 7, 7], [999, 7, 7]], link_distance=2, chain_length=10000
    )

    res = tested._smoothing(centerline)
    # when the chain reach stability the mean of y and z should be close to the
    # center of the volume 75 and 75 here
    y_mean_res = res.mean(axis=0)[1]
    z_mean_res = res.mean(axis=0)[2]
    tol = 5
    nt.ok_(abs(y_mean_res - 75) < tol)
    nt.ok_(abs(z_mean_res - 75) < tol)


def test_create_planes():
    volume = create_rectangular_shape(1000, 15)
    centerline = tested.create_centerline(
        volume, [[1, 7, 7], [999, 7, 7]], link_distance=2, chain_length=10000
    )
    centerline = tested._smoothing(centerline)
    res = tested.create_planes(centerline, plane_count=10)
    points = np.array([plane.point for plane in res])
    y_mean_res = points.mean(axis=0)[1]
    z_mean_res = points.mean(axis=0)[2]
    tol = 10
    nt.ok_(np.all(np.abs(y_mean_res - 75) < tol))
    nt.ok_(np.all(np.abs(z_mean_res - 75) < tol))
    # x position wise the first point is 1*10 + 10/2 last is 999*10 + 10/2
    # planes should be close these points
    nt.ok_(np.all(np.linspace(15, 9995, 10) - points[:, 0] < 10))

    # Difficult to predict the first correct value before finding the stability
    # remove the first and last quaternion then.
    res_q = [plane.get_quaternion().elements for plane in res[1:-1]]
    expected = np.asarray([0.7071067811865475, 0, 0.7071067811865475, 0])
    tols = [0.2, 0.2, 0.2, 0.2]
    nt.ok_(np.all(np.abs(res_q - expected) < tols))


def testcreate_centerline_planes():
    volume = create_rectangular_shape(1000, 15)
    with TemporaryDirectory() as directory:
        input_path = Path(directory, "data.nrrd")
        volume.save_nrrd(str(input_path))
        output_path = Path(directory, "res.npz")
        tested.create_centerline_planes(
            str(input_path),
            str(output_path),
            [[1, 7, 7], [999, 7, 7]],
            link_distance=2,
            chain_length=10000,
            plane_count=10,
        )
        nt.ok_(output_path.exists())

        res_planes = tested.load_planes_centerline(output_path)
        nt.assert_equal(res_planes["plane_format"], "quaternion")

        points = np.array([plane.point for plane in res_planes["planes"]])
        y_mean_res = points.mean(axis=0)[1]
        z_mean_res = points.mean(axis=0)[2]
        tol = 5
        nt.ok_(abs(y_mean_res - 75) < tol)
        nt.ok_(abs(z_mean_res - 75) < tol)
        # x position wise the first point is 1*10 + 10/2 last is 999*10 + 10/2
        # planes should be close these points
        nt.ok_(np.all(np.linspace(15, 9995, 10) - points[:, 0] < 5))

        # when the chain reach stability the mean of y and z should be close to the
        # center of the volume 75 and 75 here
        y_mean_res = res_planes["centerline"].mean(axis=0)[1]
        z_mean_res = res_planes["centerline"].mean(axis=0)[2]
        tol = 5
        nt.ok_(abs(y_mean_res - 75) < tol)
        nt.ok_(abs(z_mean_res - 75) < tol)

        # Difficult to predict the first correct value before finding the stability
        # remove the first and last quaternion then.
        res_q = [plane.get_quaternion().elements for plane in res_planes["planes"][1:-1]]
        expected = np.asarray([0.7071067811865475, 0, 0.7071067811865475, 0])
        tols = [0.2, 0.2, 0.2, 0.2]
        nt.ok_(np.all(np.abs(res_q - expected) < tols))


def testcreate_centerline_planes_with_point_normal_format():
    volume = create_rectangular_shape(1000, 15)
    with TemporaryDirectory() as directory:
        input_path = Path(directory, "data.nrrd")
        volume.save_nrrd(str(input_path))
        output_path = Path(directory, "res.npz")
        tested.create_centerline_planes(
            str(input_path),
            str(output_path),
            [[1, 7, 7], [999, 7, 7]],
            link_distance=2,
            chain_length=10000,
            plane_count=10,
            plane_format="point_normal",
        )
        nt.ok_(output_path.exists())

        res_planes = tested.load_planes_centerline(output_path)
        nt.assert_equal(res_planes["plane_format"], "point_normal")

        # Difficult to predict the first correct value before finding the stability
        # remove the first and last quaternion then
        res_q = [plane.get_quaternion().elements for plane in res_planes["planes"][1:-1]]
        expected = np.asarray([0.7071067811865475, 0, 0.7071067811865475, 0])
        tols = [0.2, 0.2, 0.2, 0.2]
        nt.ok_(np.all(np.abs(res_q - expected) < tols))
