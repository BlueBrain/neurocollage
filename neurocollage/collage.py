"""2D collage with matplotlib."""
import logging
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import neurom
import numpy as np
import trimesh
from joblib import Parallel
from joblib import delayed
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.resampling import resample_linear_density
from neurom.view import matplotlib_impl
from pyquaternion import Quaternion
from scipy.optimize import fmin
from sklearn.metrics.pairwise import pairwise_distances
from voxcell.exceptions import VoxcellError

from neurocollage.mesh_helper import MeshHelper
from neurocollage.planes import get_atlas
from neurocollage.planes import get_cells_between_planes
from neurocollage.utils import load_insitu_morphology

L = logging.getLogger(__name__)
matplotlib.use("Agg")


def get_plane_rotation_matrix(plane, current_rotation, target=None):
    """Get basis vectors best aligned to target direction.

    We define a direct orthonormal basis of the plane (e_1, e_2) such
    that || e_2 - target || is minimal. The positive axes along the
    vectors e_1  and e_2 correspond respectively to the horizontal and
    vertical dimensions of the image.

    Args:
        plane (atlas_analysis.plane.maths.Plane): plane object
        current_rotation (ndarray): rotation matrix at the location
        target (list): target vector to align each plane

    Return:
        np.ndarray: rotation matrix to map VoxelData coordinates to plane coordinates
    """
    if target is None:
        target = [0, 1, 0]
    target /= np.linalg.norm(target)

    current_direction = current_rotation.dot(target)

    rotation_matrix = plane.get_quaternion().rotation_matrix.T

    def _get_rot_matrix(angle):
        """Get rotation matrix for a given angle along [0, 0, 1]."""
        return Quaternion(axis=[0, 0, 1], angle=angle).rotation_matrix

    def _cost(angle):
        return 1 - (_get_rot_matrix(angle).dot(rotation_matrix).dot(current_direction)).dot(target)

    angle = fmin(_cost, 1.0, disp=False)[0]
    return _get_rot_matrix(angle).dot(rotation_matrix)


def _get_plane_grid(annotation, plane_origin, rotation_matrix, n_pixels):
    ids = np.vstack(np.where(annotation.raw > 0)).T
    pts = annotation.indices_to_positions(ids)
    pts = (pts - plane_origin).dot(rotation_matrix.T)
    pts_dot_x = pts.dot([1.0, 0.0, 0.0])
    pts_dot_y = pts.dot([0.0, 1.0, 0.0])
    x_min = pts[np.argmin(pts_dot_x)][0]
    x_max = pts[np.argmax(pts_dot_x)][0]
    y_min = pts[np.argmin(pts_dot_y)][1]
    y_max = pts[np.argmax(pts_dot_y)][1]

    xs_plane = np.linspace(x_min, x_max, n_pixels)
    ys_plane = np.linspace(y_min, y_max, n_pixels)
    X, Y = np.meshgrid(xs_plane, ys_plane)
    points = np.array(
        [
            np.array([x, y, 0]).dot(rotation_matrix) + plane_origin
            for x, y in zip(X.flatten(), Y.flatten())
        ]
    )
    return X, Y, points


def get_annotation_info(annotation, plane_origin, rotation_matrix, n_pixels=1024):
    """Get information to plot annotation on a plane.

    Args:
        annotation (VoxelData): atlas annotations
        plane_origin (np.ndarray): origin of plane (Plane.point)
        rotation_matrix (3*3 np.ndarray): rotation matrix to transform from real coordinates
            to plane coordinates
        n_pixels (int): number of pixel on each axis of the plane for plotting layers
    """
    X, Y, points = _get_plane_grid(annotation, plane_origin, rotation_matrix, n_pixels)
    data = annotation.lookup(points, outer_value=0).astype(float).reshape(n_pixels, n_pixels)
    return X, Y, data


def get_y_info(annotation, atlas, plane_origin, rotation_matrix, n_pixels=64):
    """Get direction of y axis on a grid on the atlas planes."""
    X, Y, points = _get_plane_grid(annotation, plane_origin, rotation_matrix, n_pixels)
    orientations = []
    for point in points:
        try:
            orientations.append(
                rotation_matrix.dot(np.dot(atlas.orientations.lookup(point)[0], [0.0, 1.0, 0.0]))
            )
        except VoxcellError:
            orientations.append([0.0, 1.0, 0.0])
    orientations = np.array(orientations)
    orientation_u = orientations[:, 0].reshape(n_pixels, n_pixels)
    orientation_v = orientations[:, 1].reshape(n_pixels, n_pixels)
    return X, Y, orientation_u, orientation_v


def _dpoint2pointcloud(X, i):
    """Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points

    Adapted from ripser.py.
    """
    ds = pairwise_distances(X, X[i, :][None, :]).flatten()
    ds[i] = 0
    return ds


def get_greedy_perm(X, sample):
    """Compute a furthest point sampling permutation of a set of points.

    Adapted from ripser.py
    """
    idx_perm = np.zeros(sample, dtype=np.int64)

    ds = _dpoint2pointcloud(X, 0)
    dperm2all = [ds]
    for i in range(1, sample):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        dperm2all.append(_dpoint2pointcloud(X, idx))
        ds = np.minimum(ds, dperm2all[-1])
    return idx_perm


# pylint: disable=too-many-locals,too-many-branches
def plot_cells(
    ax,
    cells_df,
    plane_left,
    plane_right,
    rotation_matrix=None,
    mtype=None,
    sample=10,
    plot_neuron_kwargs=None,
    linear_density=None,
    wire_plot=False,
    random=False,
):
    """Plot cells for collage."""
    if mtype is not None:
        cells_df = cells_df[cells_df.mtype == mtype]

    _plot_neuron_kwargs = {"realistic_diameters": True}
    if plot_neuron_kwargs is not None:
        _plot_neuron_kwargs.update(plot_neuron_kwargs)

    cells_df = get_cells_between_planes(cells_df, plane_left, plane_right)
    gids = []
    if len(cells_df.index) > 0:
        if sample is not None:
            sample = min(sample, len(cells_df.index))
        else:
            sample = len(cells_df.index)
        if random:
            cells_df = cells_df.sample(n=sample, random_state=42)
        else:
            ids = get_greedy_perm(cells_df[["x", "y", "z"]].to_numpy(), sample)
            cells_df = cells_df.iloc[ids]
        gids = cells_df.index

    def _wire_plot(morph, ax, lw=0.1):
        for sec in morph.sections:
            ax.plot(*sec.points.T[:2], c=matplotlib_impl.TREE_COLOR[sec.type], lw=lw)

    for gid in gids:
        m = load_insitu_morphology(cells_df, gid)

        if linear_density is not None:
            m = resample_linear_density(m, linear_density)
        morphology = neurom.core.Morphology(m)

        def _to_plane_coord(p):
            return np.dot(p - plane_left.point, rotation_matrix.T)

        # transform morphology in the plane coordinates
        morphology = morphology.transform(_to_plane_coord)
        if wire_plot:
            ax.scatter(
                *_to_plane_coord([cells_df.loc[gid, ["x", "y", "z"]].values])[0, :2], c="k", s=5
            )
            _wire_plot(morphology, ax=ax)
        else:
            matplotlib_impl.plot_morph(morphology, ax, plane="xy", **_plot_neuron_kwargs)


# pylint: disable=too-many-arguments,too-many-locals
def _plot_2d_collage(
    planes,
    layer_annotation,
    cells_df,
    atlas_path,
    mtype,
    sample,
    n_pixels,
    n_pixels_y,
    with_y_field,
    with_cells,
    plot_neuron_kwargs,
    figsize,
    cells_linear_density,
    cells_wire_plot,
    random,
):
    """Internal plot collage for multiprocessing."""
    plane_point = planes["center"].point
    atlas = get_atlas(atlas_path)

    rotation_matrix = get_plane_rotation_matrix(
        planes["center"], atlas.orientations.lookup(plane_point)[0]
    )

    X, Y, layers = get_annotation_info(
        layer_annotation["annotation"], plane_point, rotation_matrix, n_pixels
    )

    plane_point_left = planes["left"].point
    rotation_matrix_left = get_plane_rotation_matrix(
        planes["left"], atlas.orientations.lookup(plane_point_left)[0]
    )

    X_left, Y_left, layers_left = get_annotation_info(
        layer_annotation["annotation"], plane_point_left, rotation_matrix_left, n_pixels
    )

    plane_point_right = planes["right"].point
    rotation_matrix_right = get_plane_rotation_matrix(
        planes["right"], atlas.orientations.lookup(plane_point_right)[0]
    )

    X_right, Y_right, layers_right = get_annotation_info(
        layer_annotation["annotation"], plane_point_right, rotation_matrix_right, n_pixels
    )

    layer_ids = np.array([-1] + list(layer_annotation["mapping"].keys())) + 1

    cmap = matplotlib.colors.ListedColormap(
        ["0.8"] + [f"C{i}" for i in layer_annotation["mapping"]]
    )

    fig = plt.figure(figsize=figsize)
    plt.pcolormesh(X, Y, layers, shading="nearest", cmap=cmap, alpha=0.2)
    plt.contour(X_left, Y_left, layers_left, cmap=cmap, linestyles="dashed", linewidths=0.5)
    plt.contour(X_right, Y_right, layers_right, cmap=cmap, linestyles="dotted", linewidths=0.5)
    bounds = list(layer_ids - 0.5) + [layer_ids[-1] + 0.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar = plt.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        ticks=layer_ids,
        boundaries=bounds,
        shrink=0.3,
        alpha=0.2,
    )
    cbar.ax.set_yticklabels(["outside"] + list(layer_annotation["mapping"].values()))

    if with_cells:
        plot_cells(
            plt.gca(),
            cells_df,
            planes["left"],
            planes["right"],
            rotation_matrix=rotation_matrix,
            mtype=mtype,
            sample=sample,
            plot_neuron_kwargs=plot_neuron_kwargs,
            linear_density=cells_linear_density,
            wire_plot=cells_wire_plot,
            random=random,
        )
    if with_y_field:
        X_y, Y_y, orientation_u, orientation_v = get_y_info(
            layer_annotation["annotation"],
            atlas,
            plane_point,
            rotation_matrix,
            n_pixels_y,
        )
        arrow_len = abs(X_y[0, 1] - X_y[0, 0]) * 0.5
        plt.quiver(
            X_y,
            Y_y,
            orientation_u * arrow_len,
            orientation_v * arrow_len,
            width=0.0005,
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_rasterized(True)
    ax.set_title(f"plane coord: {plane_point}")
    plt.tight_layout()
    return fig


# pylint: disable=too-many-arguments
def plot_2d_collage(
    cells_df,
    planes,
    layer_annotation,
    atlas_path,
    mtype=None,
    pdf_filename="collage.pdf",
    sample=10,
    nb_jobs=-1,
    joblib_verbose=10,
    dpi=200,
    n_pixels=1000,
    with_y_field=True,
    n_pixels_y=20,
    plot_neuron_kwargs=None,
    with_cells=True,
    cells_linear_density=None,
    cells_wire_plot=False,
    figsize=(20, 20),
    random=False,
):
    """Plot collage of an mtype and a list of planes.

    Args:
        cells (cells): should contain location of soma and mtypes
        planes (list): list of plane objects from atlas_analysis
        layer_annotation (VoxelData): layer annotation on atlas
        mtype (str): mtype of cells to plot
        pdf_filename (str): pdf filename
        sample (int): maximum number of cells to plot
        nb_jobs (int): number of joblib workers
        joblib_verbose (int): verbose level of joblib
        dpi (int): dpi for pdf rendering (rasterized)
        n_pixels (int): number of pixels for plotting layers
        with_y_field (bool): plot y field
        n_pixels_y (int): number of pixels for plotting y field
        plot_neuron_kwargs (dict): dict given to ``neurom.viewer.plot_neuron`` as kwargs
        with_cells (bool): plot cells or not
        cells_linear_density (float): apply resampling to plot less points
        cells_wire_plot (bool): if true, do not use neurom.view, but plt.plot
        random (bool): randomly select cells if True, or select furthest away cells
    """
    Path(pdf_filename).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_filename) as pdf:
        f = partial(
            _plot_2d_collage,
            layer_annotation=layer_annotation,
            cells_df=cells_df,
            mtype=mtype,
            atlas_path=atlas_path,
            sample=sample,
            n_pixels=n_pixels,
            n_pixels_y=n_pixels_y,
            with_y_field=with_y_field,
            plot_neuron_kwargs=plot_neuron_kwargs,
            with_cells=with_cells,
            cells_linear_density=cells_linear_density,
            cells_wire_plot=cells_wire_plot,
            figsize=figsize,
            random=random,
        )
        for fig in Parallel(nb_jobs, verbose=joblib_verbose)(delayed(f)(plane) for plane in planes):
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)


def plot_3d_collage(
    cells_df, planes, layer_annotation, atlas_path, mtype, region, hemisphere, centerline, sample
):
    """Plot 3d collage with trimesh."""

    mesh_helper = MeshHelper(atlas_path, region, hemisphere)
    mesh_helper.layer_annotation = layer_annotation

    centerline_data = [trimesh.points.PointCloud(mesh_helper.positions_to_indices(centerline))]
    plane_data = [mesh_helper.load_planes(planes)]
    mesh_helper.render(data=plane_data + centerline_data)

    cells_df = cells_df[cells_df.mtype == mtype]
    for plane in planes:
        _cells_df = get_cells_between_planes(cells_df, plane["left"], plane["right"])
        _cells_df = _cells_df.sample(min(sample, len(_cells_df.index)))
        cell_data = mesh_helper.load_morphs(_cells_df)
        mesh_helper.render(data=cell_data, plane=plane)
    mesh_helper.show()
