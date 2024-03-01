"""Module used to create planes according to a user defined main axis."""

import itertools

import networkx
import numpy as np
import voxcell
import vtk
from geomdl import BSpline
from geomdl import utilities
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.spatial.distance import cdist

from neurocollage.planes_utils.maths import Plane
from neurocollage.planes_utils.maths import normalize_vectors

# pylint: disable=broad-exception-raised
X, Y, Z = 0, 1, 2
XVECTOR = np.array([1, 0, 0])
YVECTOR = np.array([0, 1, 0])
ZVECTOR = np.array([0, 0, 1])
XYZ = slice(0, 3)
QUAT = slice(3, None)
NORMAL = QUAT
CANONICAL = np.array([XVECTOR, YVECTOR, ZVECTOR])


def convert_points_to_vtk(points):
    """Convert points from numpy array to vtk points.

    Args:
        points (np.array([[x1,y1,z1], ..., [x2,y2,z2]])): the points to convert.

    Returns:
        Points using the vtk format (a vtkPoints object)
    """
    if isinstance(points, vtk.vtkPoints):  # pylint: disable=no-member
        return points
    vtk_points = vtk.vtkPoints()  # pylint: disable=no-member
    for point in points:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])
    return vtk_points


def create_vtk_spline(points):
    """Create a vtkParametricSpline using points as inputs."""
    spline = vtk.vtkParametricSpline()  # pylint: disable=no-member
    spline.SetPoints(convert_points_to_vtk(points))
    spline.ParameterizeByLengthOn()
    spline.ClosedOff()
    return spline


def indices_to_voxel_centers(voxel_data, idx):
    """Indices to middle of voxel positions."""
    return voxel_data.indices_to_positions(idx + 0.5)


def pairwise(iterable):
    """Do v -> (v0,v1), (v1,v2), (v2, v3), ..."""
    v1, v2 = itertools.tee(iterable)
    next(v2, None)
    return zip(v1, v2)


def add_interpolated_planes(planes, inter_plane_count):
    """Densify a plane list adding inter_plane_count between pairs of consecutive planes.

    Args:
        planes: the list of Plane objects.
        inter_plane_count: number of planes to add between two consecutive planes (int)

    Returns:
        list of Plane objects.
    """
    if inter_plane_count <= 0:
        return planes

    new_planes = []
    first = 0

    # iterate over pairs of planes (plane --> (plane1, plane2), (plane2, plane3), ...)
    for plane1, plane2 in pairwise(planes):
        locations = np.vstack(
            [
                np.linspace(
                    plane1.point[axis],
                    plane2.point[axis],
                    inter_plane_count + 2,
                    endpoint=True,
                )
                for axis in (X, Y, Z)
            ]
        ).T

        # Interpolate plane normals
        weights = np.linspace(0.0, 1.0, inter_plane_count + 2).reshape((1, inter_plane_count + 2))
        normals = np.matmul(plane1.normal.reshape((3, 1)), 1.0 - weights) + np.matmul(
            plane2.normal.reshape((3, 1)), weights
        )
        normals = normalize_vectors(normals.T)

        for location, normal in zip(locations[first:], normals[first:]):
            new_planes.append(Plane(location, normal))
        first = 1

    return new_planes


def save_planes_centerline(filepath, planes, centerline, plane_format="quaternion"):
    """Save the centerline and its orthogonal planes to file.

    Args:
        filepath: name of the output path
        planes: the planes orthogonal to the centerline to be saved under the form of a float
            array of shape (N, 7), i.e., a sequence of N planes either under the quaternionic format
             [x, y, z, a, b, c, d] or the point-normal format [x, y, z, A, B, C]. See planes.maths.
        centerline: the centerline under the form of a float array of shape (M, 3), i.e., a
            sequence of M points (x, y, z).
        plane_format: format of the out planes, either 'quaternion' or 'point_normal'.
            Defaults to 'quaternion'.

    Returns:
        filepath(str): file path where the data have been saved.

    Notes:
        The file is a npz file used for visualization and the creation of a curved coordinate
         system, see the coordinates module. It contains all the planes and the centerline points.
         It contains 2 mandatory keys: 'centerline' and 'planes' whose corresponding values hold
         the relevant numpy arrays. A third optional key, namely 'plane_format', indicates whether
         planes are represented the quaternionic (default) or point-normal format.
    """
    if plane_format not in ["quaternion", "point_normal"]:
        raise TypeError(
            f"Unknown plane format {plane_format}." " Expected: 'point_normal' or 'quaternion'"
        )

    if plane_format == "point_normal":
        planes = np.array([plane.to_numpy() for plane in planes])
    else:
        planes = np.array(
            [np.concatenate([plane.point, plane.get_quaternion().elements]) for plane in planes]
        )
    centerline = np.asarray(centerline)
    np.savez(filepath, planes=planes, centerline=centerline, plane_format=plane_format)
    return filepath


def load_planes_centerline(filepath):
    """Loads the centerline and planes file you want to access.

    Note: the expected layout of the input npz file is:

    {
        'plane_format': <str or None>
        'centerline': <nump.ndarray of shape (M, 3)>
        'planes': <numpy.ndarray of shape (N, 7) for quaternionic or (N, 6) for point-normal format>
    }
    The value of 'planes' is a sequence of planes which are either in the quaternionic or
    or the point-normal format. See planes.maths.

    Args:
        filepath: the path to the npz file

    Returns:
        return a dict of the following form:
        {
            'plane_format': <str>
            'centerline': <nump.ndarray of shape (M, 3)>
            'planes': <list of Plane objects>
        }


    Raises:
        TypeError: if the input value of 'plane_format' is defined but is neither
        'point_normal' nor 'quaternion'.
    """
    res = dict(np.load(filepath, allow_pickle=True))  # The loaded data are read-only.

    assert "centerline" in res, "Missing mandatory 'centerline' field"
    assert "planes" in res, "Missing mandatory 'planes' field"

    # If the plane format is not specified, we assume that the quaternionic format is in use.
    # This is done for backward compatibility.
    res["plane_format"] = res.get("plane_format", "quaternion")
    if res["plane_format"] == "point_normal":
        res["planes"] = [Plane(npz_plane[XYZ], npz_plane[NORMAL]) for npz_plane in res["planes"]]
    elif res["plane_format"] == "quaternion":
        res["planes"] = [
            Plane.from_quaternion(npz_plane[XYZ], npz_plane[QUAT], reference_vector=ZVECTOR)
            for npz_plane in res["planes"]
        ]
    else:
        raise TypeError(
            f"Unknown plane format {res['plane_format']}. Expected: 'point_normal' or 'quaternion'"
        )

    return res


def _distance_transform(voxeldata):
    """Compute a 3d euclidean distance transform.

    Args:
        voxeldata: a VoxelData object containing the volume

    Returns:
        an array with the same dimensions as voxeldata input containing the distance transform

    Notes:
        https://en.wikipedia.org/wiki/Distance_transform
    """
    # 0 outside volume and 1 inside
    dist = np.zeros_like(voxeldata.raw)
    dist[np.nonzero(voxeldata.raw)] = 1
    dist = distance_transform_edt(dist)
    return dist


def _chain(to_evaluate, proposal, chain_length, dist, downhill, nrun, starting_points):
    """Create a chain to evaluate the local maximum of the to_evaluate function."""
    chain = []
    # start from one side then the other
    init = starting_points[nrun % len(starting_points)]
    current_idx = init
    current_val = dist[tuple(current_idx)]
    # distance transform is 0 outside the volume so the user can provide a voxel with
    # distance_transform(voxel) == 0. So need to epsilon this for the first step.
    current_val = current_val if current_val != 0 else 0.00001
    # launch the chain
    for _ in range(chain_length):
        test_idx = proposal(current_idx)
        test_value = to_evaluate(tuple(test_idx))

        # will always go uphill if proposed but sometimes downhill if ratio is not too small
        if test_value / current_val >= downhill:
            current_val = test_value
            current_idx = test_idx
            chain.append(current_idx)
    return chain


def _explore_ridge(
    dist,
    starting_points,
    downhill=0.95,
    chain_length=100000,
    chain_count=5,
    sampling=10,
    proposal_step=3,
):
    """Detect points inside the valley of the 3d distance transform between two points.

    The goal is to find a sample of points that represent the volume's centerline that goes from
    starting_points[0] to starting_points[1]. We choose the centerline definition as: where the
    distance transform is locally maximal inside the volume. We are dealing with non-convex volumes
    with complex shapes and there is no easy way to detect the correct local max of the
    distance transform. So the idea is to propagate a stochastic chain that will follow the
    ridge and go close to the starting points and hopefully converge at some point in the volume.

    Args:
        dist: the distance transform of the voxeldata
        starting_points: the starting points of the chain in the indices representation
        downhill: the ratio of at the step i+1: distance_transform(i + 1)/distance_transform(i)
            that authorizes the chain to go downhill
        chain_length: the length of each chain (including the not selected proposals)
        sampling: sampling at the end of the chain creation
        chain_count: the number of chains that you want to run for each starting points
        proposal_step: the allowed mean distance in the indices representation for the gaussian
        proposal function

    Returns:
        an [N x 3] array of point's indexes.
    """
    # pylint: disable=too-many-locals
    dim = len(dist.shape)
    downhill = min(1.0, downhill)

    def _proposal(iidx):
        """The proposal for the next step of chain in the indices representation.

        The proposal is a normal law centered on the current point index. The standard deviation
        is proposal_step. Retries if outside the voxeldata shape.

        Args:
            iidx: the current index

        Returns:
            the proposed future index.
        """
        next_idx = np.asarray([-1] * dim)
        while np.any(~((next_idx > 0) & (next_idx < dist.shape))):
            next_idx = (iidx + np.random.normal(0, proposal_step, size=3)).astype(int)
        return next_idx

    def _dist_evaluation(index):
        """Evaluation of the distance transform."""
        return dist[index]

    all_chains = []
    # launch multiple chains to scan more efficiently the distance transform ridges
    for nrun in range(chain_count * len(starting_points)):
        chain = _chain(
            _dist_evaluation,
            _proposal,
            chain_length,
            dist,
            downhill,
            nrun,
            starting_points,
        )
        all_chains.extend(chain[::sampling])
    return np.array(all_chains)


def _clusterize_cloud(cloud, max_length=25):
    """Clusterize all points that are close enough. Returns the mean of each cluster.

    Args:
        cloud: a cloud of points positions
        max_length: the distance from a point to define neighbors and to create a cluster

    Returns:
        [N x 3] array positions of mean clusters positions.

    Notes:
         there is no overlap between clusters. If neighbors are [1,2,3], [3,4,5]
         and [4,5] then the clusters will be [1,2,3] and [4,5]. This implies that points
         are in one cluster only. It implies that the algorithm is not stable if we start from a
         different point.
    """
    clustered = []
    cloud = np.asarray(cloud)
    tree = cKDTree(cloud)  # pylint: disable=not-callable
    to_skip = set()
    neighbors_groups = tree.query_ball_point(cloud, max_length)
    for neighbors in neighbors_groups:
        if not set(neighbors).intersection(to_skip):
            neighbor_positions = cloud[np.array(neighbors)]
            clustered.append(np.mean(neighbor_positions, axis=0))
        to_skip.update(neighbors)
    return clustered


def _create_graph(cloud, link_distance=500):  # pylint: disable=too-many-locals
    """Create a graph from a could of point's positions.

    Args:
        cloud: the could of point's positions. [N*3] array.
        link_distance: the distance from which you consider 2 points as connected neighbors

    Returns:
        a networkx graph object containing all points from cloud.

    Notes:
        Nodes are considered as neighbors if the distance between them is below link_distance.
        The graph must be composed of one connected component at the end of the process. So if the
        graph is composed of multiple connected components, they are linked altogether recursively
        by finding the 2 closest points for each pair of connected component and a edge is created
        between these points. This process is repeated until only one connected component remains.
    """
    graph = networkx.Graph()
    cloud = np.asarray(cloud)

    tree = cKDTree(cloud)  # pylint: disable=not-callable

    for node_id, point in enumerate(cloud):
        indices = tree.query_ball_point(point, link_distance)
        # need to add node in the graph in case it has no neighbors
        graph.add_node(node_id)
        # first index is always self
        for idx in indices[1:]:
            graph.add_edge(node_id, idx)

    # want a unique connected component. Recursively link the closest connected components
    while True:
        connected_comp = list(networkx.connected_components(graph))
        if len(connected_comp) == 1:
            break
        distance_min = np.inf
        closest_node_1 = None
        closest_node_2 = None
        # find the min of distance matrices between all connected component element pairs
        for connected_comp_1, connected_comp_2 in itertools.combinations(connected_comp, 2):
            tmp_a1, tmp_a2 = (
                np.array(list(connected_comp_1)),
                np.array(list(connected_comp_2)),
            )
            euclidean_dists = cdist(cloud[tmp_a1], cloud[tmp_a2], "euclidean")
            # idx of the distance matrix min between c1 and c2
            tmp_d_idx = np.unravel_index(np.argmin(euclidean_dists), euclidean_dists.shape)
            if euclidean_dists[tmp_d_idx] < distance_min:
                distance_min = euclidean_dists[tmp_d_idx]
                closest_node_1 = tmp_a1[tmp_d_idx[0]]
                closest_node_2 = tmp_a2[tmp_d_idx[1]]
        # link the 2 closest connected components using the 2 closest points of these components
        graph.add_edge(closest_node_1, closest_node_2)
    return graph


def create_centerline(
    voxeldata,
    starting_points,
    link_distance=500,
    downhill=0.95,
    chain_length=100000,
    chain_count=5,
    sampling=10,
):
    """Create the centerline using a distance transform.

    Args:
        voxeldata: the considered volume
        starting_points: the starting points of the chain in the indices representation.
            Only two points are allowed for the moment. The choice of the order between
            first and second starting points will define the direction of plane orientations
        link_distance: the distance from which you consider 2 points as connected neighbors
        downhill: the ratio of loc/proposal that authorize the chain to go downhill
        chain_length: the length of each chain (including the not selected proposals)
        chain_count: the number of chain that you want to run for each starting points
        sampling: sampling at the end of the chain creation

    Returns:
        a list of points corresponding to the shortest path between both starting points
    """
    if len(starting_points) != 2:
        raise Exception("Only 2 starting points are allowed for the moment")

    starting_points = np.asarray(starting_points)
    dist = _distance_transform(voxeldata)
    ridge_pos = indices_to_voxel_centers(
        voxeldata,
        _explore_ridge(
            dist,
            starting_points,
            downhill=downhill,
            chain_length=chain_length,
            chain_count=chain_count,
            sampling=sampling,
        ),
    )
    clustered = _clusterize_cloud(ridge_pos, np.max(voxeldata.voxel_dimensions))
    clustered = np.concatenate(
        [
            [indices_to_voxel_centers(voxeldata, starting_points[0])],
            clustered,
            [indices_to_voxel_centers(voxeldata, starting_points[1])],
        ]
    )

    graph = _create_graph(clustered, link_distance=link_distance)

    def _weight(node_id1, node_id2, _):
        """Weight function for the dijkstra. Must keep the _ in signature."""
        return np.linalg.norm(clustered[node_id1] - clustered[node_id2])

    A = networkx.dijkstra_path(graph, 0, len(clustered) - 1, weight=_weight)
    return clustered[np.array(A)]


def _split_path(path, point_count):
    """Splits the path in equal length segments.

    Args:
        path(list/np.array): a list of 3d points
        point_count: the number of points to sample the path

    Returns:
        np.array: with equidistant points along the path
    """
    path = np.asarray(path)
    # distances between point i and i + 1
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_distances = np.cumsum([0] + distances.tolist())
    path_distances = np.linspace(0, cum_distances[-1], point_count, endpoint=True)[1:-1]
    # binning of path distance on cum_distance
    indices = np.digitize(path_distances, cum_distances) - 1
    # compute the remaining distance ratios on the segment [i, i+1] for all path distances
    ratios = (path_distances - cum_distances[indices]) / distances[indices]
    points = np.zeros((point_count, 3), dtype=np.float32)
    points[0] = path[0]
    points[-1] = path[-1]
    points[1:-1] = path[indices] + np.multiply((path[indices + 1] - path[indices]).T, ratios).T
    return points


def _smoothing(path, ctrl_point_count=10):
    """Use a bezier curve to smooth a path defined by multiple 3d points."""
    curve = BSpline.Curve()
    curve.degree = 5
    curve.ctrlpts = _split_path(path, ctrl_point_count).tolist()
    # Auto-generate knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)
    print(curve.knotvector)
    curve.delta = 0.01

    step_count = 100
    steps = np.linspace(0, 1, step_count, endpoint=True)
    return np.asarray(curve.evaluate_list(steps))


def create_planes(
    centerline,
    plane_count=25,
    delta=0.001,
    steps=None,
):  # pylint: disable=too-many-locals
    """Returns the plane quaternions and positions.

    We need to recreate a spline on top of the bezier curve to have a proper parametric function
    with the correct distance between planes. To create the planes we use the tangent to the
    centerline and use this as a Z (longitudinal) orientation and compute the quaternion.

    Args:
        centerline: a ndarray(N, 3) representing the centerline
        plane_count: the number of planes to return
        delta: the parametric delta to define the tangent
        steps: if not None, a 1d ndarray in [0, 1] will be used instead of plane_count equally
            spaced steps, and plane_count will be set to length of this array

    Returns:
        a list of plane_count x Plane([x, y, z], [A, B, C])
    """
    spline = create_vtk_spline(centerline)

    if steps is None:
        steps = np.linspace(0, 1, plane_count, endpoint=True)
    else:
        plane_count = len(steps)

    sampling_up = np.zeros((plane_count, 3), dtype=float)
    sampling_down = np.zeros((plane_count, 3), dtype=float)

    pointu = [0, 0, 0]
    pointd = [0, 0, 0]
    d = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ups = np.zeros((len(steps), 3))
    ups[:, 0] = steps + delta
    ups[ups[:, 0] > 1, 0] = 1
    downs = np.zeros((len(steps), 3))
    downs[:, 0] = steps
    downs[downs[:, 0] + delta > 1, 0] -= delta
    for i in range(len(steps)):
        spline.Evaluate(downs[i], pointd, d)
        sampling_down[i] = pointd
        spline.Evaluate(ups[i], pointu, d)
        sampling_up[i] = pointu

    unit_longs = normalize_vectors(sampling_up - sampling_down)
    planes = []
    for unit_long, pos in zip(unit_longs, sampling_up):
        planes.append(Plane(pos, unit_long))

    return planes


def create_centerline_planes(
    nrrd_path,
    output_path,
    starting_points,
    downhill=0.95,
    chain_length=100000,
    chain_count=5,
    sampling=10,
    link_distance=500,
    ctrl_point_count=10,
    plane_count=25,
    seed=41,
    plane_format="quaternion",
):
    """Create a centerline and orthogonal planes from a volume.

    Args:
        nrrd_path: the considered volume path
        starting_points: the starting points of the chain in the indices representation.
            Only two points are allowed for the moment. The choose of the order between
            first and second starting points will define the sense of plane orientations
        output_path: the output for the results (a npz file) with the planes and the centerline
        downhill: the ratio of loc/proposal that authorize the chain to go downhill
        chain_length: the length of each chain (including the not selected proposals)
        sampling: sampling at the end of the chain creation
        chain_count: the number of chain that you want to run for each starting points
        link_distance: the distance from which you consider 2 points as connected neighbors
        ctrl_point_count: the number of control points for the bezier smoothing
        plane_count: the number of planes to return
        seed: the seed for the pseudo random generator
        plane_format(str): either 'quaternion' for output planes P of the form [x, y, z, a, b ,c, d]
             or 'point_normal' format for output planes of the form [x, y, z, A, B, C].
            If `plane_format` is 'quaternion', then (x, y, z) represents the 3D coordinates of the
             intersection of the centerline with the plane P. The (a, b, c, d)-part is a unit
            quaternion q complying with the convention 'w, x, y, z'. The quaternion
             q = w + x * i + y * j + z * k such that q maps ZVECTOR = (0, 0, 1) to a normal vector
             of the plane, i.e., qkq^{-1} = n_x * i + n_y * j + n_z * k where (n_x, n_y, n_z) is a
             normal unit vector of P.
             (See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).
            If `plane_format` is 'point_normal', then (x, y, z) represents the 3D coordinates of the
             intersection of the centerline with the plane P while [A, B, C] is a unit normal vector
             of the plane.

    Notes:
        The algorithm will do the following steps (the related input variables are put
        between parenthesis):
          - compute the volume distance transform (nrrd_path)
          - use multiple stochastic chains to sample the distance transform distribution ridge
          ie: local minimums along the volume.
            - user defines two entry points for the volume. ie: the centerline will go through these
              points. (starting_points)
            - stochastic chains of a certain length (chain_length) are launched from these points.
              They will climb up the distance transform distribution with a high probability.
              With a small probability they will go downhill. This allows the chains to escape from
              local maximums 'hills' (downhill)
            - the volume can have multiple ridges and having multiple chains reduce the risk of
              missing some of them. (chain_count)
            - we sample the chain to reduce the autocorrelation of the chains (sampling)
          - the amount of points produced by the chains where the distance transform is high can
            be important so we do a clustering of points merging all the points that are in
            the range of voxel dimension.
          - we create a graph from these remaining points
            - nodes are point locations from the chains/clustering part
            - edges are created between close enough points (link_distance)
            - we need to find a path between the two starting points, this will define the
              centerline but it appears that the graph is not always connected. So we iteratively
              connect the graph by connecting the closest connected components first.
          - we do a shortest path research in the graph between the two starting points using a
            weighted Dijkstra. This creates the centerline.
          - the centerline usually jitter a lot and needs to be smoothed.
            - we use a bezier curve to do the smoothing with multiple control
              points (ctrl_point_count)
            - control points are taken from the shortest path points.
          - we create a parametric spline function on top of the bezier curve
          - derivative of this curve corresponds to the plane orientations (plane_count)
            - we take two points that are close on the spline. The vector between these points is
              define the normal of the local perpendicular plane.
          - save the results (output_path)
    """
    np.random.seed(seed)
    voxeldata = voxcell.VoxelData.load_nrrd(nrrd_path)
    centerline = create_centerline(
        voxeldata,
        starting_points,
        link_distance=link_distance,
        downhill=downhill,
        chain_length=chain_length,
        chain_count=chain_count,
        sampling=sampling,
    )
    centerline = _smoothing(centerline, ctrl_point_count=ctrl_point_count)
    # Returns planes under quaternionic format
    planes = create_planes(centerline, plane_count=plane_count)
    save_planes_centerline(output_path, planes, centerline, plane_format=plane_format)
