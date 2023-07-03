"""3D collage module."""
from copy import deepcopy

import matplotlib
import numpy as np
import pyglet
import trimesh
from neurom import NeuriteType
from region_grower.atlas_helper import AtlasHelper
from trimesh.voxel import VoxelGrid
from voxcell.nexus.voxelbrain import Atlas

from neurocollage.planes import get_layer_annotation
from neurocollage.utils import load_insitu_morphology

# this is for .marching_cube.visual which exists
# pylint: disable=no-member


class MeshHelper(AtlasHelper):
    """Helper class to deal with meshes in atlas with trimesh."""

    def __init__(self, atlas_path, region, hemisphere=None):
        """Load atlas helper."""
        super().__init__(
            atlas=Atlas.open(atlas_path["atlas"]), region_structure_path=atlas_path["structure"]
        )
        self.atlas_path = atlas_path
        self.region = region
        self.hemisphere = hemisphere

        self._layer_annotation = None
        self._boundary_mask = None

    @property
    def annotation(self):
        """Returns the voxell data with layer annotation."""
        return deepcopy(self.layer_annotation["annotation"])

    @property
    def layer_annotation(self):
        """Returns a dict with layer annotation and layer mapping."""
        if self._layer_annotation is None:
            self._layer_annotation = get_layer_annotation(
                self.atlas_path, region=self.region, hemisphere=self.hemisphere
            )
        return self._layer_annotation

    @layer_annotation.setter
    def layer_annotation(self, layer_annotation):
        """Setter for layer_annotation."""
        self._layer_annotation = layer_annotation

    def get_boundary_mesh(self):
        """Create boundary mesh."""
        boundary_mesh = VoxelGrid(self.annotation.raw).marching_cubes
        boundary_mesh.visual.face_colors = [100, 100, 100, 100]
        return boundary_mesh

    def get_total_boundary_mesh(self):
        """Get entire boundary mesh where depth is defined."""
        data = self.depths[self.region].raw
        data[np.isnan(data)] = 0
        data[data > 0] = 1
        mesh = VoxelGrid(data).marching_cubes
        mesh.visual.face_colors = [100, 100, 100, 20]
        return mesh

    def get_pia_mesh(self, cutoff=3):
        """Get pia mesh."""
        data = self.depths[self.region].raw

        # ensures there is 0 in outer voxels for boundary detection
        data[:, :, 0] = 0
        data[:, :, -1] = 0
        data[:, 0, :] = 0
        data[:, -1, :] = 0
        data[0, :, :] = 0
        data[-1, :, :] = 0

        data[data > cutoff * np.mean(abs(self.depths[self.region].voxel_dimensions))] = 0
        data[np.isnan(data)] = 0

        mesh = self._get_mesh(VoxelGrid(data), self.boundary_mask)
        mesh.visual.face_colors = [0, 0, 255, 100]
        return mesh

    @staticmethod
    def _get_mesh(vg, mask=None):
        """Get a mesh."""
        mesh = vg.marching_cubes
        if mask is not None:
            tri_indices = vg.points_to_indices(mesh.triangles_center)
            mesh.update_faces(~mask[tuple(tri_indices.T)])
        return mesh

    @property
    def boundary_mask(self):
        """Get a mask of inner and boundary voxel of a region."""
        if self._boundary_mask is None:
            m = VoxelGrid(self.annotation.raw).matrix
            outer_vg = VoxelGrid(self.annotation.raw)
            outer_vg.encoding.data[m == 0] = -1000
            outer_vg.encoding.data[m > 0] = 0
            d1 = outer_vg.matrix
            d2 = outer_vg.matrix
            d1[:-1] += d2[1:]
            d1[1:] += d2[:-1]
            d1[:, :-1] += d2[:, 1:]
            d1[:, 1:] += d2[:, :-1]
            d1[:, :, :-1] += d2[:, :, 1:]
            d1[:, :, 1:] += d2[:, :, :-1]
            self._boundary_mask = d1 > 0
        return self._boundary_mask

    def get_layer_meshes(self, alpha=0.5, colors=None):
        """Get layer meshes."""
        if colors is None:
            colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
        meshes = []
        for i, layer in enumerate(np.unique(self.annotation.raw)):
            if layer > 0:
                data = self.annotation.raw
                vg = VoxelGrid(data)
                vg.encoding.data[data != layer] = False
                mesh = self._get_mesh(vg, self.boundary_mask)
                color = [int(255 * v) for v in matplotlib.colors.to_rgb(colors[i - 1])]
                color.append(255 * alpha)
                mesh.visual.face_colors = color
                meshes.append(mesh)
        return meshes

    def get_meshes(self, plane=None):
        """Get layer pia and region meshes."""
        meshes = self.get_layer_meshes()
        meshes.append(self.get_pia_mesh())
        meshes.append(self.get_boundary_mesh())
        if plane is not None:
            meshes = self.slice_meshes(meshes, plane)
        else:
            meshes.append(self.get_total_boundary_mesh())
        return meshes

    def positions_to_indices(self, points):
        """Simpler positions_to_indices."""
        return self.brain_regions.positions_to_indices(points, keep_fraction=True, strict=False)

    def indices_to_positions(self, indices):
        """Simpler indices_to_positions."""
        return self.brain_regions.indices_to_positions(indices)

    def directions_to_indices(self, direction):
        """Convert directions to indices based coordinates.

        Basically just flip directions wrt signs voxel_dimensions.
        """
        return np.sign(self.brain_regions.voxel_dimensions) * direction

    def slice_mesh(self, mesh, plane, cap=False):
        """Slice mesh."""
        sliced_mesh = mesh.slice_plane(
            self.positions_to_indices(plane["left"].point),
            self.directions_to_indices(plane["left"].normal),
            cap=cap,
        )
        sliced_mesh = sliced_mesh.slice_plane(
            self.positions_to_indices(plane["right"].point),
            self.directions_to_indices(-plane["right"].normal),
            cap=cap,
        )
        sliced_mesh.visual.face_colors = mesh.visual.face_colors[0]
        return sliced_mesh

    def slice_meshes(self, meshes, plane):
        """Slice meshes."""
        sliced_meshes = []
        for mesh in meshes:
            sliced_meshes.append(self.slice_mesh(mesh, plane))
        return sliced_meshes

    def load_morph(self, morph):
        """Load a morphology."""
        paths = []
        colors = {
            NeuriteType.apical_dendrite: [200, 0, 0, 200],
            NeuriteType.basal_dendrite: [0, 200, 0, 200],
            NeuriteType.axon: [100, 100, 100, 200],
        }
        for neurite in morph.neurites:
            for section in neurite.iter_sections():
                path = trimesh.load_path(self.positions_to_indices(section.points[:, :3]))
                path.colors = [colors[neurite.type]]
                paths.append(path)
        return paths

    def load_morphs(self, cells_df):
        """Load multiple morphologies."""
        paths = []
        for gid in cells_df.index:
            m = load_insitu_morphology(cells_df, gid)
            paths += self.load_morph(m)
        return paths

    def load_planes(self, planes=None):
        """Load planes to render as meshes."""
        data = self.annotation.raw
        data = np.ones_like(self.annotation.raw)
        data[0, :, :] = 0
        data[-1, :, :] = 0
        data[:, 0, :] = 0
        data[:, -1, :] = 0
        data[:, :, 0] = 0
        data[:, :, -1] = 0

        meshes = []
        for plane in planes:
            try:
                mesh = VoxelGrid(deepcopy(data)).marching_cubes
                mesh.visual.face_colors = [0, 100, 0, 50]
                sliced_mesh = self.slice_mesh(mesh, plane, cap=True)
                meshes.append(sliced_mesh)
            except (ValueError, IndexError):
                print("cannot load a plane, it may be too close to edges.")

        return meshes

    def render(self, data=None, plane=None, filename=None, show=True, line_width=1.0):
        """Render the meshes with additional data.

        Args:
            data: pass any data which can be rendered
            plane: give a plane to slice the rendering
            filename (str): if given, the figure is exported to this file path
            show (bool): if set to `True` the figure is shown at execution time
            line_width (float): the line width used for rendering
        """
        meshes = self.get_meshes(plane=plane)

        if data is not None:
            if not isinstance(data, list):
                data = [data]
            meshes += data

        scene = trimesh.Scene(meshes)

        if filename is not None:
            with open(filename, "wb") as f:
                f.write(scene.save_image(line_settings={"line_width": line_width}))
        if show:
            scene.show(line_settings={"line_width": line_width}, start_loop=False)

    @staticmethod
    def show():
        """Show the renderings."""
        pyglet.app.run()
