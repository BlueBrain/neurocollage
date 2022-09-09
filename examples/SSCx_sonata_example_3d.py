"""Make collage on legacy circuit."""
import trimesh

import neurocollage
import neurocollage.loader
from neurocollage.mesh_helper import MeshHelper
from neurocollage.planes import get_cells_between_planes


def main():
    circuit_path = "/gpfs/bbp.cscs.ch/project/proj83/jira-tickets/NSETM-1948-extract-hex-O1/data/S1_data/circuit_config.json"  # noqa
    region = "S1"
    mtype = "L5_TPC:A"
    hemisphere = "left"
    atlas_path = {
        "atlas": "/gpfs/bbp.cscs.ch/project/proj83/data/atlas/S1/MEAN/P14-MEAN",
        "structure": "../tests/data/region_structure.yaml",
    }
    mesh_helper = MeshHelper(atlas_path, region, hemisphere)

    planes, centerline = neurocollage.create_planes(
        mesh_helper.layer_annotation,
        plane_type="centerline_curved",
        slice_thickness=200,
        plane_count=5,
    )
    # select a plane
    plane = planes[3]

    data = [mesh_helper.load_planes(planes)]

    # render region with layer, pia and centerline
    centerline_data = [trimesh.points.PointCloud(mesh_helper.positions_to_indices(centerline))]
    mesh_helper.render(data=data + centerline_data)

    # to load from scratch, use:
    cells_df = neurocollage.loader.get_cell_df_from_circuit(circuit_path, group={"mtype": mtype})
    # cells_df.to_csv('cells_df.csv')
    # cells_df = pd.read_csv("cells_df.csv")
    cells_df = cells_df[cells_df.mtype == mtype]
    cells_df = get_cells_between_planes(cells_df, plane["left"], plane["right"])
    cells_df = cells_df.head(20)

    data = mesh_helper.load_morphs(cells_df)

    # render collage plane with morphologies
    mesh_helper.render(data=data, plane=plane)

    mesh_helper.show()


if __name__ == "__main__":
    main()
