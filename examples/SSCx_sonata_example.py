"""Make collage on legacy circuit."""

import neurocollage
import neurocollage.loader


def main():
    """The function to plot the collage of the circuit."""
    circuit_path = (
        "/gpfs/bbp.cscs.ch/project/proj83/jira-tickets/NSETM-1948-extract-hex-O1/data/S1_data/"
        "circuit_config.json"
    )
    region = "S1"
    mtype = "L5_TPC:A"
    atlas_path = {
        "atlas": "/gpfs/bbp.cscs.ch/project/proj83/data/atlas/S1/MEAN/P14-MEAN",
        "structure": "../tests/data/region_structure.yaml",
    }

    cells_df = neurocollage.loader.get_cell_df_from_circuit(circuit_path, group={"mtype": mtype})

    layer_annotation = neurocollage.get_layer_annotation(atlas_path, region=region)
    planes, centerline = neurocollage.create_planes(
        layer_annotation, plane_type="centerline_straight", slice_thickness=20, plane_count=5
    )

    neurocollage.plot_collage(
        cells_df,
        planes,
        layer_annotation,
        atlas_path,
        pdf_filename="collage_sonata_S1.pdf",
        sample=10,
    )


if __name__ == "__main__":
    main()
