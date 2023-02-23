"""Make collage on sonata circuit."""
import neurocollage
import neurocollage.loader


def main():
    """The function to plot the collage of the circuit."""
    circuit_path = "/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/CircuitConfig"
    region = "CA1"
    mtype = "SP_PC"
    atlas_path = {
        "atlas": "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/",
        "structure": "region_structure_hipp.yaml",
    }

    cells_df = neurocollage.loader.get_cell_df_from_circuit(circuit_path, group={"mtype": mtype})

    layer_annotation = neurocollage.get_layer_annotation(atlas_path, region=region)
    planes, centerline = neurocollage.create_planes(
        layer_annotation,
        plane_type="centerline_curved",
        slice_thickness=20,
        plane_count=50,
        centerline_first_bound=[2041.44079784, 1583.54052927, 6687.27905367],
        centerline_last_bound=[2169.45996094, 6991.83984375, 2992.00000000],
        # centerline_first_bound=[130,108,427],
        # centerline_last_bound=[138,446,196],
    )

    neurocollage.plot_collage(
        cells_df,
        planes,
        layer_annotation,
        atlas_path,
        pdf_filename="collage_legacy_hipp.pdf",
        sample=10,
    )


if __name__ == "__main__":
    main()
