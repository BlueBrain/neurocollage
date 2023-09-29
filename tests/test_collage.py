"""Test collage module."""
from pathlib import Path

import neurocollage.collage as tested

DATA = Path(__file__).parent / "data"


def test_plot_2d_collage(tmpdir, cells_df, planes, layer_annotation, small_O1_path):
    """Test plot_collage."""
    pdf_filename = tmpdir / "collage.pdf"
    tested.plot_2d_collage(
        cells_df,
        planes,
        layer_annotation,
        small_O1_path,
        pdf_filename=pdf_filename,
        nb_jobs=1,  # Coverage fails with several workers
    )
    # write a test on plot when stable
