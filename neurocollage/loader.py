"""Load circuit and convert to internal format."""
from pathlib import Path

import bluepy
import bluepysnap
import pandas as pd

# pylint: disable=protected-access


def get_cell_df_from_circuit(circuit_path, ext="asc", group=None):
    """Load cells data from circuit."""
    if Path(circuit_path).suffix == ".json":
        return _get_cell_df_from_circuit_sonata(circuit_path, ext=ext, group=group)
    return _get_cell_df_from_circuit(circuit_path, ext=ext, group=group)


# pylint: disable=inconsistent-return-statements
def _get_cell_df_from_circuit(circuit_path, ext="asc", group=None):
    """Load data from legacy circuit."""
    circuit = bluepy.Circuit(circuit_path)
    df = circuit.cells.get(group=group)
    path = circuit.morph._morph_path
    dirnames, ext = circuit.morph._dispatch[circuit.morph._morph_type]
    _morph = df.head(1)["morphology"].to_list()[0]
    not_found = True
    for dirname in dirnames:
        p = path + dirname + "/" + _morph + "." + ext
        if Path(p).exists():
            not_found = False
            break
    if not_found:
        raise Exception("We cannot find morphologies.")

    # add path to morphology files
    df["path"] = (
        path
        + dirname
        + "/"
        + pd.Series(df["morphology"].to_list(), dtype=str, index=df.index)
        + f".{ext}"
    )
    return df


def _get_cell_df_from_circuit_sonata(circuit_path, ext="asc", group=None):
    """Load data from sonata circuit."""
    circuit = bluepysnap.Circuit(circuit_path)
    for node in circuit.nodes.values():
        if node.type == "biophysical":
            path = node.morph._get_morph_dir(extension=ext) + "/"
            df = node.get(group=group)
            # add orientation as rotation matrix
            df["orientation"] = node.orientations()
            # add path to morphology files
            df["path"] = (
                path + pd.Series(df["morphology"].to_list(), dtype=str, index=df.index) + f".{ext}"
            )
            return df
