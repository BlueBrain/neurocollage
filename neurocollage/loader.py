"""Load circuit and convert to internal format."""
from pathlib import Path

import bluepy
import bluepysnap
import pandas as pd

from neurocollage.exceptions import NeurocollageException

# pylint: disable=protected-access


def get_cell_df_from_circuit(circuit_path, ext=None, group=None):
    """Load cells data from circuit."""
    if Path(circuit_path).suffix == ".json":
        return _get_cell_df_from_circuit_sonata(circuit_path, ext=ext, group=group)
    return get_cell_df_from_circuit_legacy(circuit_path, ext=ext, group=group)


# pylint: disable=inconsistent-return-statements
def get_cell_df_from_circuit_legacy(circuit, ext=None, group=None):
    """Load data from legacy circuit."""
    if isinstance(circuit, str):
        circuit = bluepy.Circuit(circuit)
    df = circuit.cells.get(group=group)
    path = circuit.morph._morph_path
    dirnames, _ext = circuit.morph._dispatch[circuit.morph._morph_type]
    if ext is None:
        ext = _ext
    _morph = df.head(1)["morphology"].to_list()[0]
    for dirname in dirnames:
        p = path + dirname + "/" + _morph + "." + ext
        if Path(p).exists():
            df["path"] = (
                path
                + dirname
                + "/"
                + pd.Series(df["morphology"].to_list(), dtype=str, index=df.index)
                + f".{ext}"
            )
            return df
    raise NeurocollageException(f"We cannot find morphologies in {p}, {dirnames}.")


def _get_cell_df_from_circuit_sonata(circuit_path, ext=None, group=None):
    """Load data from sonata circuit."""
    if ext is None:
        ext = "asc"
    circuit = bluepysnap.Circuit(circuit_path)
    for node in circuit.nodes.values():
        if node.type == "biophysical":
            path = node.morph.get_morphology_dir(extension=ext) + "/"
            df = node.get(group=group)
            # add orientation as rotation matrix
            df["orientation"] = node.orientations()
            # add path to morphology files
            df["path"] = (
                path + pd.Series(df["morphology"].to_list(), dtype=str, index=df.index) + f".{ext}"
            )
            return df
