"""Module with utils functions."""
import numpy as np
from neurom import load_morphology


def load_insitu_morphology(df, gid):
    """Load a morphology from dataframe with position and orientation."""
    m = load_morphology(df.loc[gid, "path"])

    if "orientation" in df.columns:
        orientation = df.loc[gid, "orientation"]
        if isinstance(orientation, str):

            def str2array(s):
                # dirty, just in case we saved df to .csv
                # pylint: disable=anomalous-backslash-in-string,import-outside-toplevel
                import ast
                import re

                s = re.sub(r"\[ +", "[", s.strip())
                s = re.sub(r"[,\s]+", ", ", s)
                return np.array(ast.literal_eval(s))

            orientation = str2array(orientation)

        # pylint: disable=cell-var-from-loop
        def _trans(p):
            return p.dot(orientation.T)

        m = m.transform(_trans)

    # pylint: disable=cell-var-from-loop
    def trans(p):
        return p + df.loc[gid, ["x", "y", "z"]].to_numpy().T
    p = m.transform(trans)
    return p
