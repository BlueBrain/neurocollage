"""Click app."""
import inspect
import json
import logging
import os
from configparser import ConfigParser

import click

import neurocollage
import neurocollage.loader

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


class TupleParam(click.ParamType):
    """A `click` parameter to process parameters given as tuples."""

    name = "tuple"

    def __init__(self, value_types):
        self.value_types = value_types

    def convert(self, value, param, ctx):
        """Convert a given value."""
        if not isinstance(value, (tuple, list)):
            try:
                values = json.loads(value)
            except json.JSONDecodeError:
                try:
                    values = json.loads("[" + value[1:-1] + "]")
                except json.JSONDecodeError:
                    self.fail(f"{value!r} can not be casted to a tuple", param, ctx)
        else:
            values = value
        try:
            values = [i_type.convert(i, param, ctx) for i, i_type in zip(values, self.value_types)]
        except ValueError:
            self.fail(
                f"Some sub-elements of {value!r} can not be casted to their types", param, ctx
            )
        return tuple(values)


class DictParam(click.ParamType):
    """A `click` parameter to process parameters given as JSON objects."""

    name = "dict"

    def convert(self, value, param, ctx):
        """Convert a given value."""
        try:
            if not isinstance(value, dict):
                value = json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value!r} is not a valid JSON object", param, ctx)

        return value


def configure(ctx, param, filename):
    """Set parameter default values according to a given configuration file."""
    # pylint: disable=unused-argument
    if filename is None:
        return

    # Load the config file
    cfg = ConfigParser()
    cfg.read(filename)

    # Get current default values
    ctx.default_map = {}
    defaults = ctx.default_map

    # Get parameters from the context
    params = {i.name: i for i in ctx.command.get_params(ctx)}

    # Replace the default values for each section of the config file
    for sect in cfg.sections():
        new_defaults = {}
        for k, v in cfg[sect].items():
            param_name = f"{sect}_{k}"
            param_obj = params.get(param_name)
            if param_obj is None:
                raise ValueError(f"The {param_name} parameter (with value '{v}') is unknown")
            if v != "" or param_obj.type == click.STRING:
                new_defaults[param_name] = param_obj.type.convert(v, param_obj, ctx)
        defaults.update(new_defaults)


def _select_args(func, kwargs, mapping=None):
    """Select kwargs for a given function using a name mapping."""
    if mapping is None:
        mapping = {}

    arg_spec = inspect.getfullargspec(func)

    args = []
    if arg_spec.args:
        args.extend(arg_spec.args)

    if arg_spec.kwonlyargs:
        args.extend(arg_spec.kwonlyargs)

    mapped_kwargs = {mapping.get(k, k): v for k, v in kwargs.items() if v is not None}
    selected = {k: v for k, v in mapped_kwargs.items() if k in args}

    return selected


@click.command("collage")
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False, exists=True),
    callback=configure,
    is_eager=True,
    expose_value=False,
    help="Read option defaults from the specified INI file.",
    show_default=True,
)
@click.option(
    "--atlas-path", type=click.Path(exists=True), required=True, help="Path to the atlas directory."
)
@click.option(
    "--atlas-structure-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the structure file of the atlas.",
)
@click.option(
    "--circuit-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the circuit config file.",
)
@click.option("--circuit-region", required=False, help="The region to consider in the circuit.")
@click.option(
    "--circuit-hemisphere", required=False, help="The hemisphere to consider in the circuit."
)
@click.option("--cells-mtype", help="The mtype to consider.")
@click.option(
    "--cells-sample",
    type=click.IntRange(min=0, min_open=True),
    help="The number of cells that are randomly selected to be plotted.",
)
@click.option("--cells-ext", help="The extension of morphology files.")
@click.option(
    "--planes-count",
    type=click.IntRange(min=0, min_open=True),
    help="The number of planes to plot.",
)
@click.option("--planes-type", help="The type of planes to plot.")
@click.option(
    "--planes-slice-thickness",
    type=click.FloatRange(min=0, min_open=True),
    help="The thickness of each plane slice.",
)
@click.option("--collage-pdf-filename", type=click.Path(), help="The output file name.")
@click.option(
    "--collage-dpi",
    type=click.IntRange(min=0, min_open=True),
    help="The DPI used to generate the figure.",
)
@click.option(
    "--collage-n-pixels",
    type=click.IntRange(min=0, min_open=True),
    help="The number of pixels for plotting layers.",
)
@click.option(
    "--collage-n-pixels-y",
    type=click.IntRange(min=0, min_open=True),
    help="The umber of pixels for plotting y field.",
)
@click.option("--collage-with-y-field", type=click.BOOL, help="Trigger to plot y field.")
@click.option("--collage-with-cells", type=click.BOOL, help="Trigger to plot the cells.")
@click.option(
    "--collage-plot-neuron-kwargs",
    type=DictParam(),
    help="The kwargs passed to the `neurom.view.matplotlib_impl.plot_morph` function.",
)
@click.option(
    "--collage-cells-linear-density",
    type=click.FloatRange(min=0, min_open=True),
    help="Resample the cells to plot less points.",
)
@click.option(
    "--collage-cells-wire-plot",
    type=click.BOOL,
    help="Trigger to use `matplotlib.pyplot.plot` instead of `neurom.view`.",
)
@click.option(
    "--random", type=click.BOOL, help="Randomly sample cells, instead of furthest away choice."
)
@click.option(
    "--collage-figsize",
    type=TupleParam([click.IntRange(min=0, min_open=True), click.IntRange(min=0, min_open=True)]),
    help="The size of the output figure.",
)
@click.option(
    "--collage-nb-jobs", type=click.FloatRange(min=-1), help="The number of parallel jobs to use."
)
@click.option(
    "--collage-joblib-verbose",
    type=click.FloatRange(min=0),
    help="The verbosity of the parallel library.",
)
@click.option("--3d", "is_3d", is_flag=True)
def main(**kwargs):
    """Load data from an atlas and a circuit and create a collage figure."""
    # Handle args
    mtype = kwargs.pop("cells_mtype", None)
    if mtype:
        group = {"mtype": mtype}
    else:
        group = None

    atlas_path = {
        "atlas": kwargs.pop("atlas_path"),
        "structure": kwargs.pop("atlas_structure_path"),
    }
    circuit_path = kwargs.pop("circuit_path")
    region = kwargs.pop("circuit_region")
    hemisphere = kwargs.pop("circuit_hemisphere")

    # Select args for each function
    cells_kwargs = _select_args(
        neurocollage.loader.get_cell_df_from_circuit, kwargs, {"cells_ext": "ext"}
    )
    plane_kwargs = _select_args(
        neurocollage.create_planes,
        kwargs,
        {
            "planes_type": "plane_type",
            "planes_count": "plane_count",
            "planes_slice_thickness": "slice_thickness",
            "planes_centerline_first_bound": "centerline_first_bound",
            "planes_centerline_last_bound": "centerline_last_bound",
            "planes_centerline_axis": "centerline_axis",
        },
    )
    collage_kwargs = _select_args(
        neurocollage.plot_2d_collage,
        kwargs,
        {
            "collage_pdf_filename": "pdf_filename",
            "cells_sample": "sample",
            "collage_dpi": "dpi",
            "collage_n_pixels": "n_pixels",
            "collage_n_pixels_y": "n_pixels_y",
            "collage_with_y_field": "with_y_field",
            "collage_with_cells": "with_cells",
            "collage_plot_neuron_kwargs": "plot_neuron_kwargs",
            "collage_cells_linear_density": "cells_linear_density",
            "collage_cells_wire_plot": "cells_wire_plot",
            "collage_figsize": "figsize",
            "collage_nb_jobs": "nb_jobs",
            "collage_joblib_verbose": "joblib_verbose",
            "random": "random",
        },
    )

    # Load cells
    cells_df = neurocollage.loader.get_cell_df_from_circuit(
        circuit_path, group=group, **cells_kwargs
    )

    # Load annotations
    layer_annotation = neurocollage.get_layer_annotation(
        atlas_path, region=region, hemisphere=hemisphere
    )

    # Create planes
    planes, centerline = neurocollage.create_planes(layer_annotation, **plane_kwargs)
    if kwargs["is_3d"]:
        neurocollage.plot_3d_collage(
            cells_df,
            planes,
            layer_annotation,
            atlas_path,
            mtype,
            region,
            hemisphere,
            centerline,
            collage_kwargs["sample"],
        )
    else:
        # Plot and export the figure
        neurocollage.plot_2d_collage(
            cells_df, planes, layer_annotation, atlas_path, mtype=mtype, **collage_kwargs
        )
