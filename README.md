# NeuroCollage

A tool to create 2D morphology collage plots based on matplotlib.


## Installation

It is recommended to install ``NeuroCollage`` using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install --index-url https://bbpteam.epfl.ch/repository/devpi/simple neurocollage
```

## Usage

This package provides only one command that aims at building figures of morphologies in atlas
planes (i.e. collage plots).

### Inputs

The collage requires the following inputs:

* the path to a standard `CircuitConfig`, or the path to a sonata `circuit_config.json` file of
  a [SONATA circuit](https://sonata-extension.readthedocs.io/en/latest/sonata_overview.html)
  and the path to an Atlas directory that can be read by
  [Voxcell](https://voxcell.readthedocs.io/en/latest/index.html).
* [optional] a configuration file containing the default values used for the CLI arguments (all
  these values are overridden by the ones passed to the CLI). The config file is a `INI` file
  divided in sections. These sections correspond to the first part of the CLI parameter names. For
  example, the `atlas-path` parameter of the CLI corresponds to the `path` parameter of the `atlas`
  section in the configuration file.

### Outputs

This package contains three main functions:
* `get_layer_annotation`: can generate annotation of layers for plotting or other uses
* `create_planes`: defines a set of planes to create collage plots, with various algorithms.
  Planes are sampled along a centerline, which can be straight aligned or not with world
  coordinates or curved using an algorithm in `atlas_analysis` package. The first and last point
  of the centerline can be defined manually, or estimated internally to span the given region best.
* `plot_collage`: make the collage plot, see API for possible arguments.

### Command

This package provides a CLI whose parameters are described in the Command Line Interface page of
this documentation. It is also possible to get help from the command:

```bash
neuro-collage --help
```

If all the arguments are provided in the configuration file, the command is just:

```bash
neuro-collage -c <config-file>
```

Any argument from the configuration file can be overridden through the CLI:

```bash
neuro-collage -c <config-file> --cells-sample 20 --collage-pdf-filename custom_collage_name.pdf
```

Note that the parameter names of the CLI use the section in the configuration file as prefix. In the
previous example, the `--cells-sample` overrides the `sample` parameter of the `cells` section of
the configuration file.


## Examples

The `examples` folder contains a simple example on `S1` region of `SSCx` with `L5_TPC:A` morphologies. It
also provides examples of programmatic use of the `NeuroCollage` API with both types of circuit formats.

![](images/collage.png)
