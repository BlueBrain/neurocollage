"""Tests for the neurocollage.cli module."""
import neurocollage.cli as tested


def test_cli(cli_runner, data_path, tmpdir, small_O1_path, cell_collection):
    # pylint: disable=unused-argument
    """Test the CLI."""
    cell_collection.population_name = "S1"
    cell_collection.save(data_path / "nodes.sonata")
    result = cli_runner.invoke(
        tested.main,
        [
            "--config",
            data_path / "test_config.ini",
            "--atlas-path",
            small_O1_path["atlas"],
            "--circuit-path",
            data_path / "circuit_config.json",
            "--collage-pdf-filename",
            tmpdir / "collage.pdf",
        ],
    )
    assert result.exit_code == 0


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run("neurocollage", "--version")
    assert ret.success
    assert ret.stdout.startswith("neurocollage, version ")
    assert ret.stderr == ""
