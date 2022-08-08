"""Test collage module."""
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
        ],
    )
    print(result.output)
    assert result.exit_code == 0
