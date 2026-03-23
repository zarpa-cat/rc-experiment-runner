"""Tests for the CLI interface."""

import os
import tempfile

import pytest
from typer.testing import CliRunner

from rc_experiment_runner.cli import app

runner = CliRunner()


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _create_experiment(db_path: str, experiment_id: str = "exp-1") -> None:
    result = runner.invoke(app, [
        "create", experiment_id, "Test Experiment",
        "--variant", "control:0.5",
        "--variant", "treatment:0.5",
        "--db-path", db_path,
    ])
    assert result.exit_code == 0


class TestCliCreate:
    def test_create_experiment(self, db_path):
        result = runner.invoke(app, [
            "create", "exp-1", "My Experiment",
            "--variant", "control:0.5",
            "--variant", "treatment:0.5",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0
        assert "Created experiment" in result.output

    def test_create_with_end_date(self, db_path):
        result = runner.invoke(app, [
            "create", "exp-1", "My Experiment",
            "--variant", "control:0.5",
            "--variant", "treatment:0.5",
            "--end-date", "2025-12-31",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0

    def test_create_with_salt(self, db_path):
        result = runner.invoke(app, [
            "create", "exp-1", "My Experiment",
            "--variant", "control:0.5",
            "--variant", "treatment:0.5",
            "--salt", "my-salt",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0

    def test_create_three_variants(self, db_path):
        result = runner.invoke(app, [
            "create", "exp-1", "Three Way",
            "--variant", "a:0.33",
            "--variant", "b:0.34",
            "--variant", "c:0.33",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0
        assert "3 variants" in result.output


class TestCliList:
    def test_list_empty(self, db_path):
        result = runner.invoke(app, ["list", "--db-path", db_path])
        assert result.exit_code == 0
        assert "No experiments" in result.output

    def test_list_with_experiments(self, db_path):
        _create_experiment(db_path, "exp-1")
        _create_experiment(db_path, "exp-2")
        result = runner.invoke(app, ["list", "--db-path", db_path])
        assert result.exit_code == 0
        assert "exp-1" in result.output
        assert "exp-2" in result.output


class TestCliAssign:
    def test_assign(self, db_path):
        _create_experiment(db_path)
        result = runner.invoke(app, [
            "assign", "user-123", "exp-1",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0
        assert "user-123" in result.output

    def test_assign_deterministic(self, db_path):
        _create_experiment(db_path)
        r1 = runner.invoke(app, ["assign", "user-1", "exp-1", "--db-path", db_path])
        r2 = runner.invoke(app, ["assign", "user-1", "exp-1", "--db-path", db_path])
        assert r1.output == r2.output


class TestCliConvert:
    def test_convert(self, db_path):
        _create_experiment(db_path)
        runner.invoke(app, ["assign", "user-1", "exp-1", "--db-path", db_path])
        result = runner.invoke(app, [
            "convert", "user-1", "exp-1",
            "--value", "9.99",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0
        assert "Recorded conversion" in result.output

    def test_convert_with_event_type(self, db_path):
        _create_experiment(db_path)
        runner.invoke(app, ["assign", "user-1", "exp-1", "--db-path", db_path])
        result = runner.invoke(app, [
            "convert", "user-1", "exp-1",
            "--event-type", "purchase",
            "--db-path", db_path,
        ])
        assert result.exit_code == 0


class TestCliResults:
    def test_results(self, db_path):
        _create_experiment(db_path)
        runner.invoke(app, ["assign", "user-1", "exp-1", "--db-path", db_path])
        runner.invoke(app, ["convert", "user-1", "exp-1", "--value", "9.99", "--db-path", db_path])
        result = runner.invoke(app, ["results", "exp-1", "--db-path", db_path])
        assert result.exit_code == 0
        assert "Total subjects" in result.output


class TestCliDelete:
    def test_delete(self, db_path):
        _create_experiment(db_path)
        result = runner.invoke(app, ["delete", "exp-1", "--db-path", db_path])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Verify it's gone
        list_result = runner.invoke(app, ["list", "--db-path", db_path])
        assert "exp-1" not in list_result.output or "No experiments" in list_result.output
