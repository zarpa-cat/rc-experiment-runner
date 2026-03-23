"""SQLite-backed storage for experiments, assignments, and conversions."""

import json
import sqlite3
from datetime import datetime

from rc_experiment_runner.models import (
    Assignment,
    ConversionEvent,
    Experiment,
    ExperimentResults,
    Variant,
    VariantStats,
)


class ExperimentStore:
    """SQLite-backed persistence for experiment data."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                variants_json TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                salt TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS assignments (
                subscriber_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                assigned_at TEXT NOT NULL,
                PRIMARY KEY (subscriber_id, experiment_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS conversions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscriber_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'conversion',
                value REAL NOT NULL DEFAULT 0.0,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
        """)
        self._conn.commit()

    def create_experiment(self, experiment: Experiment) -> None:
        """Store a new experiment."""
        variants_json = json.dumps([v.model_dump() for v in experiment.variants])
        self._conn.execute(
            "INSERT INTO experiments (id, name, variants_json, start_date, end_date, salt) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                experiment.id,
                experiment.name,
                variants_json,
                experiment.start_date.isoformat(),
                experiment.end_date.isoformat() if experiment.end_date else None,
                experiment.salt,
            ),
        )
        self._conn.commit()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Retrieve an experiment by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experiment(row)

    def list_experiments(self) -> list[Experiment]:
        """List all experiments."""
        rows = self._conn.execute("SELECT * FROM experiments ORDER BY start_date").fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def record_assignment(self, assignment: Assignment) -> None:
        """Record a variant assignment. Idempotent — same subscriber+experiment is a no-op."""
        self._conn.execute(
            "INSERT OR IGNORE INTO assignments "
            "(subscriber_id, experiment_id, variant_id, assigned_at) "
            "VALUES (?, ?, ?, ?)",
            (
                assignment.subscriber_id,
                assignment.experiment_id,
                assignment.variant_id,
                assignment.assigned_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_assignment(self, subscriber_id: str, experiment_id: str) -> Assignment | None:
        """Get an existing assignment for a subscriber in an experiment."""
        row = self._conn.execute(
            "SELECT * FROM assignments WHERE subscriber_id = ? AND experiment_id = ?",
            (subscriber_id, experiment_id),
        ).fetchone()
        if row is None:
            return None
        return Assignment(
            subscriber_id=row["subscriber_id"],
            experiment_id=row["experiment_id"],
            variant_id=row["variant_id"],
            assigned_at=datetime.fromisoformat(row["assigned_at"]),
        )

    def record_conversion(self, event: ConversionEvent) -> None:
        """Record a conversion event."""
        self._conn.execute(
            "INSERT INTO conversions "
            "(subscriber_id, experiment_id, event_type, value, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event.subscriber_id,
                event.experiment_id,
                event.event_type,
                event.value,
                event.recorded_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_results(self, experiment_id: str) -> ExperimentResults:
        """Calculate aggregated results for an experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        variant_stats: dict[str, VariantStats] = {}
        total_subjects = 0

        for variant in experiment.variants:
            # Count assignments
            assignments_row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM assignments "
                "WHERE experiment_id = ? AND variant_id = ?",
                (experiment_id, variant.id),
            ).fetchone()
            assignments = assignments_row["cnt"]
            total_subjects += assignments

            # Count conversions and sum values
            conv_row = self._conn.execute(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(value), 0.0) as total_val "
                "FROM conversions "
                "WHERE experiment_id = ? AND subscriber_id IN "
                "(SELECT subscriber_id FROM assignments "
                "WHERE experiment_id = ? AND variant_id = ?)",
                (experiment_id, experiment_id, variant.id),
            ).fetchone()
            conversions = conv_row["cnt"]
            total_value = conv_row["total_val"]

            conversion_rate = conversions / assignments if assignments > 0 else 0.0
            avg_value = total_value / conversions if conversions > 0 else 0.0

            variant_stats[variant.id] = VariantStats(
                variant_id=variant.id,
                assignments=assignments,
                conversions=conversions,
                conversion_rate=conversion_rate,
                total_value=total_value,
                avg_value=avg_value,
            )

        return ExperimentResults(
            experiment_id=experiment_id,
            total_subjects=total_subjects,
            variant_stats=variant_stats,
        )

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and all associated data."""
        self._conn.execute("DELETE FROM conversions WHERE experiment_id = ?", (experiment_id,))
        self._conn.execute("DELETE FROM assignments WHERE experiment_id = ?", (experiment_id,))
        self._conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        self._conn.commit()

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        variants_data = json.loads(row["variants_json"])
        variants = [Variant(**v) for v in variants_data]
        return Experiment(
            id=row["id"],
            name=row["name"],
            variants=variants,
            start_date=datetime.fromisoformat(row["start_date"]),
            end_date=datetime.fromisoformat(row["end_date"]) if row["end_date"] else None,
            salt=row["salt"],
        )
