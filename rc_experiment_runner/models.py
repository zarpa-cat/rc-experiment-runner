"""Pydantic models for experiments, assignments, and results."""

from datetime import datetime

from pydantic import BaseModel, model_validator


class Variant(BaseModel):
    """An experiment arm (e.g., control or treatment)."""

    id: str
    name: str
    description: str = ""
    weight: float = 0.5


class Experiment(BaseModel):
    """An A/B experiment definition with weighted variants."""

    id: str
    name: str
    variants: list[Variant]
    start_date: datetime
    end_date: datetime | None = None
    salt: str = ""

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "Experiment":
        """Ensure variant weights sum to 1.0 (within floating-point tolerance)."""
        total = sum(v.weight for v in self.variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Variant weights must sum to 1.0, got {total}")
        return self


class Assignment(BaseModel):
    """Records which variant a subscriber was assigned to."""

    subscriber_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime


class ConversionEvent(BaseModel):
    """A conversion event for a subscriber in an experiment."""

    subscriber_id: str
    experiment_id: str
    event_type: str = "conversion"
    value: float = 0.0
    recorded_at: datetime


class VariantStats(BaseModel):
    """Aggregated statistics for a single variant."""

    variant_id: str
    assignments: int
    conversions: int
    conversion_rate: float
    total_value: float
    avg_value: float


class ExperimentResults(BaseModel):
    """Aggregated results for an experiment across all variants."""

    experiment_id: str
    total_subjects: int
    variant_stats: dict[str, VariantStats]
