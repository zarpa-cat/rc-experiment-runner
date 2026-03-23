"""Deterministic variant assignment using HMAC-SHA256."""

import hashlib
import hmac

from rc_experiment_runner.models import Experiment, Variant


def assign_variant(subscriber_id: str, experiment: Experiment) -> Variant:
    """Deterministically assign a subscriber to a variant.

    Uses HMAC-SHA256(key=experiment.id + experiment.salt, msg=subscriber_id)
    to produce a deterministic hash, then maps it to a variant based on
    cumulative weights.

    Args:
        subscriber_id: The subscriber to assign.
        experiment: The experiment with weighted variants.

    Returns:
        The assigned Variant.
    """
    key = (experiment.id + experiment.salt).encode("utf-8")
    msg = subscriber_id.encode("utf-8")
    digest = hmac.new(key, msg, hashlib.sha256).hexdigest()

    # Convert first 8 hex chars to a float in [0, 1)
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    cumulative = 0.0
    for variant in experiment.variants:
        cumulative += variant.weight
        if bucket < cumulative:
            return variant

    # Fallback for floating-point edge case
    return experiment.variants[-1]
