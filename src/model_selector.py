from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentResult:
    name: str
    validation_f1: float
    validation_accuracy: float
    parameter_count: int


def select_best_experiment(
    experiments: list[ExperimentResult],
) -> Optional[ExperimentResult]:
    """
    Select the best experiment using:

    1. Validation F1 (higher is better)
    2. Validation accuracy (higher is better)
    3. Parameter count (lower is better)
    """

    if not experiments:
        return None

    return max(
        experiments,
        key=lambda experiment: (
            experiment.validation_f1,
            experiment.validation_accuracy,
            -experiment.parameter_count,
        ),
    )
