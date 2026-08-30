from pathlib import Path

from src.artifact_manager import verify_artifacts
from src.model_selector import ExperimentResult, select_best_experiment


def test_selects_highest_validation_f1():
    experiments = [
        ExperimentResult("a", 0.90, 0.95, 1000),
        ExperimentResult("b", 0.95, 0.90, 2000),
    ]

    winner = select_best_experiment(experiments)

    assert winner.name == "b"


def test_uses_accuracy_as_tiebreaker():
    experiments = [
        ExperimentResult("a", 0.95, 0.90, 1000),
        ExperimentResult("b", 0.95, 0.95, 2000),
    ]

    winner = select_best_experiment(experiments)

    assert winner.name == "b"


def test_uses_parameter_count_as_final_tiebreaker():
    experiments = [
        ExperimentResult("large", 1.0, 1.0, 6433),
        ExperimentResult("small", 1.0, 1.0, 1025),
    ]

    winner = select_best_experiment(experiments)

    assert winner.name == "small"


def test_empty_experiments_returns_none():
    assert select_best_experiment([]) is None


def test_artifacts_are_verified():
    output_dir = Path("experiments/validated-optimization")

    report = verify_artifacts(output_dir)

    assert report["all_required_artifacts_exist"] is True
    assert report["metrics_json_valid"] is True
    assert report["experiments_json_valid"] is True
