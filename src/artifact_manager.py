from __future__ import annotations

import json
from pathlib import Path


REQUIRED_ARTIFACTS = (
    "model.pt",
    "metrics.json",
    "experiments.json",
)


def verify_artifacts(output_dir: str | Path) -> dict[str, object]:
    """
    Verify that all expected experiment artifacts exist and are readable.

    Returns a structured verification report rather than silently assuming
    that artifact creation succeeded.
    """
    directory = Path(output_dir)

    artifacts: dict[str, dict[str, object]] = {}

    for filename in REQUIRED_ARTIFACTS:
        path = directory / filename

        exists = path.is_file()
        size = path.stat().st_size if exists else 0

        artifacts[filename] = {
            "exists": exists,
            "size_bytes": size,
        }

    metrics_valid = False
    experiments_valid = False

    metrics_path = directory / "metrics.json"
    experiments_path = directory / "experiments.json"

    if metrics_path.is_file():
        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                json.load(file)
            metrics_valid = True
        except (json.JSONDecodeError, OSError):
            metrics_valid = False

    if experiments_path.is_file():
        try:
            with experiments_path.open("r", encoding="utf-8") as file:
                json.load(file)
            experiments_valid = True
        except (json.JSONDecodeError, OSError):
            experiments_valid = False

    all_exist = all(
        artifact["exists"]
        for artifact in artifacts.values()
    )

    report = {
        "output_directory": str(directory),
        "all_required_artifacts_exist": all_exist,
        "metrics_json_valid": metrics_valid,
        "experiments_json_valid": experiments_valid,
        "artifacts": artifacts,
    }

    return report


def assert_artifacts_valid(output_dir: str | Path) -> None:
    """Raise an error if the experiment artifacts are incomplete or invalid."""
    report = verify_artifacts(output_dir)

    if not report["all_required_artifacts_exist"]:
        raise FileNotFoundError(
            f"Missing required artifacts in {output_dir}"
        )

    if not report["metrics_json_valid"]:
        raise ValueError("metrics.json is missing or invalid")

    if not report["experiments_json_valid"]:
        raise ValueError("experiments.json is missing or invalid")
