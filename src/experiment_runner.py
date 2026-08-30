from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make the experiment as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MediumMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 32)
        self.block1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.block2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        residual = x
        x = self.block1(x) + residual
        residual = x
        x = self.block2(x) + residual
        return self.output(x)


@dataclass
class ExperimentResult:
    name: str
    architecture: str
    parameter_count: int
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_loss: float


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    model.eval()

    with torch.no_grad():
        logits = model(features).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
        true_labels = labels.cpu().numpy()

    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(
            precision_score(true_labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(true_labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(true_labels, predictions, zero_division=0)),
        "loss": float(log_loss(true_labels, probabilities, labels=[0, 1])),
    }


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    epochs: int = 80,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_f1 = -1.0
    patience = 12
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        logits = model(x_train).squeeze(1)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        validation_metrics = evaluate(
            model,
            x_validation,
            y_validation,
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return evaluate(model, x_validation, y_validation)


def prepare_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    dict[str, int],
]:
    dataset = load_breast_cancer()

    x = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)

    # First split: 70% train, 30% temporary.
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    # Second split: temporary 50/50 -> 15% validation, 15% test.
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_test = scaler.transform(x_test)

    split_sizes = {
        "train": len(x_train),
        "validation": len(x_validation),
        "test": len(x_test),
    }

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(x_validation, dtype=torch.float32),
        torch.tensor(y_validation, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        x.shape[1],
        split_sizes,
    )


def run_experiment(
    name: str,
    architecture: str,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
) -> ExperimentResult:
    validation = train_model(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    return ExperimentResult(
        name=name,
        architecture=architecture,
        parameter_count=count_parameters(model),
        validation_accuracy=validation["accuracy"],
        validation_precision=validation["precision"],
        validation_recall=validation["recall"],
        validation_f1=validation["f1"],
        validation_loss=validation["loss"],
    )


def main(output_dir: str = "experiments/validated-optimization") -> None:
    set_seed()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        input_dim,
        split_sizes,
    ) = prepare_data()

    print(f"Dataset input features: {input_dim}")
    print(f"Split sizes: {split_sizes}")

    experiments: list[ExperimentResult] = []

    candidates = [
        (
            "experiment-1",
            "Simple MLP",
            SimpleMLP(input_dim),
        ),
        (
            "experiment-2",
            "Medium MLP + BatchNorm",
            MediumMLP(input_dim),
        ),
        (
            "experiment-3",
            "Deep Residual MLP",
            DeepResidualMLP(input_dim),
        ),
    ]

    trained_models: dict[str, nn.Module] = {}

    for name, architecture, model in candidates:
        print(f"\nRunning {name}: {architecture}")

        result = run_experiment(
            name,
            architecture,
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

        experiments.append(result)
        trained_models[name] = model

        print(
            f"Validation F1={result.validation_f1:.4f}, "
            f"Accuracy={result.validation_accuracy:.4f}, "
            f"Parameters={result.parameter_count}"
        )

    # Selection uses validation metrics only.
    experiments.sort(
        key=lambda item: (
            item.validation_f1,
            item.validation_accuracy,
            -item.parameter_count,
        ),
        reverse=True,
    )

    winner = experiments[0]
    winning_model = trained_models[winner.name]

    # The test set is intentionally evaluated only after selection.
    final_test = evaluate(
        winning_model,
        x_test,
        y_test,
    )

    metrics = {
        "seed": SEED,
        "dataset": "sklearn breast cancer",
        "input_features": input_dim,
        "split_sizes": split_sizes,
        "selection_rule": [
            "validation_f1_descending",
            "validation_accuracy_descending",
            "parameter_count_ascending",
        ],
        "winner": winner.name,
        "final_test": {
            "accuracy": final_test["accuracy"],
            "precision": final_test["precision"],
            "recall": final_test["recall"],
            "f1": final_test["f1"],
            "loss": final_test["loss"],
        },
    }

    experiments_data = {
        "seed": SEED,
        "experiments": [asdict(item) for item in experiments],
        "winner": winner.name,
    }

    torch.save(winning_model.state_dict(), output_path / "model.pt")

    with open(output_path / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(
        output_path / "experiments.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(experiments_data, file, indent=2)

    print("\nMODEL SELECTION")
    print(f"Winner: {winner.name}")
    print(f"Validation F1: {winner.validation_f1:.4f}")
    print(f"Parameters: {winner.parameter_count}")

    print("\nFINAL UNTOUCHED TEST")
    print(f"Accuracy: {final_test['accuracy']:.4f}")
    print(f"Precision: {final_test['precision']:.4f}")
    print(f"Recall: {final_test['recall']:.4f}")
    print(f"F1: {final_test['f1']:.4f}")
    print(f"Loss: {final_test['loss']:.4f}")

    from .artifact_manager import assert_artifacts_valid

    print("\nVERIFYING ARTIFACTS")

    try:
        assert_artifacts_valid(output_path)
        print("All required artifacts verified successfully.")
    except (FileNotFoundError, ValueError) as error:
        print(f"Artifact verification failed: {error}")
        raise

if __name__ == "__main__":
    main()


