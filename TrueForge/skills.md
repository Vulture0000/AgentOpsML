# AgentOpsML TrueForge Skills

These skills define the operations required by the AgentOpsML workflow when executed through the TrueForge agent harness.

## 1. Inspect Environment

The agent should inspect the execution environment before starting an experiment.

The agent should determine:

- Python version
- available compute resources
- available disk space
- installed dependencies
- working directory

## 2. Inspect Dataset

The agent should inspect the dataset before designing the experiment.

The inspection should determine:

- number of samples
- number of features
- target variable
- class distribution
- data types
- missing values where applicable

## 3. Prepare Experiment

The agent should create a reproducible experiment configuration.

The configuration should include:

- random seed
- train/validation/test split
- preprocessing strategy
- candidate model architectures
- training configuration
- evaluation metrics

## 4. Execute ML Workloads

Execute training and evaluation inside the isolated execution environment.

Prefer resource-efficient approaches when GPU resources are unavailable.

For the current demonstration, the experiment uses CPU-compatible PyTorch.

## 5. Compare Experiments

Collect validation results from each candidate experiment.

The model-selection policy is:

1. Highest validation F1
2. Highest validation accuracy when F1 is tied
3. Lowest parameter count when both metrics are tied

The final test set must not influence this selection.

## 6. Final Evaluation

After selecting the winning experiment, evaluate the selected model on the untouched test set.

Record:

- accuracy
- precision
- recall
- F1
- loss

## 7. Artifact Management

The agent should ensure that the experiment produces the expected artifacts:

- `model.pt`
- `metrics.json`
- `experiments.json`

Artifacts should be verified after generation.

## 8. Failure Recovery

If execution fails, the agent should:

1. inspect the error
2. identify the likely cause
3. apply a targeted correction
4. rerun the affected operation
5. verify the result
6. report the failure and correction

The agent must not claim success when an operation has not actually succeeded.

## 9. Experiment Reporting

The final report should include:

- dataset information
- experiment configuration
- candidate models
- validation results
- selected model
- final test results
- generated artifacts
- verification status

## 10. Execution Boundary

TrueForge provides the agent execution and orchestration layer.

Daytona provides the isolated execution environment.

AgentOpsML provides the ML experimentation and evaluation logic.

The language model provider is an external component used by the agent and is not represented as an AgentOpsML-trained foundation model.
