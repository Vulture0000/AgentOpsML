# AgentOpsML

AgentOpsML is an autonomous machine-learning experimentation pipeline designed to let an AI agent run reproducible ML experiments inside an isolated execution environment.

The system can:

- inspect a dataset automatically
- create reproducible train/validation/test splits
- preprocess data without leaking test information
- train multiple candidate PyTorch architectures
- evaluate candidates using validation metrics
- automatically select the best experiment
- evaluate the selected model on an untouched test set
- save model and experiment artifacts
- verify that the generated artifacts actually exist and contain valid data

## Architecture

```text
User Goal
   |
   v
AI Agent / TrueForge
   |
   v
Isolated Daytona Sandbox
   |
   v
AgentOpsML Experiment Runner
   |
   +-------------------+
   |                   |
   v                   v
Candidate Models    Evaluation
   |                   |
   +---------+---------+
             |
             v
      Model Selection
             |
             v
      Untouched Test
             |
             v
        Artifacts
             |
             v
     Artifact Verification
