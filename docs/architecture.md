# AgentOpsML Architecture

## Overview

AgentOpsML separates AI-agent orchestration from deterministic ML execution.

The AI agent is responsible for planning and operating the workflow, while the repository contains the reproducible ML experiment logic.

```text
                    AI Agent
                       |
                       v
                 TrueForge
                       |
                       v
              Daytona Sandbox
                       |
                       v
              AgentOpsML Code
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Dataset      Training     Evaluation
                       |
                       v
                Model Selection
                       |
                       v
                 Final Test
                       |
                       v
                   Artifacts
                       |
                       v
               Verification
