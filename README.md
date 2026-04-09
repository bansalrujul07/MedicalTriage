---
title: Medical Triage
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
tags:
    - openenv
    - fastapi
    - healthcare
    - reinforcement-learning
---

# MedicalTriage

MedicalTriage is an action-based triage simulation framework for comparing Random, Rule-based, LLM, and RL agents across three progressively harder tasks.

## Project Overview

The environment simulates high-stakes patient triage under constrained resources.
Difficulty is modeled through formal task configurations:
- task1: basic triage
- task2: resource-constrained triage
- task3: high-pressure triage

Detailed architecture notes are in [triage_env/docs/task_architecture.md](triage_env/docs/task_architecture.md).

## Task Definitions and Difficulty

- task1 (easy): baseline triage workflow with fewer patients and less severe deterioration.
- task2 (medium): constrained resource allocation with tighter survival and stabilization targets.
- task3 (hard): high-pressure triage with multiple critical patients and harsher penalties.

Each task has deterministic programmatic graders in:
- `graders/task1_grader.py`
- `graders/task2_grader.py`
- `graders/task3_grader.py`

## Installation

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./triage_env
```

The editable install lets you run module commands from any subdirectory.

## Environment Variables

All environment variables are loaded from `.env` file automatically.

### Quick LLM Setup
See [LLM_SETUP.md](LLM_SETUP.md) for complete OpenAI configuration guide.

Example `.env` file:
```bash
OPENAI_API_KEY=sk-proj-your_key_here
TRIAGE_LLM_MODEL=gpt-4.1-mini
TRIAGE_LLM_TEMPERATURE=0.0
TRIAGE_LLM_MAX_TOKENS=200
TRIAGE_LLM_TIMEOUT=20
TRIAGE_DEFAULT_TASK=task2
TRIAGE_SEED=42
TRIAGE_TRAIN_EPISODES=200
TRIAGE_EVAL_EPISODES=30
```

⚠️ **Important:** Never commit `.env` to git (already in `.gitignore`)

## Action Schema

```python
TriageAction(
    action_type="treat" | "allocate_ventilator" | "wait",
    patient_id=int,  # use -1 for wait
)
```

## Observation Schema

Each step returns an observation with:
- patients
- resources
- step_count
- message
- reward
- done
- metadata

Metadata includes task name, reward breakdown, invalid action count, and resource usage.

## Reward Schema

The environment exposes both:
- scalar `reward: float` per step
- typed `reward_detail: TriageReward` payload for structured reward components

See `triage_env/models.py` for `TriageReward` fields.

## Run Tests

From repository root:

```bash
python -m pytest -q
```

## Run Agents

### Random
```bash
python -m triage_env.scripts.run_random --task task1
```

### Rule-based
```bash
python -m triage_env.scripts.run_rule_based --task task2
```

### LLM
```bash
python -m triage_env.scripts.run_llm_agent --task task3
```

If OPENAI_API_KEY is missing, LLMAgent runs with a safe fallback policy.

## OpenAI Baseline Inference (Submission Format)

The root `inference.py` script uses the OpenAI client and emits strict stdout logs for one task run.

Required env vars:
```bash
export HF_TOKEN=<your_api_key>
export LOCAL_IMAGE_NAME=medicaltriage:latest
```

Optional (with defaults in script):
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export TRIAGE_TASK=task3
```

Run baseline:
```bash
python inference.py
```

Stdout format is restricted to:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

The final score emitted in `[END]` is normalized to `[0, 1]`.

## Train Agents

### RL
```bash
python -m triage_env.scripts.train_rl
```

Trains across task1, task2, task3 and writes:
- triage_env/training/triage_rl_qtable.json

### Q-learning
```bash
python -m triage_env.scripts.train_q_agent
```

Trains across task1, task2, task3 and writes:
- triage_env/training/q_agent.pkl

## Benchmark All Agents Across Tasks

```bash
python -m triage_env.scripts.run_benchmark
```

Optional filters:

```bash
python -m triage_env.scripts.run_benchmark --task task2
python -m triage_env.scripts.run_benchmark --agent RLAgent
python -m triage_env.scripts.run_benchmark --task task3 --agent LLMAgent --episodes 10
python -m triage_env.scripts.run_benchmark --tasks task1,task2 --agents RandomAgent,RuleBasedAgent
python -m triage_env.scripts.run_benchmark --tasks task1 --agents RLAgent --output benchmark_task1.csv
```

CSV output:
- triage_env/evaluation/results/benchmark_summary.csv

## Baseline Scores

Reference baseline metrics from `benchmark_final.csv` (3 episodes):

| Task | Agent | Avg Total Reward | Survival Rate |
|------|-------|------------------|---------------|
| task1 | RuleBasedAgent | 250.92 | 1.00 |
| task2 | TrainedQAgent | 221.63 | 0.75 |
| task3 | RLAgent | 19.43 | 0.27 |

To generate a submission-format baseline run, set `HF_TOKEN` and execute `python inference.py`.

## Server

```bash
python -m triage_env.server.app --port 8000
```

## Deployment

Production deployment files are included at repository root:
- `Dockerfile`
- `docker-compose.yml`
- `deployment/k8s/`
- `scripts/deploy_dockerhub.sh`
- `scripts/deploy_ghcr.sh`
- `scripts/deploy_k8s.sh`

See `DEPLOYMENT.md` for end-to-end local, registry, and Kubernetes deployment commands.

## Troubleshooting

### ModuleNotFoundError: No module named triage_env
Run this once from root:
```bash
pip install -e ./triage_env
```

### LLM agent not using real API
Check:
- OPENAI_API_KEY exists
- model/env vars are set

### Benchmark missing trained agent performance
Train models first:
```bash
python -m triage_env.scripts.train_rl
python -m triage_env.scripts.train_q_agent
```

### Running commands from nested directories
Use module mode always:
```bash
python -m triage_env.scripts.run_benchmark
```
