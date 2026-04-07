# Migration Guide: Legacy Layout to Task-Based Framework

Date: 2026-04-07

## Old Behavior

- Difficulty flags were loosely defined and not fully wired into dynamics.
- Reward behavior was mostly global and not task-specific.
- Training/evaluation scripts had import and naming drift.
- Some docs referenced stale message-based examples.

## New Behavior

### 1. Formal task system

A dedicated task configuration module now defines:
- task1
- task2
- task3

Each task includes:
- number of patients
- max steps
- initial resources
- severity mix
- deterioration rates
- reward coefficients
- terminal success criteria

### 2. Task-specific reward system

Rewards are now composed from explicit components per task, including:
- treatment success by severity
- ventilator allocation reward
- invalid action penalties
- wait penalties
- death penalties by severity
- stabilization bonus
- terminal success bonus
- all-critical-survive bonus

### 3. Environment contract consistency

The action-based API remains the source of truth:
- action_type
- patient_id

Observations remain state-centric and include metadata with:
- task
- reward_breakdown
- invalid_action_count
- resource_usage

### 4. Evaluator API

Canonical evaluator:
- evaluate_agent(...)

Compatibility wrapper retained:
- evaluate(...)

New metrics include:
- avg_total_reward
- survival_rate
- critical_survival_rate
- avg_episode_length
- invalid_action_count
- deaths_by_severity
- resource_utilization
- success_rate

### 5. Scripts and canonical entrypoints

Canonical module entrypoints are under triage_env.scripts:
- run_random
- run_rule_based
- run_llm_agent
- train_rl
- train_q_agent
- run_benchmark

run_benchmark supports single-task/single-agent and full matrix execution.

### 6. RL and Q-learning compatibility

- Shared state encoder now uses only real observation fields + task metadata.
- No references to nonexistent observation attributes.
- RL/Q training scripts run across task1/task2/task3.

### 7. LLM integration

LLMAgent is env-var driven and robust:
- OPENAI_API_KEY
- TRIAGE_LLM_MODEL
- TRIAGE_LLM_TEMPERATURE
- TRIAGE_LLM_MAX_TOKENS
- TRIAGE_LLM_TIMEOUT

Prompt builder is integrated and always returns valid prompts.
Parser validates strict JSON and safely falls back when invalid.

### 8. Packaging and path stability

- Packaging includes all key subpackages.
- Editable install enables running commands from nested directories.
- Artifact paths are file-relative to avoid cwd breakage.

## Command Changes

Recommended commands from repo root:

```bash
python -m pytest -q
python -m triage_env.scripts.run_random --task task1
python -m triage_env.scripts.run_rule_based --task task2
python -m triage_env.scripts.run_llm_agent --task task3
python -m triage_env.scripts.train_rl
python -m triage_env.scripts.train_q_agent
python -m triage_env.scripts.run_benchmark
```
