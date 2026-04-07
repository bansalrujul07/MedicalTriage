# Task Architecture

This document defines the operational differences between `task1`, `task2`, and `task3`, and the expected behavior progression for triage agents.

## Design Goals

- Keep one stable action contract for all tasks.
- Increase pressure gradually across tasks.
- Make reward shaping explicit and task-specific.
- Support direct comparison of Random, Rule-based, LLM, RL, and Q agents.

## Shared Contract Across Tasks

All tasks use the same action schema:
- `treat(patient_id)`
- `allocate_ventilator(patient_id)`
- `wait(patient_id=-1)`

All tasks expose the same observation schema:
- `patients`
- `resources`
- `step_count`
- `reward`
- `done`
- `metadata`

The environment implementation is task-driven through `TaskConfig` entries in [triage_env/tasks.py](../tasks.py).

## Task Progression

### task1: Baseline triage

Operational profile:
- Smaller patient set.
- Lower deterioration rates.
- Strong positive reward for effective treatment.
- Lower penalties for mistakes than higher tasks.

Intended learning signal:
- Teaches basic patient prioritization.
- Encourages avoiding unnecessary waits.

Expected agent behavior:
- `RuleBasedAgent` should achieve high survival and stable outcomes.
- `RandomAgent` should be clearly weaker but non-catastrophic.
- RL/Q agents should converge quickly relative to other tasks.

### task2: Moderate pressure

Operational profile:
- More patients and longer episode horizon than task1.
- Higher deterioration and death risk.
- More meaningful resource contention.
- Stronger penalties for invalid or low-value actions.

Intended learning signal:
- Forces scheduling tradeoffs between treatment value and urgency.
- Requires more robust handling of mixed severities.

Expected agent behavior:
- `RuleBasedAgent` remains a strong baseline but with lower margin than task1.
- RL/Q agents should show material gains over `RandomAgent`.
- LLM outcomes should depend on action selection consistency.

### task3: High pressure

Operational profile:
- Largest pressure profile.
- Highest deterioration rates.
- Tightest resources relative to acuity.
- Harsh death penalties and stricter reward tradeoffs.

Intended learning signal:
- Prioritizes critical survival under severe constraints.
- Rewards robust policy behavior over naive local optimization.

Expected agent behavior:
- `RandomAgent` should perform poorly.
- `RuleBasedAgent` remains competitive but may not be globally optimal in all runs.
- Well-trained RL/Q agents should close or exceed baseline performance where policy generalization is effective.

## Reward Semantics

Reward is composed from explicit components in `RewardWeights`:
- Positive components:
  - successful treatment and ventilator allocation
  - stabilization and terminal success bonuses
  - all-critical-survive bonus
- Negative components:
  - ineffective or invalid actions
  - unnecessary waits
  - severity-weighted death penalties

This structure supports task-specific tuning while keeping reward categories stable for analytics.

## Evaluation Expectations

The evaluator reports task- and safety-oriented metrics including:
- `survival_rate`
- `critical_survival_rate`
- `success_rate`
- `invalid_action_count`
- `deaths_by_severity`
- `resource_utilization`

Interpretation guideline:
- Moving from `task1` to `task3`, absolute scores should generally degrade.
- Stronger agents should degrade more gracefully than weaker agents.
- Improvements are meaningful when they increase critical survival without exploding invalid actions or resource waste.

## Practical Benchmarking Pattern

Use multi-task benchmarking to validate progression:

```bash
python -m triage_env.scripts.run_benchmark --tasks task1,task2,task3
```

Agent-filtered benchmarking example:

```bash
python -m triage_env.scripts.run_benchmark --tasks task1,task2,task3 --agents RuleBasedAgent,RLAgent
```

## Notes for Extension

When adding a new task profile:
- Add one `TaskConfig` entry in [triage_env/tasks.py](../tasks.py).
- Keep action and observation schema unchanged.
- Preserve metric names so existing benchmark tooling remains compatible.
- Add tests for reset profile, reward behavior, and benchmark smoke coverage.
