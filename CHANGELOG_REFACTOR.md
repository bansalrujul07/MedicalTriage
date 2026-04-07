# MedicalTriage Refactor Change Log

Date: 2026-04-07

## Summary

This document captures the end-to-end refactor and repair work performed to make the repository runnable, consistent, and production-ready while preserving triage environment semantics.

## Major Changes

### 1. Module and Import Consistency

- Standardized canonical modules:
  - triage_env.agents.rl_agents
  - triage_env.agents.q_learning_agents
- Added compatibility aliases:
  - triage_env.agents.rl_agent
  - triage_env.agents.q_learning_agent
- Normalized imports across training, evaluation, and scripts.

### 2. Environment Contract Alignment

- Kept the action contract as source of truth:
  - action_type
  - patient_id
- Refactored surrounding layers to use current observation/action models.
- Removed stale message-echo assumptions.

### 3. Training and Rollout Repairs

- Fixed rollout reset mismatch:
  - run_episode now calls env.reset() correctly.
- Kept backward-compatible task argument in rollout/trainer as ignored plumbing.
- Added shared state encoding for tabular RL/Q-learning.
- Fixed RL update stability for unseen action keys.

### 4. Evaluation Layer Unification

- Canonical evaluator API:
  - evaluate_agent(...)
- Added backward-compatible wrapper:
  - evaluate(env, agent, episodes=...)
- Added consistent aggregate outputs including:
  - avg_total_reward
  - avg_survivors
  - avg_deaths
  - avg_steps
  - avg_health_alive
  - avg_stabilization_rate
  - avg_action_distribution

### 5. LLM Agent Integration

- Added central environment-variable config layer.
- LLMAgent now:
  - reads OPENAI_API_KEY from env
  - supports TRIAGE_LLM_MODEL, TRIAGE_LLM_TEMPERATURE, TRIAGE_LLM_MAX_TOKENS, TRIAGE_LLM_TIMEOUT
  - uses integrated system/user prompt builders
  - enforces strict JSON action parsing
  - safely falls back on malformed output or missing API key
  - logs warnings rather than failing silently

### 6. Prompt and Parser Improvements

- Integrated prompt_builder into LLMAgent flow.
- Prompt builder now always returns a valid prompt.
- Added dedicated parser with robust JSON extraction and validation.

### 7. Packaging and Executability

- Fixed pyproject package mapping so triage_env is importable from nested directories.
- Added package init modules for agents/evaluation/training/scripts.
- Added top-level script wrappers under scripts/ for convenience.
- Canonical runnable module entrypoints:
  - triage_env.scripts.run_random
  - triage_env.scripts.run_rule_based
  - triage_env.scripts.run_llm_agent
  - triage_env.scripts.train_q_agent
  - triage_env.scripts.train_rl
  - triage_env.scripts.run_benchmark

### 8. Path Robustness Fixes

- Changed training/benchmark default artifact paths to file-relative resolution instead of cwd-relative strings.
- Removed a shadowing artifact directory that caused import failure when running from nested paths.

### 9. Documentation Updates

- Rewrote README to match the real action/observation API.
- Added MIGRATION.md with implementation notes and compatibility details.

### 10. Test Coverage Expansion

Added tests for:
- import smoke checks
- evaluator API compatibility
- rollout initialization
- state encoder behavior
- LLM parser behavior and fallback safety
- README contract sanity

## Validation Performed

- Full test suite pass:
  - 26 passed
- Smoke-run success for canonical scripts:
  - run_random
  - run_rule_based
  - run_llm_agent
  - train_q_agent
  - train_rl
  - run_benchmark

## How To Run

From project root:

```bash
python -m pytest -q
python -m triage_env.scripts.run_random
python -m triage_env.scripts.run_rule_based
python -m triage_env.scripts.run_llm_agent
python -m triage_env.scripts.train_q_agent
python -m triage_env.scripts.train_rl
python -m triage_env.scripts.run_benchmark
```

If running from nested directories, ensure editable install is present:

```bash
pip install -e ./triage_env
```

## Known Remaining Limitations

- Difficulty currently changes initial patient profiles only; transition/reward coefficients are not difficulty-specific.
- Legacy wrappers are retained for compatibility and can be removed in a later cleanup cycle.
