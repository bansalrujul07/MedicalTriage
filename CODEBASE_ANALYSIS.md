# MedicalTriage Codebase Analysis

Date: 2026-04-07
Scope: Full repository review of environment logic, agents, training/evaluation pipeline, scripts, packaging, docs, and tests.

## 1. Executive Summary

This repository contains a working triage simulation core and passing unit tests for the environment itself, but the surrounding training/evaluation ecosystem is partially broken due to naming drift and API mismatches.

In short:
- The core environment loop is functional and reasonably well-shaped for RL experimentation.
- Most script entrypoints for RL/Q-learning training and comparison are currently not runnable as-is.
- Documentation and examples are partially stale and describe an older message-echo API that no longer matches the triage action schema.
- Packaging configuration is incomplete for distributable usage.

## 2. What The System Is Doing

### 2.1 Core Runtime Model

The main simulation is implemented in `TriageEnvironment` and follows a standard episodic loop:
1. `reset()` initializes 3 patients and limited resources.
2. `step(action)` processes one action (`treat`, `allocate_ventilator`, `wait`).
3. Reward is computed from:
   - immediate action quality,
   - time progression penalties,
   - health delta,
   - global stability bonus,
   - terminal reward at episode end.
4. Episode ends on step limit, all-dead state, or all-alive stabilized threshold.

Evidence:
- `triage_env/server/triage_env_environment.py:39`
- `triage_env/server/triage_env_environment.py:63`
- `triage_env/server/triage_env_environment.py:176`
- `triage_env/server/triage_env_environment.py:190`
- `triage_env/server/triage_env_environment.py:304`

### 2.2 API Surface

- Client payload shape is action-first (`action_type`, `patient_id`), not message-first.
- Observation includes `patients`, `resources`, `step_count`, `message`, `reward`, `done`, `metadata`.

Evidence:
- `triage_env/client.py:12`
- `triage_env/models.py:20`
- `triage_env/models.py:25`

### 2.3 Agent Layer

Current agents include:
- `RandomAgent`: random valid action among wait/treat (does not use ventilators).
- `RuleBasedAgent`: treats alive patient with lowest health.
- `LLMAgent`: builds prompt from patient status and parses JSON response.
- RL/Q-learning implementations exist but are inconsistent across files.

Evidence:
- `triage_env/agents/random_agent.py:8`
- `triage_env/agents/rule_based_agent.py:10`
- `triage_env/agents/llm_agent.py:19`
- `triage_env/agents/rl_agents.py:13`
- `triage_env/agents/q_learning_agents.py:9`

## 3. Validation Performed

### 3.1 Tests

Executed:
- `python -m pytest -q`

Result:
- 17 passed

Interpretation:
- Environment core behavior is stable for covered scenarios.
- Passing tests do not guarantee script/packaging/training pipeline health.

### 3.2 Compile/Syntax Check

Executed:
- `python -m compileall -q triage_env`

Result:
- No syntax compile errors.

Interpretation:
- Most breakages are semantic/runtime (imports, wrong API assumptions), not syntax errors.

### 3.3 Runtime Checks For Entry Points

Validated failures:
- `triage_env.scripts.train_rl` fails due to missing module `triage_env.agents.rl_agent`.
- `triage_env.training.train_q_agent` fails due to missing module `triage_env.agents.q_learning_agent`.
- `triage_env.scripts.compare_baselines` fails due to importing non-existent `evaluate` symbol.
- `training.rollout.run_episode` fails because `env.reset(task=...)` passes unsupported kwarg.
- `RLAgent.act` fails because `observation.task` does not exist in model.

## 4. Findings (Prioritized)

## Critical

1. Broken RL/Q-learning import paths (hard runtime failure)
- `trained_q_agent.py` imports `triage_env.agents.q_learning_agent`, but file is `q_learning_agents.py`.
- `train_q_agent.py` uses same bad import.
- Multiple scripts import `triage_env.agents.rl_agent`, but file is `rl_agents.py`.

Evidence:
- `triage_env/agents/trained_q_agent.py:1`
- `triage_env/training/train_q_agent.py:3`
- `triage_env/scripts/train_rl.py:3`
- `triage_env/scripts/evaluate_all_agents.py:5`
- `triage_env/scripts/evaluate_rl.py:3`

Impact:
- RL and Q-learning workflows are effectively unusable without manual fixes.

2. Training rollout uses incompatible environment API
- `run_episode()` calls `env.reset(task=task)`, but `TriageEnvironment.reset()` accepts no `task` argument.

Evidence:
- `triage_env/training/rollout.py:2`
- `triage_env/server/triage_env_environment.py:39`

Impact:
- Any pipeline depending on `training.rollout.run_episode` crashes immediately.

3. RL state encoding relies on nonexistent observation field
- `RLAgent._state_key()` accesses `observation.task`, not present in `TriageObservation`.

Evidence:
- `triage_env/agents/rl_agents.py:33`
- `triage_env/agents/rl_agents.py:44`
- `triage_env/models.py:25`

Impact:
- RL action selection and updates crash at runtime.

## High

4. Evaluator API mismatch across scripts
- `evaluation/evaluator.py` defines `evaluate_agent`, but several scripts import/use `evaluate`.

Evidence:
- `triage_env/evaluation/evaluator.py:22`
- `triage_env/scripts/compare_baselines.py:5`
- `triage_env/scripts/evaluate_all_agents.py:6`
- `triage_env/scripts/evaluate_rule_based_agent.py:4`
- `triage_env/scripts/evaluate_random_agent.py:4`

Impact:
- Baseline comparison scripts fail or require ad-hoc edits.

5. Packaging metadata omits major subpackages
- `pyproject.toml` only includes `triage_env` and `triage_env.server` in setuptools package list.
- `triage_env.agents`, `triage_env.evaluation`, `triage_env.training`, `triage_env.scripts` are not packaged for distribution.

Evidence:
- `triage_env/pyproject.toml:44`

Impact:
- Installed package may work partially in development but fails in clean/distributed usage.

6. README examples are stale and describe old message-echo API
- Uses `TriageAction(message=...)` and `observation.echoed_message`, which are not in current models.

Evidence:
- `README.md:94`
- `README.md:100`
- `triage_env/models.py:20`
- `triage_env/models.py:25`

Impact:
- New contributors receive incorrect onboarding instructions and hit immediate errors.

## Medium

7. Concurrency intent mismatch between environment and app settings
- Environment declares `SUPPORTS_CONCURRENT_SESSIONS = True`.
- Server app is configured with `max_concurrent_envs=1`.

Evidence:
- `triage_env/server/triage_env_environment.py:24`
- `triage_env/server/app.py:52`

Impact:
- Performance/scaling behavior may not match expectations from code comments/docs.

8. Unused/partially integrated prompt tooling
- `prompt_builder.py` defines a richer prompt pipeline but is not integrated into `LLMAgent`.
- Also returns nothing when no alive patients (return path only inside `if sorted_alive`).

Evidence:
- `triage_env/agents/prompt_builder.py:7`
- `triage_env/agents/prompt_builder.py:27`
- `triage_env/agents/prompt_builder.py:35`
- `triage_env/agents/llm_agent.py:20`

Impact:
- Prompt quality and safety controls are fragmented; hidden bug in edge state if reused.

9. Difficulty/task concept is declared but not used in environment dynamics
- `difficulty` exists in constructor but does not influence reset distributions or transition behavior.

Evidence:
- `triage_env/server/triage_env_environment.py:26`
- `triage_env/server/triage_env_environment.py:28`
- `triage_env/server/triage_env_environment.py:39`

Impact:
- Evaluation across "easy/medium/hard" in scripts is currently nominal, not environmental.

## Low

10. Duplicate/parallel script ecosystems increase drift risk
- Similar logic appears under both `triage_env/evaluation` and `triage_env/scripts` with inconsistent imports.

Evidence:
- `triage_env/evaluation/run_benchmark.py:1`
- `triage_env/scripts/compare_baselines.py:1`
- `triage_env/evaluation/run_rule_based.py:1`
- `triage_env/scripts/run_random.py:1`

Impact:
- Maintenance burden and future regressions increase.

11. Trailing whitespace / formatting cleanliness in some modules
- Not functionally harmful but indicates uneven code hygiene.

Evidence:
- `triage_env/agents/llm_agent.py:75`

## 5. Strengths

1. Core environment logic is coherent and test-covered.
- Reward decomposition is explicit and auditable via metadata (`reward_breakdown`).
- Resource reset and patient progression are deterministic and understandable.

2. Unit tests validate important environment invariants.
- Reset, step progression, invalid action penalties, death behavior, and done state are covered.

3. Model layer is clear and strongly typed.
- Pydantic models for action/observation/state improve interface clarity.

## 6. Gaps In Current Test Strategy

Current tests focus almost exclusively on environment internals and do not cover:
- Script entrypoint execution (`triage_env/scripts/*`)
- Import path correctness after packaging/install
- RL/Q-learning training loops
- LLM integration safety and fallback behavior
- README quickstart correctness

Practical result: core tests pass while user-facing workflows remain broken.

## 7. Recommended Remediation Plan

### Phase 1 (Stabilize Runtime)
1. Normalize module names/imports:
   - pick singular or plural convention (`rl_agent` vs `rl_agents`, `q_learning_agent` vs `q_learning_agents`) and align all imports.
2. Fix evaluator API usage:
   - either expose `evaluate()` wrapper in evaluator module or update all scripts to `evaluate_agent`.
3. Repair rollout/task wiring:
   - remove `task` kwarg in reset call, or formally add task support in environment model/state.
4. Fix RL observation schema usage:
   - replace `observation.task` with valid features from current observation/state.

### Phase 2 (Consistency + Packaging)
1. Update README and examples to current action schema (`action_type`, `patient_id`).
2. Update `pyproject.toml` to include all importable subpackages.
3. Consolidate duplicate script sets into one canonical runner path.

### Phase 3 (Quality + Coverage)
1. Add smoke tests that execute each main script module.
2. Add regression tests for RL and Q-learning initialization paths.
3. Add docs-validation test to ensure README snippets match public models.

## 8. Architecture Snapshot

Primary flow:
- Agent -> `TriageAction` -> `TriageEnvironment.step()` -> `TriageObservation` + reward metadata
- Training/evaluation wrappers orchestrate repeated episodes and aggregate metrics
- OpenEnv server adapter exposes environment over HTTP/WebSocket

Data contracts are good at the model level, but orchestration layers have drifted from those contracts.

## 9. Bottom Line

The simulation kernel is in good shape and test-backed, but your surrounding experimentation stack is in a partially broken state due to API and naming drift. If your goal is to iterate quickly on agent strategies, you should first complete Phase 1 fixes; otherwise most RL/evaluation scripts will continue to fail despite green unit tests.
