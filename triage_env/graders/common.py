from __future__ import annotations

import json
import math
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

# Ensure triage_env package can be imported when graders are executed via file path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triage_env.agents.random_agent import RandomAgent
from triage_env.agents.rl_agents import RLAgent
from triage_env.agents.base_agent import BaseAgent
from triage_env.agents.rule_based_agent import RuleBasedAgent
from triage_env.agents.trained_q_agent import TrainedQAgent
from triage_env.evaluation.evaluator import evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS, TASK_TARGETS

GRADER_VERSION = "v2.3"
SCORE_EPSILON = 1e-4
CONFIGURE_ROOT_LOGGER = os.getenv("TRIAGE_GRADER_CONFIGURE_ROOT_LOGGER", "").strip().lower() in {
    "1", "true", "yes", "on",
}

LOGGER = logging.getLogger(__name__)

if CONFIGURE_ROOT_LOGGER and not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("TRIAGE_LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

_RESULT_SCHEMA = {
    "type": "object",
    "required": ["grader_version", "task", "task_id", "episodes", "score", "reward", "score_range", "components", "signals", "summary"],
    "properties": {
        "grader_version": {"type": "string"},
        "task": {"type": "string"},
        "task_id": {"type": "string"},
        "episodes": {"type": "integer", "minimum": 1},
        "score": {"type": "number", "minimum": SCORE_EPSILON, "maximum": 1.0 - SCORE_EPSILON},
        "reward": {"type": "number"},
        "score_range": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "components": {"type": "object"},
        "signals": {"type": "object"},
        "summary": {"type": "object"},
    },
}


class SafeAgent(BaseAgent):
    def __init__(self, wrapped_agent: BaseAgent):
        self._wrapped_agent = wrapped_agent

    @property
    def name(self) -> str:
        return getattr(self._wrapped_agent, "name", self._wrapped_agent.__class__.__name__)

    def reset(self):
        try:
            reset = getattr(self._wrapped_agent, "reset", None)
            if callable(reset):
                reset()
        except Exception:
            LOGGER.exception("SafeAgent reset failed; continuing")

    def act(self, observation):
        try:
            return self._wrapped_agent.act(observation)
        except Exception:
            LOGGER.exception("Wrapped agent crashed; returning safe wait action")
            return _safe_wait_action()


def _safe_wait_action():
    from triage_env.models import TriageAction
    return TriageAction(action_type="wait", patient_id=-1)


def _resolve_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _fallback_grade(task_name: str, episodes: int, reason: str) -> dict[str, Any]:
    safe_score = _clip_open_01(0.5)
    return {
        "grader_version": GRADER_VERSION,
        "status": "error",
        "task": task_name,
        "task_id": task_name,
        "episodes": episodes,
        "score": safe_score,
        "reward": safe_score,
        "score_range": [0.0, 1.0],
        "components": {
            "rollout_achievement": safe_score,
            "safety_errors": safe_score,
            "efficiency": safe_score,
            "task_specific": safe_score,
        },
        "signals": {
            "fallback": 1.0,
            "error": reason,
        },
        "summary": {
            "task": task_name,
            "fallback_reason": reason,
            "success_rate": safe_score,
            "survival_rate": safe_score,
            "critical_survival_rate": safe_score,
            "avg_total_reward": safe_score,
        },
    }


def _validate_result_schema(result: dict[str, Any]) -> dict[str, Any]:
    try:
        validate(instance=result, schema=_RESULT_SCHEMA)
        return result
    except ValidationError as exc:
        LOGGER.error("Grader result failed schema validation: %s", exc)
        return _fallback_grade(result.get("task", "unknown"), int(result.get("episodes", 1)), f"schema:{exc}")


def _grade_task_impl(task_name: str, episodes: int) -> dict[str, Any]:
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {task_name}")

    task_config = TASK_CONFIGS[task_name]
    agent, agent_meta = _build_evaluated_agent(task_name)
    agent = SafeAgent(agent)
    
    summary, _ = evaluate_agent(
        env_class=TriageEnvironment,
        agent=agent,
        task=task_name,
        num_episodes=episodes,
        max_steps=task_config.max_steps,
    )

    components = _compute_components(task_name, summary)
    final_score = _compute_final_score(components)

    return {
        "grader_version": GRADER_VERSION,
        "task": task_name,
        "task_id": task_name,
        "episodes": episodes,
        "score": final_score,
        "reward": final_score,
        "score_range": [0.0, 1.0],
        "components": {
            "rollout_achievement": components["rollout_achievement"],
            "safety_errors": components["safety_errors"],
            "efficiency": components["efficiency"],
            "task_specific": components["task_specific"],
        },
        "signals": {
            "selected_agent": agent_meta.get("selected_agent"),
            "selected_checkpoint": agent_meta.get("checkpoint"),
            "selection_reason": agent_meta.get("selection_reason"),
            "survival_rate": components["survival_rate"],
            "critical_survival_rate": components["critical_survival_rate"],
            "success_rate": components["success_rate"],
            "reward_norm": components["reward_norm"],
            "invalid_rate": components["invalid_rate"],
            "stabilization_threshold": components["stabilization_threshold"],
        },
        "summary": summary,
    }


def _build_evaluated_agent(task_name: str):
    package_root = Path(__file__).resolve().parents[1]

    injected_api_key = os.getenv("API_KEY", "").strip()
    injected_base_url = os.getenv("API_BASE_URL", "").strip()
    proxy_env_detected = bool(injected_api_key or injected_base_url)

    requested = os.getenv("TRIAGE_GRADER_AGENT", "").strip().lower()
    if not requested:
        if proxy_env_detected:
            from triage_env.agents.llm_agent import LLMAgent
            from triage_env.config import get_llm_config

            llm_config = get_llm_config()
            llm_config = replace(
                llm_config,
                api_key=injected_api_key or llm_config.api_key,
                base_url=injected_base_url or llm_config.base_url,
            )
            final_base = llm_config.base_url
            
            return LLMAgent(config=llm_config), {
                "selected_agent": "LLMAgent",
                "selection_reason": "validator-proxy-env-detected",
                "api_endpoint": final_base,
                "model": llm_config.model,
            }

        rl_path = _resolve_existing_path(
            [
                package_root / "training" / f"triage_rl_qtable_{task_name}.json",
                package_root / "training" / "triage_rl_qtable.json",
            ]
        )
        if rl_path is not None:
            agent = RLAgent(epsilon=0.0)
            agent.load(str(rl_path))
            agent.epsilon = 0.0
            return agent, {"selected_agent": "RLAgent", "checkpoint": str(rl_path)}

        q_path = _resolve_existing_path(
            [
                package_root / "training" / f"q_agent_{task_name}.pkl",
                package_root / "training" / "q_agent.pkl",
            ]
        )
        if q_path is not None:
            agent = TrainedQAgent(str(q_path))
            return agent, {"selected_agent": "TrainedQAgent", "checkpoint": str(q_path)}

        from triage_env.agents.llm_agent import LLMAgent
        from triage_env.config import get_llm_config
        
        llm_config = get_llm_config()
        if llm_config.api_key:
            return LLMAgent(config=llm_config), {
                "selected_agent": "LLMAgent",
                "selection_reason": "fallback-to-llm-proxy",
                "api_endpoint": llm_config.base_url,
                "model": llm_config.model,
            }

        # Final non-error fallback: use RLAgent even without checkpoint so
        # validator runs still exercise the RL policy path instead of returning
        # synthetic placeholder scores.
        return RLAgent(epsilon=0.0), {
            "selected_agent": "RLAgent",
            "selection_reason": "fallback-untrained-rl-no-checkpoint-no-api",
        }

    if requested in {"rulebased", "rulebasedagent"}:
        if proxy_env_detected:
            from triage_env.agents.llm_agent import LLMAgent
            from triage_env.config import get_llm_config

            llm_config = get_llm_config()
            llm_config = replace(
                llm_config,
                api_key=injected_api_key or llm_config.api_key,
                base_url=injected_base_url or llm_config.base_url,
            )
            final_base = llm_config.base_url

            return LLMAgent(config=llm_config), {
                "selected_agent": "LLMAgent",
                "selection_reason": "rulebased-overridden-by-validator-proxy-env",
                "api_endpoint": final_base,
                "model": llm_config.model,
            }
        return RuleBasedAgent(), {"selected_agent": "RuleBasedAgent", "selection_reason": "explicit"}

    if requested in {"random", "randomagent"}:
        return RandomAgent(), {"selected_agent": "RandomAgent", "selection_reason": "explicit"}

    if requested in {"trainedq", "trainedqagent", "q", "qlearning"}:
        q_model = os.getenv("TRIAGE_Q_MODEL_PATH", "").strip()
        candidates = [Path(q_model)] if q_model else []
        candidates.extend(
            [
                package_root / "training" / f"q_agent_{task_name}.pkl",
                package_root / "training" / "q_agent.pkl",
            ]
        )
        q_path = _resolve_existing_path(candidates)
        if q_path is None:
            raise FileNotFoundError("TRIAGE_GRADER_AGENT=trainedq requested but no Q checkpoint found")
        return TrainedQAgent(str(q_path)), {"selected_agent": "TrainedQAgent", "checkpoint": str(q_path)}

    if requested in {"rl", "rlagent"}:
        rl_model = os.getenv("TRIAGE_RL_MODEL_PATH", "").strip()
        candidates = [Path(rl_model)] if rl_model else []
        candidates.extend(
            [
                package_root / "training" / f"triage_rl_qtable_{task_name}.json",
                package_root / "training" / "triage_rl_qtable.json",
            ]
        )
        rl_path = _resolve_existing_path(candidates)
        agent = RLAgent(epsilon=0.0)
        if rl_path is None:
            return agent, {
                "selected_agent": "RLAgent",
                "selection_reason": "explicit-untrained-rl-no-checkpoint",
            }
        agent.load(str(rl_path))
        agent.epsilon = 0.0
        return agent, {"selected_agent": "RLAgent", "checkpoint": str(rl_path)}

    raise ValueError(f"Unsupported TRIAGE_GRADER_AGENT value: {requested}")


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clip_open_01(value: float) -> float:
    clipped = _clip_01(value)
    return SCORE_EPSILON + clipped * (1.0 - 2.0 * SCORE_EPSILON)


def _mean(*values: float) -> float:
    filtered = [float(v) for v in values]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def _safe_get(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalized_reward(avg_total_reward: float) -> float:
    return _clip_01(0.5 + 0.5 * math.tanh(avg_total_reward / 200.0))


def _compute_components(task_name: str, summary: dict[str, Any]) -> dict[str, float]:
    task_config = TASK_CONFIGS[task_name]
    num_patients = float(task_config.num_patients)
    max_steps = float(task_config.max_steps)
    stabilization_threshold = float(task_config.reward_weights.stabilization_threshold)

    survival_rate = _clip_01(_safe_get(summary, "survival_rate"))
    critical_survival_rate = _clip_01(_safe_get(summary, "critical_survival_rate"))
    success_rate = _clip_01(_safe_get(summary, "success_rate"))
    stabilization_rate = _clip_01(_safe_get(summary, "stabilization_rate"))
    avg_health_alive = _clip_01(_safe_get(summary, "avg_health_alive") / 100.0)
    avg_total_reward = _safe_get(summary, "avg_total_reward")
    avg_episode_length = _safe_get(summary, "avg_episode_length")
    avg_deaths = _safe_get(summary, "avg_deaths")
    invalid_action_count = _safe_get(summary, "invalid_action_count")

    vent_util = 0.0
    resources = summary.get("resource_utilization", {})
    if isinstance(resources, dict):
        vent_util = _clip_01(float(resources.get("ventilators", 0.0)))

    reward_norm = _normalized_reward(avg_total_reward)
    death_penalty = _clip_01(avg_deaths / max(1.0, num_patients))
    invalid_rate = invalid_action_count / max(1.0, max_steps)
    invalid_penalty = _clip_01(invalid_rate)
    step_efficiency = _clip_01(1.0 - (avg_episode_length / max(1.0, max_steps)))

    rollout_achievement = _clip_01(_mean(survival_rate, critical_survival_rate, success_rate))
    safety_errors = _clip_01(_mean(1.0 - death_penalty, 1.0 - invalid_penalty))
    efficiency = _clip_01(_mean(stabilization_rate, step_efficiency, reward_norm))

    if task_name == "task1":
        task_specific = _clip_01(_mean(avg_health_alive, success_rate, reward_norm))
    elif task_name == "task2":
        target = TASK_TARGETS.get("task2")
        if target is None:
            vent_balance = _clip_01(1.0 - abs(vent_util - 0.5) / 0.5)
        else:
            lower = float(target.min_ventilator_utilization)
            upper = float(target.max_ventilator_utilization or 1.0)
            if lower <= vent_util <= upper:
                vent_balance = 1.0
            elif vent_util < lower:
                margin = max(lower, 1e-6)
                vent_balance = _clip_01(1.0 - (lower - vent_util) / margin)
            else:
                margin = max(1.0 - upper, 1e-6)
                vent_balance = _clip_01(1.0 - (vent_util - upper) / margin)
        task_specific = _clip_01(_mean(critical_survival_rate, vent_balance, reward_norm))
    else:
        task_specific = _clip_01(_mean(critical_survival_rate, survival_rate, avg_health_alive))

    return {
        "rollout_achievement": rollout_achievement,
        "safety_errors": safety_errors,
        "efficiency": efficiency,
        "task_specific": task_specific,
        "reward_norm": reward_norm,
        "invalid_rate": _clip_01(invalid_rate),
        "stabilization_threshold": stabilization_threshold,
        "survival_rate": survival_rate,
        "critical_survival_rate": critical_survival_rate,
        "success_rate": success_rate,
    }


def _compute_final_score(components: dict[str, float]) -> float:
    score = (
        components["rollout_achievement"] * 0.40
        + components["safety_errors"] * 0.25
        + components["efficiency"] * 0.20
        + components["task_specific"] * 0.15
    )
    return _clip_open_01(score)


# FIX: Removed multiprocessing. Running synchronously prevents platform hangs and "Not enough tasks" errors.
def grade_task(task_name: str, episodes: int = 1) -> dict[str, Any]:
    try:
        # Execute evaluation synchronously.
        result = _grade_task_impl(task_name, episodes)
        return _validate_result_schema(result)
    except Exception as exc:
        LOGGER.exception("Unexpected grader failure for task %s", task_name)
        return _fallback_grade(task_name, episodes, f"{type(exc).__name__}:{exc}")


def print_grader_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))