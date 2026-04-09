import csv
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from triage_env.agents.random_agent import RandomAgent
from triage_env.agents.rl_agents import RLAgent
from triage_env.agents.rule_based_agent import RuleBasedAgent
from triage_env.agents.trained_q_agent import TrainedQAgent
from triage_env.config import get_runtime_config
from triage_env.evaluation.evaluator import evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
STRICT_BENCHMARK_ERRORS = os.getenv("TRIAGE_BENCHMARK_STRICT_ERRORS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _read_checkpoint_metadata(model_path: Path) -> dict:
    if not model_path.exists():
        return {}

    if model_path.suffix == ".json":
        try:
            with model_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return dict(data.get("metadata", {}))
        except Exception:
            return {}

    sidecar = Path(f"{model_path}.meta.json")
    if sidecar.exists():
        try:
            with sidecar.open("r", encoding="utf-8") as handle:
                return dict(json.load(handle))
        except Exception:
            return {}
    return {}


def _checkpoint_status(metadata: dict, requested_task: str | None, model_path: Path) -> tuple[str, str | None]:
    if not metadata:
        return "unknown", f"checkpoint metadata missing for {model_path.name}"

    source_task = metadata.get("task")
    if requested_task is not None and source_task is not None and source_task != requested_task:
        return "stale", f"checkpoint task={source_task} does not match evaluation task={requested_task}"

    if metadata.get("training_version") != 2:
        return "stale", f"checkpoint training_version={metadata.get('training_version')} is not current"

    return "fresh", None


def _build_agent(agent_name: str, q_model_path: Path, rl_model_path: Path, requested_task: str | None):
    if agent_name == "RandomAgent":
        return RandomAgent()
    if agent_name == "RuleBasedAgent":
        return RuleBasedAgent()
    if agent_name == "LLMAgent":
        from triage_env.agents.llm_agent import LLMAgent

        return LLMAgent()
    if agent_name == "TrainedQAgent":
        if not q_model_path.exists():
            return None
        agent = TrainedQAgent(str(q_model_path))
        agent.checkpoint_path = str(q_model_path)
        metadata = _read_checkpoint_metadata(q_model_path)
        status, warning = _checkpoint_status(metadata, requested_task, q_model_path)
        agent.checkpoint_metadata = metadata
        agent.checkpoint_warning = warning or None
        agent.checkpoint_status = status
        return agent
    if agent_name == "RLAgent":
        rl_agent = RLAgent()
        rl_agent.checkpoint_path = str(rl_model_path)
        if rl_model_path.exists():
            rl_agent.load(str(rl_model_path))
        metadata = _read_checkpoint_metadata(rl_model_path)
        status, warning = _checkpoint_status(metadata, requested_task, rl_model_path)
        rl_agent.checkpoint_metadata = metadata
        rl_agent.checkpoint_warning = warning or None
        rl_agent.checkpoint_status = status
        rl_agent.epsilon = 0.0
        return rl_agent
    return None


def benchmark_agents(
    num_episodes: int | None = None,
    task: str | None = None,
    agent_name: str | None = None,
    q_model_path: str | None = None,
    rl_model_path: str | None = None,
) -> list[dict]:
    runtime = get_runtime_config()
    episodes = num_episodes or runtime.eval_episodes
    default_q_model_path = Path(q_model_path) if q_model_path else (PACKAGE_ROOT / "training" / "q_agent.pkl")
    if rl_model_path:
        resolved_rl_model_path = Path(rl_model_path)
    elif task == "task3":
        resolved_rl_model_path = PACKAGE_ROOT / "training" / "triage_rl_qtable_task3.json"
    elif task == "task2":
        resolved_rl_model_path = PACKAGE_ROOT / "training" / "triage_rl_qtable_task2.json"
    else:
        resolved_rl_model_path = PACKAGE_ROOT / "training" / "triage_rl_qtable.json"

    requested_agent_names = (
        [agent_name]
        if agent_name is not None
        else ["RandomAgent", "RuleBasedAgent", "LLMAgent", "TrainedQAgent", "RLAgent"]
    )

    tasks = [task] if task else list(TASK_CONFIGS.keys())

    results: list[dict] = []
    for task_name in tasks:
        task_config = TASK_CONFIGS[task_name]
        task_specific_q_model = PACKAGE_ROOT / "training" / f"q_agent_{task_name}.pkl"
        resolved_q_model_path = (
            task_specific_q_model
            if q_model_path is None and task_specific_q_model.exists()
            else default_q_model_path
        )

        filtered_agents: dict[str, object] = {}
        for name in requested_agent_names:
            built_agent = _build_agent(name, resolved_q_model_path, resolved_rl_model_path, task_name)
            if built_agent is not None:
                filtered_agents[name] = built_agent

        for _, agent in filtered_agents.items():
            try:
                summary, _ = evaluate_agent(
                    env_class=TriageEnvironment,
                    agent=agent,
                    task=task_name,
                    num_episodes=episodes,
                    max_steps=task_config.max_steps,
                )
            except Exception as exc:
                if STRICT_BENCHMARK_ERRORS:
                    raise
                LOGGER.warning(
                    "Skipping agent %s on task %s due to evaluation error: %s",
                    getattr(agent, "name", agent.__class__.__name__),
                    task_name,
                    exc,
                )
                continue
            if summary.get("checkpoint_warning"):
                print(f"[checkpoint-warning] {summary['agent_name']} @ {task_name}: {summary['checkpoint_warning']}")
            results.append(summary)

    return results


def save_summary_csv(results: Iterable[dict], output_path: str) -> None:
    fieldnames = [
        "task",
        "agent_name",
        "episodes",
        "avg_total_reward",
        "avg_episode_length",
        "avg_steps",
        "avg_survivors",
        "avg_deaths",
        "survival_rate",
        "critical_survival_rate",
        "avg_health_alive",
        "stabilization_rate",
        "avg_stabilization_rate",
        "invalid_action_count",
        "avg_invalid_action_count",
        "success_rate",
        "avg_unseen_states",
    ]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            serializable = dict(row)
            if "avg_steps" in serializable and "avg_episode_length" not in serializable:
                serializable["avg_episode_length"] = serializable["avg_steps"]
            if "avg_stabilization_rate" in serializable and "stabilization_rate" not in serializable:
                serializable["stabilization_rate"] = serializable["avg_stabilization_rate"]
            if "avg_invalid_action_count" in serializable and "invalid_action_count" not in serializable:
                serializable["invalid_action_count"] = serializable["avg_invalid_action_count"]
            serializable.pop("action_distribution", None)
            serializable.pop("avg_action_distribution", None)
            serializable.pop("deaths_by_severity", None)
            serializable.pop("resource_utilization", None)
            serializable.pop("failure_reason_counts", None)
            serializable.pop("terminal_diagnostics", None)
            serializable.pop("checkpoint_metadata", None)
            serializable.pop("checkpoint_warning", None)
            serializable.pop("checkpoint_status", None)
            serializable.pop("checkpoint_path", None)
            writer.writerow(serializable)
