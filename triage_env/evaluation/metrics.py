from dataclasses import dataclass

from triage_env.tasks import TASK_CONFIGS


@dataclass
class EpisodeMetrics:
    task: str
    total_reward: float
    steps: int
    survivors: int
    deaths: int
    survival_rate: float
    critical_survival_rate: float
    avg_health_alive: float
    stabilization_rate: float
    action_distribution: dict[str, float]
    invalid_action_count: int
    deaths_by_severity: dict[str, int]
    resource_utilization: dict[str, float]
    success: bool
    terminal_failure_reason: str | None = None
    terminal_diagnostics: dict[str, float | int | dict[str, int]] | None = None


def compute_episode_metrics(final_observation, total_reward: float, action_counts: dict[str, int] | None = None):
    alive_patients = [p for p in final_observation.patients if p.alive]
    dead_patients = [p for p in final_observation.patients if not p.alive]
    total_patients = len(final_observation.patients)
    metadata = final_observation.metadata or {}

    critical_patients = [p for p in final_observation.patients if p.severity == "critical"]
    critical_survivors = [p for p in critical_patients if p.alive]

    deaths_by_severity = {
        "critical": sum(1 for p in dead_patients if p.severity == "critical"),
        "severe": sum(1 for p in dead_patients if p.severity == "severe"),
        "moderate": sum(1 for p in dead_patients if p.severity == "moderate"),
        "mild": sum(1 for p in dead_patients if p.severity == "mild"),
    }

    avg_health_alive = (
        sum(p.health for p in alive_patients) / len(alive_patients)
        if alive_patients
        else 0.0
    )

    counts = action_counts or {"wait": 0, "treat": 0, "allocate_ventilator": 0}
    total_actions = sum(counts.values())
    if total_actions == 0:
        action_distribution = {k: 0.0 for k in counts}
    else:
        action_distribution = {
            k: float(v) / float(total_actions) for k, v in counts.items()
        }

    resource_usage = metadata.get("resource_usage", {})
    initial_resources = metadata.get("initial_resources", {})
    medics_total = float(initial_resources.get("medics_available", 0) * max(1, final_observation.step_count))
    ventilators_total = float(initial_resources.get("ventilators_available", 0) * max(1, final_observation.step_count))
    medics_used = float(resource_usage.get("medics_used", 0))
    ventilators_used = float(
        resource_usage.get(
            "ventilator_steps_used",
            resource_usage.get("ventilators_used", 0),
        )
    )

    resource_utilization = {
        "medics": (medics_used / medics_total) if medics_total > 0 else 0.0,
        "ventilators": (ventilators_used / ventilators_total) if ventilators_total > 0 else 0.0,
    }

    survival_rate = (len(alive_patients) / total_patients) if total_patients else 0.0
    critical_survival_rate = (
        len(critical_survivors) / len(critical_patients)
        if critical_patients
        else 1.0
    )
    task_name = str(metadata.get("task", "task2"))
    task_config = TASK_CONFIGS.get(task_name)
    terminal_failure_reason = None
    terminal_diagnostics = {
        "survival_rate": survival_rate,
        "critical_survival_rate": critical_survival_rate,
        "avg_health_alive": float(avg_health_alive),
        "alive_count": len(alive_patients),
        "deaths_by_severity": deaths_by_severity,
        "ventilator_utilization": resource_utilization["ventilators"],
        "ventilator_occupancy": resource_utilization["ventilators"],
    }
    if task_config is not None:
        success = (
            survival_rate >= task_config.terminal_success_survival_rate
            and avg_health_alive >= task_config.terminal_success_avg_health
        )
        if not success:
            failed_survival = survival_rate < task_config.terminal_success_survival_rate
            failed_health = avg_health_alive < task_config.terminal_success_avg_health
            if failed_survival and failed_health:
                terminal_failure_reason = "failed_both"
            elif failed_survival:
                terminal_failure_reason = "failed_survival_threshold"
            elif failed_health:
                terminal_failure_reason = "failed_avg_health_threshold"
    else:
        # Backward-compatible fallback for unknown task metadata.
        success = survival_rate >= 0.6 and critical_survival_rate >= 0.5

    return EpisodeMetrics(
        task=str(metadata.get("task", "task2")),
        total_reward=float(total_reward),
        steps=int(final_observation.step_count),
        survivors=len(alive_patients),
        deaths=len(dead_patients),
        survival_rate=survival_rate,
        critical_survival_rate=critical_survival_rate,
        avg_health_alive=float(avg_health_alive),
        stabilization_rate=(len(alive_patients) / total_patients) if total_patients else 0.0,
        action_distribution=action_distribution,
        invalid_action_count=int(metadata.get("invalid_action_count", 0)),
        deaths_by_severity=deaths_by_severity,
        resource_utilization=resource_utilization,
        success=success,
        terminal_failure_reason=terminal_failure_reason,
        terminal_diagnostics=terminal_diagnostics,
    )