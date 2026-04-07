from dataclasses import dataclass

from triage_env.tasks import TASK_TARGETS


@dataclass(frozen=True)
class Task2Assessment:
    agent_name: str
    avg_total_reward: float
    critical_survival_rate: float
    success_rate: float
    ventilator_utilization: float
    invalid_action_count: float
    rule_based_reward: float
    meets_critical_survival: bool
    preferred_critical_band: bool
    meets_success_rate: bool
    beats_rule_based_reward: bool
    uses_ventilator_meaningfully: bool
    avoids_ventilator_overuse: bool
    zero_invalid_actions: bool
    failure_modes: tuple[str, ...]

    @property
    def meets_targets(self) -> bool:
        return (
            self.meets_critical_survival
            and self.meets_success_rate
            and self.beats_rule_based_reward
            and self.uses_ventilator_meaningfully
            and self.avoids_ventilator_overuse
            and self.zero_invalid_actions
        )


def assess_task2_summary(summary: dict, rule_based_reward: float) -> Task2Assessment:
    targets = TASK_TARGETS["task2"]
    critical = float(summary["critical_survival_rate"])
    success = float(summary["success_rate"])
    reward = float(summary["avg_total_reward"])
    ventilator_util = float(summary["resource_utilization"].get("ventilators", 0.0))
    invalid_actions = float(summary["invalid_action_count"])

    meets_critical = critical >= targets.critical_survival_min
    preferred_band = (
        targets.critical_survival_max is None
        or critical <= targets.critical_survival_max
    ) and meets_critical
    meets_success = success >= targets.success_rate_min
    beats_rule_based = reward > rule_based_reward
    uses_ventilator = ventilator_util >= targets.min_ventilator_utilization
    avoids_ventilator_overuse = (
        targets.max_ventilator_utilization is None
        or ventilator_util <= targets.max_ventilator_utilization
    )
    zero_invalid_actions = invalid_actions <= targets.max_invalid_actions

    failure_modes: list[str] = []
    if not meets_critical:
        failure_modes.append("critical_survival_too_low")

    if not meets_success:
        failure_modes.append("success_rate_too_low")
    if not beats_rule_based:
        failure_modes.append("reward_not_above_rule_based")
    if not uses_ventilator:
        failure_modes.append("ventilator_use_too_low")
    if not avoids_ventilator_overuse:
        failure_modes.append("ventilator_overuse")
    if not zero_invalid_actions:
        failure_modes.append("invalid_actions_present")

    if not preferred_band:
        failure_modes.append("advisory: over-conservative critical policy")

    return Task2Assessment(
        agent_name=str(summary.get("agent_name", "unknown")),
        avg_total_reward=reward,
        critical_survival_rate=critical,
        success_rate=success,
        ventilator_utilization=ventilator_util,
        invalid_action_count=invalid_actions,
        rule_based_reward=rule_based_reward,
        meets_critical_survival=meets_critical,
        preferred_critical_band=preferred_band,
        meets_success_rate=meets_success,
        beats_rule_based_reward=beats_rule_based,
        uses_ventilator_meaningfully=uses_ventilator,
        avoids_ventilator_overuse=avoids_ventilator_overuse,
        zero_invalid_actions=zero_invalid_actions,
        failure_modes=tuple(failure_modes),
    )
