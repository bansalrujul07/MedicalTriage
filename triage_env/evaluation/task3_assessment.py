from dataclasses import dataclass

from triage_env.tasks import TASK_TARGETS


@dataclass(frozen=True)
class Task3Assessment:
    agent_name: str
    avg_total_reward: float
    critical_survival_rate: float
    success_rate: float
    ventilator_utilization: float
    invalid_action_count: float
    rule_based_reward: float
    meets_critical_survival: bool
    meets_success_rate: bool
    beats_rule_based_reward: bool
    uses_ventilator_meaningfully: bool
    zero_invalid_actions: bool
    failure_modes: tuple[str, ...]
    failure_reason_counts: dict[str, int]
    checkpoint_status: str | None = None
    checkpoint_warning: str | None = None

    @property
    def meets_targets(self) -> bool:
        return (
            self.meets_critical_survival
            and self.meets_success_rate
            and self.beats_rule_based_reward
            and self.uses_ventilator_meaningfully
            and self.zero_invalid_actions
        )

    @property
    def milestone_a(self) -> bool:
        return (
            self.success_rate > 0.0
            and self.zero_invalid_actions
            and self.beats_rule_based_reward
        )

    @property
    def milestone_b(self) -> bool:
        return (
            self.success_rate >= 0.15
            and self.critical_survival_rate >= 0.50
            and self.ventilator_utilization >= 0.20
        )

    @property
    def milestone_c(self) -> bool:
        return (
            self.success_rate >= 0.25
            and self.critical_survival_rate >= 0.60
        )


def assess_task3_summary(summary: dict, rule_based_reward: float) -> Task3Assessment:
    targets = TASK_TARGETS["task3"]
    critical = float(summary["critical_survival_rate"])
    success = float(summary["success_rate"])
    reward = float(summary["avg_total_reward"])
    ventilator_util = float(summary["resource_utilization"].get("ventilators", 0.0))
    invalid_actions = float(summary["invalid_action_count"])
    failure_reason_counts = dict(summary.get("failure_reason_counts", {}))

    meets_critical = critical >= targets.critical_survival_min
    meets_success = success >= targets.success_rate_min
    beats_rule_based = reward > rule_based_reward
    uses_ventilator = ventilator_util >= targets.min_ventilator_utilization
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
    if not zero_invalid_actions:
        failure_modes.append("invalid_actions_present")

    if failure_reason_counts:
        ordered = sorted(failure_reason_counts.items(), key=lambda item: (-item[1], item[0]))
        failure_modes.append("failure_reasons=" + ";".join(f"{k}:{v}" for k, v in ordered))

    return Task3Assessment(
        agent_name=str(summary.get("agent_name", "unknown")),
        avg_total_reward=reward,
        critical_survival_rate=critical,
        success_rate=success,
        ventilator_utilization=ventilator_util,
        invalid_action_count=invalid_actions,
        rule_based_reward=rule_based_reward,
        meets_critical_survival=meets_critical,
        meets_success_rate=meets_success,
        beats_rule_based_reward=beats_rule_based,
        uses_ventilator_meaningfully=uses_ventilator,
        zero_invalid_actions=zero_invalid_actions,
        failure_modes=tuple(failure_modes),
        failure_reason_counts=failure_reason_counts,
        checkpoint_status=summary.get("checkpoint_status"),
        checkpoint_warning=summary.get("checkpoint_warning"),
    )
