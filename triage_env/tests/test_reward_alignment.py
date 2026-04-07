from triage_env.tasks import TASK_CONFIGS


def _mild_survival_upper_bound(task_name: str) -> float:
    cfg = TASK_CONFIGS[task_name]
    rw = cfg.reward_weights

    # Conservative bound for marginal "extra mild survival" upside:
    # - avoid mild death penalty
    # - mild treatment + health-gain component
    # - local stability contribution
    # Excludes episode-level success bonus because that reflects global completion,
    # not the incremental value of one mild patient.
    return (
        abs(rw.death_penalty_mild)
        + rw.successful_treat_mild
        + cfg.treatment_gain["mild"] * rw.health_gain_scale
        + rw.stabilization_bonus
    )


def test_critical_loss_always_worse_than_extra_mild_survival():
    for task_name, cfg in TASK_CONFIGS.items():
        rw = cfg.reward_weights
        critical_loss_cost = abs(rw.death_penalty_critical)
        mild_survival_gain = _mild_survival_upper_bound(task_name)

        assert critical_loss_cost > mild_survival_gain, (
            f"Reward misalignment in {task_name}: "
            f"critical_loss_cost={critical_loss_cost} must exceed "
            f"mild_survival_gain={mild_survival_gain}"
        )
