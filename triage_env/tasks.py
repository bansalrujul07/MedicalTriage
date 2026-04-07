from dataclasses import dataclass


# Reward tuning by difficulty level so task configs can reference one source.
DIFFICULTY_REWARD_PROFILE = {
    "easy": {
        "successful_treat_critical": 5.0,
        "episode_success_bonus": 20.0,
    },
    "medium": {
        "successful_treat_critical": 6.0,
        "episode_success_bonus": 24.0,
    },
    "hard": {
        "successful_treat_critical": 8.0,
        "episode_success_bonus": 30.0,
    },
}


@dataclass(frozen=True)
class RewardWeights:
    successful_treat_critical: float
    successful_treat_moderate: float
    successful_treat_mild: float
    successful_ventilator_allocation: float
    ineffective_treatment_penalty: float
    invalid_action_penalty: float
    unnecessary_wait_penalty: float
    death_penalty_critical: float
    death_penalty_moderate: float
    death_penalty_mild: float
    stabilization_bonus: float
    episode_success_bonus: float
    all_critical_survive_bonus: float
    health_gain_scale: float
    episode_failure_penalty: float = 0.0
    stabilization_threshold: float = 70.0
    stabilization_cross_bonus: float = 0.0
    ventilator_allocation_cost: float = 0.0
    unnecessary_ventilator_penalty: float = 0.0
    alive_patient_step_bonus: float = 0.0
    terminal_alive_bonus: float = 0.0
    terminal_survival_rate_bonus_scale: float = 0.0


@dataclass(frozen=True)
class TaskTargetProfile:
    critical_survival_min: float
    critical_survival_max: float | None
    success_rate_min: float
    min_ventilator_utilization: float
    max_ventilator_utilization: float | None
    max_invalid_actions: int


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    num_patients: int
    max_steps: int
    medics_available: int
    ventilators_available: int
    severity_mix: tuple[str, ...]
    deterioration_rate: dict[str, float]
    treatment_gain: dict[str, float]
    initial_health_ranges: dict[str, tuple[float, float]]
    terminal_success_survival_rate: float
    terminal_success_avg_health: float
    reward_weights: RewardWeights


TASK_CONFIGS: dict[str, TaskConfig] = {
    "task1": TaskConfig(
        task_name="task1",
        num_patients=3,
        max_steps=20,
        medics_available=2,
        ventilators_available=1,
        severity_mix=("critical", "severe", "moderate"),
        deterioration_rate={"mild": 1.0, "moderate": 2.0, "severe": 4.0, "critical": 7.0},
        treatment_gain={"mild": 6.0, "moderate": 10.0, "severe": 14.0, "critical": 20.0},
        initial_health_ranges={
            "mild": (78.0, 90.0),
            "moderate": (65.0, 76.0),
            "severe": (50.0, 62.0),
            "critical": (34.0, 42.0),
        },
        terminal_success_survival_rate=0.66,
        terminal_success_avg_health=70.0,
        reward_weights=RewardWeights(
            successful_treat_critical=10.0,
            successful_treat_moderate=2.0,
            successful_treat_mild=1.0,
            successful_ventilator_allocation=5.0,
            ineffective_treatment_penalty=-2.0,
            invalid_action_penalty=-3.0,
            unnecessary_wait_penalty=-3.0,
            death_penalty_critical=-85.0,
            death_penalty_moderate=-12.0,
            death_penalty_mild=-8.0,
            stabilization_bonus=3.0,
            episode_success_bonus=45.0,
            all_critical_survive_bonus=20.0,
            health_gain_scale=0.35,
            episode_failure_penalty=0.0,
            stabilization_threshold=70.0,
            stabilization_cross_bonus=0.0,
            ventilator_allocation_cost=0.0,
            unnecessary_ventilator_penalty=0.0,
            alive_patient_step_bonus=0.0,
            terminal_alive_bonus=0.0,
            terminal_survival_rate_bonus_scale=0.0,
        ),
    ),
    "task2": TaskConfig(
        task_name="task2",
        num_patients=4,
        max_steps=24,
        medics_available=2,
        ventilators_available=1,
        severity_mix=("critical", "severe", "moderate", "moderate"),
        deterioration_rate={"mild": 1.0, "moderate": 3.0, "severe": 6.0, "critical": 10.0},
        treatment_gain={"mild": 6.0, "moderate": 9.0, "severe": 13.0, "critical": 19.0},
        initial_health_ranges={
            "mild": (70.0, 85.0),
            "moderate": (56.0, 70.0),
            "severe": (42.0, 56.0),
            "critical": (30.0, 38.0),
        },
        terminal_success_survival_rate=0.60,
        terminal_success_avg_health=65.0,
        reward_weights=RewardWeights(
            successful_treat_critical=5.0,
            successful_treat_moderate=3.0,
            successful_treat_mild=0.8,
            successful_ventilator_allocation=12.0,
            ineffective_treatment_penalty=-3.0,
            invalid_action_penalty=-4.0,
            unnecessary_wait_penalty=-5.0,
            death_penalty_critical=-52.0,
            death_penalty_moderate=-22.0,
            death_penalty_mild=-9.0,
            stabilization_bonus=4.5,
            episode_success_bonus=90.0,
            all_critical_survive_bonus=8.0,
            health_gain_scale=0.35,
            episode_failure_penalty=-70.0,
            stabilization_threshold=70.0,
            stabilization_cross_bonus=8.0,
            ventilator_allocation_cost=-1.0,
            unnecessary_ventilator_penalty=-2.0,
            alive_patient_step_bonus=1.2,
            terminal_alive_bonus=0.0,
            terminal_survival_rate_bonus_scale=30.0,
        ),
    ),
    "task3": TaskConfig(
        task_name="task3",
        num_patients=5,
        max_steps=28,
        medics_available=4,
        ventilators_available=3,
        severity_mix=("critical", "critical", "severe", "moderate", "moderate"),
        deterioration_rate={"mild": 1.5, "moderate": 4.0, "severe": 8.0, "critical": 11.0},
        treatment_gain={"mild": 5.5, "moderate": 8.0, "severe": 12.0, "critical": 24.0},
        initial_health_ranges={
            "mild": (65.0, 78.0),
            "moderate": (46.0, 60.0),
            "severe": (34.0, 48.0),
            "critical": (22.0, 32.0),
        },
        terminal_success_survival_rate=0.30,
        terminal_success_avg_health=60.0,
        reward_weights=RewardWeights(
            successful_treat_critical=14.0,
            successful_treat_moderate=1.5,
            successful_treat_mild=0.6,
            successful_ventilator_allocation=16.0,
            ineffective_treatment_penalty=-4.0,
            invalid_action_penalty=-5.0,
            unnecessary_wait_penalty=-6.0,
            death_penalty_critical=-120.0,
            death_penalty_moderate=-18.0,
            death_penalty_mild=-10.0,
            stabilization_bonus=5.0,
            episode_success_bonus=120.0,
            all_critical_survive_bonus=20.0,
            health_gain_scale=0.30,
            episode_failure_penalty=-80.0,
            stabilization_threshold=70.0,
            stabilization_cross_bonus=5.0,
            ventilator_allocation_cost=0.0,
            unnecessary_ventilator_penalty=0.0,
            alive_patient_step_bonus=0.0,
            terminal_alive_bonus=12.0,
            terminal_survival_rate_bonus_scale=50.0,
        ),
    ),
}


LEGACY_DIFFICULTY_TO_TASK = {
    "easy": "task1",
    "medium": "task2",
    "hard": "task3",
}


TASK_TARGETS: dict[str, TaskTargetProfile] = {
    "task2": TaskTargetProfile(
        critical_survival_min=0.85,
        critical_survival_max=0.95,
        success_rate_min=0.80,
        min_ventilator_utilization=0.20,
        max_ventilator_utilization=0.60,
        max_invalid_actions=0,
    ),
    "task3": TaskTargetProfile(
        critical_survival_min=0.60,
        critical_survival_max=None,
        success_rate_min=0.40,
        min_ventilator_utilization=0.20,
        max_ventilator_utilization=None,
        max_invalid_actions=0,
    ),
}


TASK_TRAINING_DEFAULTS: dict[str, dict[str, float | int]] = {
    "task1": {
        "episodes": 200,
        "epsilon_start": 0.20,
        "epsilon_end": 0.05,
    },
    "task2": {
        "episodes": 500,
        "epsilon_start": 0.20,
        "epsilon_end": 0.05,
    },
    "task3": {
        "episodes": 2000,
        "epsilon_start": 0.30,
        "epsilon_end": 0.05,
    },
}


def resolve_task_name(task: str | None, difficulty: str | None) -> str:
    if task is not None:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task: {task}")
        return task

    if difficulty is not None:
        mapped = LEGACY_DIFFICULTY_TO_TASK.get(difficulty)
        if mapped is not None:
            return mapped

    return "task2"