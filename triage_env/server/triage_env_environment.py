from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import Patient, Resources, TriageAction, TriageObservation, TriageState
    from ..tasks import TASK_CONFIGS, TaskConfig, resolve_task_name
except ImportError:
    from models import Patient, Resources, TriageAction, TriageObservation, TriageState
    from tasks import TASK_CONFIGS, TaskConfig, resolve_task_name


class TriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        task: str | None = None,
        max_steps: int | None = None,
        difficulty: str = "medium",
    ):
        self.task_name = resolve_task_name(task=task, difficulty=difficulty)
        self.task_config = TASK_CONFIGS[self.task_name]
        self.max_steps = max_steps if max_steps is not None else self.task_config.max_steps

        self._resource_usage = {"medics_used": 0, "ventilators_used": 0}
        self._invalid_action_count = 0

        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            patients=self._build_initial_patients(),
            resources=Resources(
                medics_available=self.task_config.medics_available,
                ventilators_available=self.task_config.ventilators_available,
            ),
            max_steps=self.max_steps,
            total_reward=0.0,
        )

    def _set_task(self, task: str | None = None, difficulty: str | None = None) -> None:
        self.task_name = resolve_task_name(task=task, difficulty=difficulty)
        self.task_config = TASK_CONFIGS[self.task_name]

    def _initial_health(self, severity: str) -> float:
        low, high = self.task_config.initial_health_ranges[severity]
        return (low + high) / 2.0

    def _build_initial_patients(self) -> list[Patient]:
        patients: list[Patient] = []
        for pid, severity in enumerate(self.task_config.severity_mix):
            patients.append(
                Patient(
                    id=pid,
                    severity=severity,
                    health=self._initial_health(severity),
                )
            )
        return patients

    def reset(self, task: str | None = None, difficulty: str | None = None) -> TriageObservation:
        if task is not None or difficulty is not None:
            self._set_task(task=task, difficulty=difficulty)
            self.max_steps = self.task_config.max_steps

        self._resource_usage = {"medics_used": 0, "ventilators_used": 0}
        self._invalid_action_count = 0

        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            patients=self._build_initial_patients(),
            resources=Resources(
                medics_available=self.task_config.medics_available,
                ventilators_available=self.task_config.ventilators_available,
            ),
            max_steps=self.max_steps,
            total_reward=0.0,
        )

        return TriageObservation(
            patients=self._state.patients,
            resources=self._state.resources,
            step_count=self._state.step_count,
            done=False,
            reward=0.0,
            message="Environment reset successfully",
            metadata=self._build_metadata(reward_breakdown={}),
        )

    def step(self, action: TriageAction) -> TriageObservation:  # type: ignore[override]
        self._state.step_count += 1
        rw = self.task_config.reward_weights

        reward = 0.0
        message = "No action taken"
        reward_breakdown: dict[str, float] = {}
        action_was_valid = False
        invalid_action_taken = False

        pre_alive = [p for p in self._state.patients if p.alive]
        pre_total_health = sum(p.health for p in pre_alive)
        pre_stable_ids = {
            p.id
            for p in self._state.patients
            if p.alive and p.health >= rw.stabilization_threshold
        }

        if action.action_type == "treat" and action.patient_id >= 0:
            patient = self._get_patient(action.patient_id)
            if patient and patient.alive and self._state.resources.medics_available > 0:
                self._state.resources.medics_available -= 1
                self._resource_usage["medics_used"] += 1

                old_health = patient.health
                gain = self.task_config.treatment_gain.get(patient.severity, 8.0)
                if patient.ventilated:
                    gain += 3.0

                patient.health = min(100.0, patient.health + gain)
                patient.waiting_time = 0

                severity_component = {
                    "critical": rw.successful_treat_critical,
                    "severe": rw.successful_treat_moderate,
                    "moderate": rw.successful_treat_moderate,
                    "mild": rw.successful_treat_mild,
                }[patient.severity]
                health_gain_component = (patient.health - old_health) * rw.health_gain_scale
                reward += severity_component + health_gain_component

                reward_breakdown[f"successful_treat_{patient.severity}"] = severity_component
                reward_breakdown["health_gain_component"] = health_gain_component
                message = f"Treated patient {patient.id}"
                action_was_valid = True
            else:
                reward += rw.invalid_action_penalty
                self._invalid_action_count += 1
                invalid_action_taken = True
                reward_breakdown["invalid_action_penalty"] = rw.invalid_action_penalty
                message = "Invalid treatment action"

        elif action.action_type == "allocate_ventilator" and action.patient_id >= 0:
            patient = self._get_patient(action.patient_id)
            if (
                patient
                and patient.alive
                and not patient.ventilated
                and self._state.resources.ventilators_available > 0
            ):
                patient.ventilated = True
                self._state.resources.ventilators_available -= 1
                self._resource_usage["ventilators_used"] += 1

                if patient.severity in ("critical", "severe"):
                    vent_reward = rw.successful_ventilator_allocation
                else:
                    vent_reward = rw.ineffective_treatment_penalty

                # Task-configured operational cost for ventilator usage.
                vent_reward += rw.ventilator_allocation_cost
                if rw.ventilator_allocation_cost != 0.0:
                    reward_breakdown["ventilator_allocation_cost"] = rw.ventilator_allocation_cost

                unnecessary = patient.severity != "critical" or patient.health >= rw.stabilization_threshold
                if unnecessary and rw.unnecessary_ventilator_penalty != 0.0:
                    vent_reward += rw.unnecessary_ventilator_penalty
                    reward_breakdown["unnecessary_ventilator_penalty"] = rw.unnecessary_ventilator_penalty

                reward += vent_reward
                reward_breakdown["successful_ventilator_allocation"] = vent_reward
                message = f"Ventilator assigned to patient {patient.id}"
                action_was_valid = True
            else:
                reward += rw.invalid_action_penalty
                self._invalid_action_count += 1
                invalid_action_taken = True
                reward_breakdown["invalid_action_penalty"] = rw.invalid_action_penalty
                message = "Invalid ventilator allocation"

        elif action.action_type == "wait":
            urgent_exists = any(
                p.alive and (p.severity in ("critical", "severe") or p.health < 40)
                for p in self._state.patients
            )
            wait_penalty = rw.unnecessary_wait_penalty if urgent_exists else (rw.unnecessary_wait_penalty / 3.0)
            reward += wait_penalty
            reward_breakdown["unnecessary_wait_penalty"] = wait_penalty
            message = "Waited one step"
            action_was_valid = True

        else:
            reward += rw.invalid_action_penalty
            self._invalid_action_count += 1
            invalid_action_taken = True
            reward_breakdown["invalid_action_penalty"] = rw.invalid_action_penalty
            message = "Invalid action"

        time_penalty, death_penalties = self._advance_time()
        reward += time_penalty
        reward_breakdown["time_penalty"] = time_penalty
        reward_breakdown.update(death_penalties)

        post_alive = [p for p in self._state.patients if p.alive]
        post_total_health = sum(p.health for p in post_alive)
        health_delta_reward = (post_total_health - pre_total_health) * 0.05
        reward += health_delta_reward
        reward_breakdown["health_delta_reward"] = health_delta_reward

        if rw.alive_patient_step_bonus != 0.0 and action_was_valid:
            alive_step_reward = len(post_alive) * rw.alive_patient_step_bonus
            reward += alive_step_reward
            reward_breakdown["alive_patient_step_bonus"] = alive_step_reward

        if invalid_action_taken and reward >= 0.0:
            correction = rw.invalid_action_penalty * 0.5
            reward += correction
            reward_breakdown["invalid_action_correction_penalty"] = correction

        post_stable_ids = {
            p.id
            for p in self._state.patients
            if p.alive and p.health >= rw.stabilization_threshold
        }
        newly_stabilized = len(post_stable_ids - pre_stable_ids)
        if newly_stabilized > 0 and rw.stabilization_cross_bonus != 0.0:
            stabilization_cross_reward = newly_stabilized * rw.stabilization_cross_bonus
            reward += stabilization_cross_reward
            reward_breakdown["stabilization_cross_bonus"] = stabilization_cross_reward

        stability_bonus = self._compute_global_stability_bonus()
        reward += stability_bonus
        reward_breakdown["stabilization_bonus"] = stability_bonus

        done = self._is_done()
        if done:
            terminal_reward, success_achieved = self._compute_terminal_reward()
            if rw.terminal_survival_rate_bonus_scale != 0.0:
                patient_count = len(self._state.patients)
                alive_count = len([p for p in self._state.patients if p.alive])
                survival_rate = (alive_count / patient_count) if patient_count > 0 else 0.0
                terminal_survival_bonus = survival_rate * rw.terminal_survival_rate_bonus_scale
                terminal_reward += terminal_survival_bonus
                reward_breakdown["terminal_survival_rate_bonus"] = terminal_survival_bonus
            if rw.terminal_alive_bonus != 0.0:
                terminal_alive_reward = len([p for p in self._state.patients if p.alive]) * rw.terminal_alive_bonus
                terminal_reward += terminal_alive_reward
                reward_breakdown["terminal_alive_bonus"] = terminal_alive_reward
            if not success_achieved and rw.episode_failure_penalty != 0.0:
                terminal_reward += rw.episode_failure_penalty
                reward_breakdown["episode_failure_penalty"] = rw.episode_failure_penalty
            reward += terminal_reward
            reward_breakdown["episode_terminal_reward"] = terminal_reward

        self._state.total_reward += reward

        return TriageObservation(
            patients=self._state.patients,
            resources=self._state.resources,
            step_count=self._state.step_count,
            done=done,
            reward=reward,
            message=message,
            metadata=self._build_metadata(reward_breakdown=reward_breakdown),
        )

    @property
    def state(self) -> TriageState:
        return self._state

    def _get_patient(self, patient_id: int):
        for patient in self._state.patients:
            if patient.id == patient_id:
                return patient
        return None

    def _advance_time(self) -> tuple[float, dict[str, float]]:
        rw = self.task_config.reward_weights
        penalty = 0.0
        death_penalties: dict[str, float] = {}

        for patient in self._state.patients:
            if not patient.alive:
                continue

            base_decay = self.task_config.deterioration_rate[patient.severity]
            if patient.ventilated:
                base_decay = max(0.0, base_decay - 4.0)

            if patient.health < 40:
                base_decay += 1.5
            if patient.health < 20:
                base_decay += 2.0

            patient.health = max(0.0, patient.health - base_decay)
            patient.waiting_time += 1

            # Release ventilator when patient is stable enough for weaning.
            if patient.ventilated and patient.health >= self.task_config.reward_weights.stabilization_threshold:
                patient.ventilated = False
                self._state.resources.ventilators_available = min(
                    self.task_config.ventilators_available,
                    self._state.resources.ventilators_available + 1,
                )

            if patient.health == 0.0:
                patient.alive = False
                if patient.ventilated:
                    patient.ventilated = False
                    self._state.resources.ventilators_available = min(
                        self.task_config.ventilators_available,
                        self._state.resources.ventilators_available + 1,
                    )
                severity_penalty = {
                    "critical": rw.death_penalty_critical,
                    "severe": rw.death_penalty_moderate,
                    "moderate": rw.death_penalty_moderate,
                    "mild": rw.death_penalty_mild,
                }[patient.severity]
                penalty += severity_penalty
                death_penalties[f"death_penalty_{patient.severity}"] = (
                    death_penalties.get(f"death_penalty_{patient.severity}", 0.0)
                    + severity_penalty
                )

        # Track time-based ventilator occupancy for smoother utilization metrics.
        ventilator_occupancy = sum(
            1 for p in self._state.patients if p.alive and p.ventilated
        )
        self._resource_usage["ventilator_steps_used"] = (
            self._resource_usage.get("ventilator_steps_used", 0) + ventilator_occupancy
        )

        self._state.resources.medics_available = self.task_config.medics_available
        return penalty, death_penalties

    def _compute_global_stability_bonus(self) -> float:
        alive_patients = [p for p in self._state.patients if p.alive]
        if not alive_patients:
            return -10.0

        avg_health = sum(p.health for p in alive_patients) / len(alive_patients)
        critical_low = sum(1 for p in alive_patients if p.severity == "critical" and p.health < 45)

        bonus = (avg_health / 100.0) * self.task_config.reward_weights.stabilization_bonus
        bonus -= critical_low * 1.5
        return bonus

    def _compute_terminal_reward(self) -> tuple[float, bool]:
        rw = self.task_config.reward_weights
        alive_patients = [p for p in self._state.patients if p.alive]
        critical_patients = [p for p in self._state.patients if p.severity == "critical"]
        surviving_critical = [p for p in critical_patients if p.alive]

        if not alive_patients:
            return rw.death_penalty_critical, False

        survival_rate = len(alive_patients) / len(self._state.patients)
        avg_health = sum(p.health for p in alive_patients) / len(alive_patients)

        terminal_reward = 0.0
        success_achieved = False
        if (
            survival_rate >= self.task_config.terminal_success_survival_rate
            and avg_health >= self.task_config.terminal_success_avg_health
        ):
            terminal_reward += rw.episode_success_bonus
            success_achieved = True

        if critical_patients and len(surviving_critical) == len(critical_patients):
            terminal_reward += rw.all_critical_survive_bonus

        return terminal_reward, success_achieved

    def _is_done(self) -> bool:
        if self._state.step_count >= self._state.max_steps:
            return True

        alive_patients = [p for p in self._state.patients if p.alive]
        if not alive_patients:
            return True

        if all(p.health >= 88.0 for p in alive_patients):
            return True

        return False

    def _build_metadata(self, reward_breakdown: dict) -> dict:
        alive_patients = [p for p in self._state.patients if p.alive]
        critical_patients = [p for p in self._state.patients if p.severity == "critical"]
        surviving_critical = [p for p in critical_patients if p.alive]

        return {
            "episode_id": self._state.episode_id,
            "task": self.task_name,
            "total_reward": self._state.total_reward,
            "alive_patients": len(alive_patients),
            "dead_patients": len(self._state.patients) - len(alive_patients),
            "critical_patients": len(critical_patients),
            "critical_survivors": len(surviving_critical),
            "invalid_action_count": self._invalid_action_count,
            "resource_usage": dict(self._resource_usage),
            "initial_resources": {
                "medics_available": self.task_config.medics_available,
                "ventilators_available": self.task_config.ventilators_available,
            },
            "reward_breakdown": reward_breakdown,
        }