from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        Patient,
        Resources,
        TriageAction,
        TriageObservation,
        TriageState,
    )
except ImportError:
    from models import (
        Patient,
        Resources,
        TriageAction,
        TriageObservation,
        TriageState,
    )


class TriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, max_steps: int = 20, difficulty: str = "medium"):
        self.max_steps = max_steps
        self.difficulty = difficulty

        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            patients=[],
            resources=Resources(medics_available=0, ventilators_available=0),
            max_steps=max_steps,
            total_reward=0.0,
        )

    def reset(self) -> TriageObservation:
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            patients=[
                Patient(id=0, severity="critical", health=35.0),
                Patient(id=1, severity="severe", health=55.0),
                Patient(id=2, severity="moderate", health=70.0),
            ],
            resources=Resources(medics_available=2, ventilators_available=1),
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

        reward = 0.0
        message = "No action taken"
        reward_breakdown = {}

        pre_alive_patients = [p for p in self._state.patients if p.alive]
        pre_total_health = sum(p.health for p in pre_alive_patients)

        if action.action_type == "treat" and action.patient_id >= 0:
            patient = self._get_patient(action.patient_id)

            if patient and patient.alive and self._state.resources.medics_available > 0:
                self._state.resources.medics_available -= 1

                old_health = patient.health

                treatment_gain = {
                    "mild": 6.0,
                    "moderate": 10.0,
                    "severe": 14.0,
                    "critical": 20.0,
                }.get(patient.severity, 8.0)

                if patient.ventilated:
                    treatment_gain += 4.0

                patient.health = min(100.0, patient.health + treatment_gain)
                patient.waiting_time = 0
                actual_gain = patient.health - old_health

                treatment_reward = actual_gain * 0.4
                severity_bonus = {
                    "mild": 0.5,
                    "moderate": 1.0,
                    "severe": 2.5,
                    "critical": 4.0,
                }.get(patient.severity, 0.0)

                danger_bonus = 0.0
                if old_health < 40:
                    danger_bonus += 2.0
                if old_health < 25:
                    danger_bonus += 3.0

                urgency_penalty = 0.0
                most_urgent = self._get_most_urgent_patient(exclude_patient_id=patient.id)
                chosen_urgency = self._urgency_score(patient)
                if most_urgent is not None:
                    urgent_score = self._urgency_score(most_urgent)
                    if urgent_score - chosen_urgency >= 80:
                        urgency_penalty = -1.5

                action_reward = treatment_reward + severity_bonus + danger_bonus + urgency_penalty
                reward += action_reward

                reward_breakdown["treatment_reward"] = treatment_reward
                reward_breakdown["severity_bonus"] = severity_bonus
                reward_breakdown["danger_bonus"] = danger_bonus
                reward_breakdown["urgency_penalty"] = urgency_penalty

                message = f"Treated patient {patient.id}"
            else:
                reward -= 3.0
                reward_breakdown["invalid_action_penalty"] = -3.0
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

                ventilator_reward = 0.0
                if patient.severity in ("critical", "severe"):
                    ventilator_reward += 5.0
                else:
                    ventilator_reward -= 1.0

                if patient.health < 35:
                    ventilator_reward += 2.0

                reward += ventilator_reward
                reward_breakdown["ventilator_reward"] = ventilator_reward

                message = f"Ventilator assigned to patient {patient.id}"
            else:
                reward -= 3.0
                reward_breakdown["invalid_action_penalty"] = -3.0
                message = "Invalid ventilator allocation"

        elif action.action_type == "wait":
            urgent_exists = any(
                p.alive and (p.severity in ("critical", "severe") or p.health < 40)
                for p in self._state.patients
            )
            wait_penalty = -4.0 if urgent_exists else -1.0
            reward += wait_penalty
            reward_breakdown["wait_penalty"] = wait_penalty
            message = "Waited one step"

        else:
            reward -= 3.0
            reward_breakdown["invalid_action_penalty"] = -3.0
            message = "Invalid action"

        time_penalty = self._advance_time()
        reward += time_penalty
        reward_breakdown["time_penalty"] = time_penalty

        post_alive_patients = [p for p in self._state.patients if p.alive]
        post_total_health = sum(p.health for p in post_alive_patients)
        health_delta_reward = (post_total_health - pre_total_health) * 0.05
        reward += health_delta_reward
        reward_breakdown["health_delta_reward"] = health_delta_reward

        stability_bonus = self._compute_global_stability_bonus()
        reward += stability_bonus
        reward_breakdown["stability_bonus"] = stability_bonus

        done = self._is_done()

        if done:
            terminal_reward = self._compute_terminal_reward()
            reward += terminal_reward
            reward_breakdown["terminal_reward"] = terminal_reward

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

    def _urgency_score(self, patient: Patient) -> float:
        severity_score = {
            "mild": 1,
            "moderate": 2,
            "severe": 3,
            "critical": 4,
        }[patient.severity]

        return (severity_score * 100) - patient.health + (patient.waiting_time * 5)

    def _get_most_urgent_patient(self, exclude_patient_id: int | None = None):
        alive_patients = [
            p
            for p in self._state.patients
            if p.alive and (exclude_patient_id is None or p.id != exclude_patient_id)
        ]
        if not alive_patients:
            return None
        return max(alive_patients, key=self._urgency_score)

    def _advance_time(self) -> float:
        penalty = 0.0

        for patient in self._state.patients:
            if not patient.alive:
                continue

            base_decay = {
                "mild": 1.0,
                "moderate": 3.0,
                "severe": 6.0,
                "critical": 10.0,
            }[patient.severity]

            if patient.ventilated:
                base_decay = max(0.0, base_decay - 5.0)

            if patient.health < 40:
                base_decay += 2.0
            if patient.health < 20:
                base_decay += 3.0

            patient.health = max(0.0, patient.health - base_decay)
            patient.waiting_time += 1

            if patient.waiting_time > 2:
                if patient.severity == "critical":
                    penalty -= 2.5
                elif patient.severity == "severe":
                    penalty -= 1.5
                elif patient.severity == "moderate":
                    penalty -= 0.75
                else:
                    penalty -= 0.25

            if patient.health == 0.0:
                patient.alive = False
                penalty -= 25.0

        self._state.resources.medics_available = 2
        return penalty

    def _compute_global_stability_bonus(self) -> float:
        alive_patients = [p for p in self._state.patients if p.alive]
        if not alive_patients:
            return -20.0

        avg_health = sum(p.health for p in alive_patients) / len(alive_patients)
        critical_count = sum(
            1 for p in alive_patients if p.severity == "critical" and p.health < 50
        )
        severe_low_count = sum(
            1 for p in alive_patients if p.severity == "severe" and p.health < 50
        )

        bonus = 0.0
        bonus += avg_health * 0.05
        bonus -= critical_count * 2.0
        bonus -= severe_low_count * 1.0

        if all(p.health >= 60 for p in alive_patients):
            bonus += 5.0

        return bonus

    def _compute_terminal_reward(self) -> float:
        alive_patients = [p for p in self._state.patients if p.alive]
        dead_patients = [p for p in self._state.patients if not p.alive]

        if not alive_patients:
            return -50.0

        reward = 0.0
        reward += len(alive_patients) * 10.0
        reward -= len(dead_patients) * 20.0

        avg_health = sum(p.health for p in alive_patients) / len(alive_patients)
        reward += avg_health * 0.2

        if all(p.health >= 85.0 for p in alive_patients):
            reward += 25.0

        return reward

    def _is_done(self) -> bool:
        if self._state.step_count >= self._state.max_steps:
            return True

        alive_patients = [p for p in self._state.patients if p.alive]
        if not alive_patients:
            return True

        if all(p.health >= 85.0 for p in alive_patients):
            return True

        return False

    def _build_metadata(self, reward_breakdown: dict) -> dict:
        alive_patients = [p for p in self._state.patients if p.alive]
        return {
            "episode_id": self._state.episode_id,
            "total_reward": self._state.total_reward,
            "alive_patients": len(alive_patients),
            "dead_patients": len(self._state.patients) - len(alive_patients),
            "critical_patients": sum(
                1 for p in alive_patients if p.severity == "critical"
            ),
            "reward_breakdown": reward_breakdown,
        }