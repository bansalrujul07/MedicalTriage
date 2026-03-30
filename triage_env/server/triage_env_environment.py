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

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
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
        )

    def step(self, action: TriageAction) -> TriageObservation:  # type: ignore[override]
        self._state.step_count += 1

        reward = 0.0
        message = "No action taken"

        if action.action_type == "treat" and action.patient_id >= 0:
            patient = self._get_patient(action.patient_id)
            if patient and patient.alive and self._state.resources.medics_available > 0:
                patient.health = min(100.0, patient.health + 15.0)
                self._state.resources.medics_available -= 1
                reward += 2.0
                if patient.severity == "critical":
                    reward += 3.0
                message = f"Treated patient {patient.id}"
            else:
                reward -= 2.0
                message = "Invalid treatment action"

        elif action.action_type == "allocate_ventilator" and action.patient_id >= 0:
            patient = self._get_patient(action.patient_id)
            if patient and patient.alive and self._state.resources.ventilators_available > 0:
                patient.ventilated = True
                self._state.resources.ventilators_available -= 1
                reward += 2.0
                message = f"Ventilator assigned to patient {patient.id}"
            else:
                reward -= 2.0
                message = "Invalid ventilator allocation"

        elif action.action_type == "wait":
            reward -= 1.0
            message = "Waited one step"

        self._advance_time()
        self._state.total_reward += reward
        done = self._is_done()

        return TriageObservation(
            patients=self._state.patients,
            resources=self._state.resources,
            step_count=self._state.step_count,
            done=done,
            reward=reward,
            message=message,
            metadata={
                "episode_id": self._state.episode_id,
                "total_reward": self._state.total_reward,
            },
        )

    @property
    def state(self) -> TriageState:
        return self._state

    def _get_patient(self, patient_id: int):
        for patient in self._state.patients:
            if patient.id == patient_id:
                return patient
        return None

    def _advance_time(self):
        for patient in self._state.patients:
            if not patient.alive:
                continue

        decay = {
            "mild": 1.0,
            "moderate": 3.0,
            "severe": 6.0,
            "critical": 10.0,
        }[patient.severity]

        if patient.ventilated:
            decay = max(0.0, decay - 4.0)

        new_health = max(0.0, patient.health - decay)
        patient.health = new_health
        patient.waiting_time += 1

        if patient.health == 0.0:
            patient.alive = False

        self._state.resources.medics_available = min(
            2, self._state.resources.medics_available + 1
        )

    def _is_done(self) -> bool:
        if self._state.step_count >= self._state.max_steps:
            return True

        alive_patients = [p for p in self._state.patients if p.alive]
        if not alive_patients:
            return True

        if all(p.health >= 85.0 for p in alive_patients):
            return True

        return False