from typing import List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class Patient(Observation):
    id: int
    severity: Literal["mild", "moderate", "severe", "critical"]
    health: float = Field(ge=0.0, le=100.0)
    waiting_time: int = Field(default=0, ge=0)
    alive: bool = True
    ventilated: bool = False


class Resources(Observation):
    medics_available: int = Field(ge=0)
    ventilators_available: int = Field(ge=0)


class TriageAction(Action):
    action_type: Literal["treat", "allocate_ventilator", "wait"]
    patient_id: int = -1


class TriageObservation(Observation):
    patients: List[Patient]
    resources: Resources
    step_count: int
    message: str = ""


class TriageState(State):
    patients: List[Patient] = Field(default_factory=list)
    resources: Resources = Field(
        default_factory=lambda: Resources(
            medics_available=0,
            ventilators_available=0,
        )
    )
    max_steps: int = 20
    total_reward: float = 0.0