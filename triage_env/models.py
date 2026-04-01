from typing import Dict, List, Literal
from pydantic import BaseModel, Field


Severity = Literal["mild", "moderate", "severe", "critical"]
ActionType = Literal["treat", "allocate_ventilator", "wait"]


class Patient(BaseModel):
    id: int
    severity: Severity
    health: float
    waiting_time: int = 0
    alive: bool = True
    ventilated: bool = False


class Resources(BaseModel):
    medics_available: int
    ventilators_available: int


class TriageAction(BaseModel):
    action_type: ActionType
    patient_id: int = -1


class TriageObservation(BaseModel):
    done: bool
    reward: float
    metadata: Dict = Field(default_factory=dict)
    patients: List[Patient]
    resources: Resources
    step_count: int
    message: str


class TriageState(BaseModel):
    episode_id: str
    step_count: int
    patients: List[Patient]
    resources: Resources
    max_steps: int
    total_reward: float = 0.0