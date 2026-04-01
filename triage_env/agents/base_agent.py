from abc import ABC, abstractmethod

try:
    from triage_env.models import TriageAction, TriageObservation
except ImportError:
    from models import TriageAction, TriageObservation


class BaseAgent(ABC):
    @abstractmethod
    def act(self, observation: TriageObservation) -> TriageAction:
        pass