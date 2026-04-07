from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Common interface for all triage agents.
    Every agent should implement act(observation).
    """

    @abstractmethod
    def act(self, observation):
        raise NotImplementedError

    def reset(self):
        """
        Optional hook for agents that maintain episode-level memory.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


ObservationLike = Any