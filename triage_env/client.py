from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import TriageAction, TriageObservation, TriageState


class TriageEnv(
    EnvClient[TriageAction, TriageObservation, TriageState]
):
    def _step_payload(self, action: TriageAction) -> Dict:
        return {
            "action_type": action.action_type,
            "patient_id": action.patient_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        obs_data = dict(payload.get("observation", {}))

        # OpenEnv wraps done/reward at top level; keep observation model complete.
        if "done" not in obs_data:
            obs_data["done"] = bool(payload.get("done", False))
        if "reward" not in obs_data:
            obs_data["reward"] = float(payload.get("reward") or 0.0)

        observation = TriageObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TriageState:
        return TriageState.model_validate(payload)