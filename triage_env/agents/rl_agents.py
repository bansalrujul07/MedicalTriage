import json
import os
import random

try:
    from triage_env.agents.base_agent import BaseAgent
    from triage_env.models import TriageAction, TriageObservation
    from triage_env.training.state_encoder import encode_observation
except ImportError:
    from agents.base_agent import BaseAgent
    from models import TriageAction, TriageObservation
    from training.state_encoder import encode_observation


class RLAgent(BaseAgent):
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.checkpoint_metadata = {}
        self.checkpoint_warning = None
        self.unseen_state_count = 0

    def _state_key(self, observation: TriageObservation):
        return encode_observation(observation)

    def _freeze_json_value(self, value):
        if isinstance(value, list):
            return tuple(self._freeze_json_value(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((k, self._freeze_json_value(v)) for k, v in value.items()))
        return value

    def _valid_actions(self, observation: TriageObservation):
        alive = [p for p in observation.patients if p.alive]
        actions = [("wait", -1)]

        for p in alive:
            actions.append(("treat", p.id))
            if observation.resources.ventilators_available > 0 and not p.ventilated:
                actions.append(("allocate_ventilator", p.id))

        return actions

    def _ensure_state_actions(self, state_key, actions):
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        for action in actions:
            if action not in self.q_table[state_key]:
                self.q_table[state_key][action] = 0.0

    def act(self, observation: TriageObservation) -> TriageAction:
        state_key = self._state_key(observation)
        actions = self._valid_actions(observation)
        if state_key not in self.q_table:
            self.unseen_state_count += 1
        self._ensure_state_actions(state_key, actions)

        if random.random() < self.epsilon:
            action_type, patient_id = random.choice(actions)
        else:
            best_value = max(self.q_table[state_key][a] for a in actions)
            best_actions = [a for a in actions if self.q_table[state_key][a] == best_value]
            action_type, patient_id = random.choice(best_actions)

        return TriageAction(action_type=action_type, patient_id=patient_id)

    def update(
        self,
        observation: TriageObservation,
        action: TriageAction,
        reward: float,
        next_observation: TriageObservation,
    ):
        state_key = self._state_key(observation)
        action_key = (action.action_type, action.patient_id)

        current_actions = self._valid_actions(observation)
        self._ensure_state_actions(state_key, current_actions)
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0

        next_state_key = self._state_key(next_observation)
        next_actions = self._valid_actions(next_observation)
        self._ensure_state_actions(next_state_key, next_actions)

        current_q = self.q_table[state_key][action_key]

        if next_observation.done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state_key][a] for a in next_actions)
            target = reward + self.gamma * max_next_q

        self.q_table[state_key][action_key] = current_q + self.alpha * (target - current_q)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str, metadata: dict | None = None):
        serializable = {}
        for state, actions in self.q_table.items():
            state_str = json.dumps(state)
            serializable[state_str] = {
                json.dumps(action): value for action, value in actions.items()
            }

        checkpoint_metadata = metadata or {}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "metadata": checkpoint_metadata,
                    "q_table": serializable,
                },
                f,
                indent=2,
            )

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        self.checkpoint_metadata = data.get("metadata", {})
        self.checkpoint_warning = None

        self.q_table = {}
        for state_str, actions in data["q_table"].items():
            state = self._freeze_json_value(json.loads(state_str))
            self.q_table[state] = {}
            for action_str, value in actions.items():
                action = self._freeze_json_value(json.loads(action_str))
                self.q_table[state][action] = float(value)