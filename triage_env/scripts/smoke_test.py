from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from triage_env.models import TriageAction
from triage_env.server.triage_env_environment import TriageEnvironment

__test__ = False


def main() -> None:
    env = TriageEnvironment(task="task2")
    obs = env.reset()

    print("RESET:")
    print(obs.model_dump())

    while not obs.done:
        alive = [p for p in obs.patients if p.alive]

        if alive:
            target = min(alive, key=lambda p: p.health)
            action = TriageAction(action_type="treat", patient_id=target.id)
        else:
            action = TriageAction(action_type="wait", patient_id=-1)

        obs = env.step(action)
        print(obs.model_dump())

    print("FINAL STATE:")
    print(env.state.model_dump())


if __name__ == "__main__":
    main()