from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from server.triage_env_environment import TriageEnvironment
from models import TriageAction

env = TriageEnvironment()
obs = env.reset()

print("RESET:")
print(obs.model_dump())

done = False

while not done:
    alive = [p for p in obs.patients if p.alive]

    if alive:
        target = min(alive, key=lambda p: p.health)
        action = TriageAction(action_type="treat", patient_id=target.id)
    else:
        action = TriageAction(action_type="wait", patient_id=-1)

    obs = env.step(action)
    print(obs.model_dump())
    done = obs.done

print("FINAL STATE:")
print(env.state.model_dump())