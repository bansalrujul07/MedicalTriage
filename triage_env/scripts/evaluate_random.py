import random

from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.models import TriageAction


def choose_random_action(obs):
    alive_patients = [p for p in obs.patients if p.alive]

    action_type = random.choice(["treat", "allocate_ventilator", "wait"])

    if action_type == "wait" or not alive_patients:
        return TriageAction(action_type="wait", patient_id=-1)

    patient = random.choice(alive_patients)
    return TriageAction(action_type=action_type, patient_id=patient.id)


def run_episode():
    env = TriageEnvironment()
    obs = env.reset()

    done = False
    while not done:
        action = choose_random_action(obs)
        obs = env.step(action)
        done = obs.done

    return env.state.total_reward


def main():
    random.seed(42)
    scores = [run_episode() for _ in range(20)]
    avg_score = sum(scores) / len(scores)

    print("Random policy scores:", scores)
    print("Average random score:", avg_score)


if __name__ == "__main__":
    main()