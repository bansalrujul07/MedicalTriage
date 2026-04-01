# import random

# from triage_env.server.triage_env_environment import TriageEnvironment
# from triage_env.models import TriageAction


# def choose_random_action(obs):
#     alive_patients = [p for p in obs.patients if p.alive]

#     action_type = random.choice(["treat", "allocate_ventilator", "wait"])

#     if action_type == "wait" or not alive_patients:
#         return TriageAction(action_type="wait", patient_id=-1)

#     patient = random.choice(alive_patients)
#     return TriageAction(action_type=action_type, patient_id=patient.id)


# def run_episode():
#     env = TriageEnvironment()
#     obs = env.reset()

#     done = False
#     while not done:
#         action = choose_random_action(obs)
#         obs = env.step(action)
#         done = obs.done

#     return env.state.total_reward


# def main():
#     random.seed(42)
#     scores = [run_episode() for _ in range(20)]
#     avg_score = sum(scores) / len(scores)

#     print("Random policy scores:", scores)
#     print("Average random score:", avg_score)


# if __name__ == "__main__":
#     main()


import random

try:
    from server.triage_env_environment import TriageEnvironment
    from models import TriageAction
except ImportError:
    from triage_env.server.triage_env_environment import TriageEnvironment
    from triage_env.models import TriageAction


def random_action(obs):
    patient_ids = [p.id for p in obs.patients if p.alive]
    choices = [TriageAction(action_type="wait", patient_id=-1)]

    for pid in patient_ids:
        choices.append(TriageAction(action_type="treat", patient_id=pid))
        choices.append(TriageAction(action_type="allocate_ventilator", patient_id=pid))

    return random.choice(choices)


def main():
    env = TriageEnvironment(max_steps=20)
    obs = env.reset()

    print("Initial Observation:")
    print(obs.model_dump())

    while not obs.done:
        action = random_action(obs)
        print("\nAction:", action.model_dump())
        obs = env.step(action)
        print("Observation:", obs.model_dump())

    print("\nFinal State:")
    print(env.state.model_dump())


if __name__ == "__main__":
    main()