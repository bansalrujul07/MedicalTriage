# from triage_env.server.triage_env_environment import TriageEnvironment
# from triage_env.models import TriageAction


# def test_episode_eventually_finishes():
#     env = TriageEnvironment()
#     obs = env.reset()

#     done = False
#     steps = 0

#     while not done and steps < 100:
#         alive_patients = [p for p in obs.patients if p.alive]

#         if alive_patients:
#             action = TriageAction(action_type="treat", patient_id=alive_patients[0].id)
#         else:
#             action = TriageAction(action_type="wait", patient_id=-1)

#         obs = env.step(action)
#         done = obs.done
#         steps += 1

#     assert done is True

try:
    from server.triage_env_environment import TriageEnvironment
except ImportError:
    from triage_env.server.triage_env_environment import TriageEnvironment


def test_environment_eventually_finishes():
    env = TriageEnvironment(max_steps=5)
    obs = env.reset()

    for _ in range(5):
        if obs.done:
            break
        obs = env.step(type("Action", (), {"action_type": "wait", "patient_id": -1})())

    assert obs.done is True