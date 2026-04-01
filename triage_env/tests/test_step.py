# from triage_env.server.triage_env_environment import TriageEnvironment
# from triage_env.models import TriageAction


# def test_step_returns_valid_transition():
#     env = TriageEnvironment()
#     obs = env.reset()

#     alive_patients = [p for p in obs.patients if p.alive]

#     if alive_patients:
#         action = TriageAction(action_type="treat", patient_id=alive_patients[0].id)
#     else:
#         action = TriageAction(action_type="wait", patient_id=-1)

#     next_obs = env.step(action)

#     # Basic checks
#     assert next_obs is not None
#     assert hasattr(next_obs, "patients")
#     assert hasattr(next_obs, "resources")
#     assert hasattr(next_obs, "step_count")

#     # IMPORTANT checks
#     assert isinstance(next_obs.step_count, int)
#     assert next_obs.step_count >= 0

#     # Check done exists
#     assert hasattr(next_obs, "done")

try:
    from server.triage_env_environment import TriageEnvironment
    from models import TriageAction
except ImportError:
    from triage_env.server.triage_env_environment import TriageEnvironment
    from triage_env.models import TriageAction


def test_step_increments_step_count():
    env = TriageEnvironment()
    env.reset()

    obs = env.step(TriageAction(action_type="wait", patient_id=-1))
    assert obs.step_count == 1


def test_treat_action_returns_observation():
    env = TriageEnvironment()
    env.reset()

    obs = env.step(TriageAction(action_type="treat", patient_id=0))
    assert isinstance(obs.reward, float)
    assert hasattr(obs, "patients")