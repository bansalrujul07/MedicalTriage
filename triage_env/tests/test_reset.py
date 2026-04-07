# from triage_env.server.triage_env_environment import TriageEnvironment

# def test_reset_returns_valid_observation():
#     env = TriageEnvironment()
#     obs = env.reset()

#     assert obs is not None
#     assert hasattr(obs, "patients")
#     assert isinstance(obs.patients, list)
#     assert hasattr(obs, "resources")
#     assert hasattr(obs, "step_count")
#     assert len(obs.patients) > 0


from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


def test_reset_initializes_environment():
    env = TriageEnvironment()
    obs = env.reset()

    assert obs.step_count == 0
    assert obs.done is False
    assert len(obs.patients) == TASK_CONFIGS[env.task_name].num_patients
    assert obs.resources.medics_available == 2
    assert obs.resources.ventilators_available == 1