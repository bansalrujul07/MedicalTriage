from triage_env.server.triage_env_environment import TriageEnvironment

def test_reset_returns_valid_observation():
    env = TriageEnvironment()
    obs = env.reset()

    assert obs is not None
    assert hasattr(obs, "patients")
    assert isinstance(obs.patients, list)
    assert hasattr(obs, "resources")
    assert hasattr(obs, "step_count")
    assert len(obs.patients) > 0