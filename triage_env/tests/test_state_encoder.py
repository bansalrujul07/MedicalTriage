from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.training.state_encoder import encode_observation


def test_encode_observation_returns_tuple():
    env = TriageEnvironment()
    obs = env.reset()

    state = encode_observation(obs)

    assert isinstance(state, tuple)
    assert len(state) >= 3
