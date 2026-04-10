from triage_env.models import TriageAction
from triage_env.server.triage_env_environment import TriageEnvironment


def _assert_step_reward_contract(value: float) -> None:
    assert 0.0 < value < 1.0
    assert value == round(value, 2)


def test_wait_penalty_harsher_on_task3_than_task1():
    env_easy = TriageEnvironment(task="task1")
    env_easy.reset(task="task1")
    reward_easy = env_easy.step(TriageAction(action_type="wait", patient_id=-1)).reward

    env_hard = TriageEnvironment(task="task3")
    env_hard.reset(task="task3")
    reward_hard = env_hard.step(TriageAction(action_type="wait", patient_id=-1)).reward

    _assert_step_reward_contract(reward_easy)
    _assert_step_reward_contract(reward_hard)
    assert reward_hard < reward_easy
