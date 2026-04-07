from triage_env.models import TriageAction
from triage_env.server.triage_env_environment import TriageEnvironment


def test_wait_penalty_harsher_on_task3_than_task1():
    env_easy = TriageEnvironment(task="task1")
    env_easy.reset(task="task1")
    reward_easy = env_easy.step(TriageAction(action_type="wait", patient_id=-1)).reward

    env_hard = TriageEnvironment(task="task3")
    env_hard.reset(task="task3")
    reward_hard = env_hard.step(TriageAction(action_type="wait", patient_id=-1)).reward

    assert reward_hard < reward_easy
