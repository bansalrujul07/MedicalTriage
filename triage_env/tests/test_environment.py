import pytest

from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS
from triage_env.models import TriageAction


@pytest.fixture
def env():
    environment = TriageEnvironment(max_steps=20)
    environment.reset()
    return environment


def test_reset_initial_state(env):
    obs = env.reset()

    assert obs.step_count == 0
    assert obs.done is False
    assert len(obs.patients) == TASK_CONFIGS[env.task_name].num_patients
    assert obs.resources.medics_available == 2
    assert obs.resources.ventilators_available == 1
    assert obs.metadata["total_reward"] == 0.0


def test_treat_critical_patient_improves_health(env):
    critical_before = env._get_patient(0).health

    obs = env.step(TriageAction(action_type="treat", patient_id=0))

    critical_after = env._get_patient(0).health

    assert obs.message == "Treated patient 0"
    assert critical_after > critical_before
    assert obs.step_count == 1


def test_treat_more_urgent_patient_better_than_wait():
    env1 = TriageEnvironment()
    env1.reset()
    treat_obs = env1.step(TriageAction(action_type="treat", patient_id=0))

    env2 = TriageEnvironment()
    env2.reset()
    wait_obs = env2.step(TriageAction(action_type="wait"))

    assert treat_obs.reward > wait_obs.reward


def test_treat_critical_better_than_moderate():
    env1 = TriageEnvironment()
    env1.reset()
    critical_reward = env1.step(
        TriageAction(action_type="treat", patient_id=0)
    ).reward

    env2 = TriageEnvironment()
    env2.reset()
    moderate_reward = env2.step(
        TriageAction(action_type="treat", patient_id=2)
    ).reward

    assert critical_reward > moderate_reward


def test_invalid_treatment_gets_penalty(env):
    obs = env.step(TriageAction(action_type="treat", patient_id=999))

    assert obs.reward < 0
    assert obs.message == "Invalid treatment action"


def test_allocate_ventilator_to_critical_is_better_than_moderate():
    env1 = TriageEnvironment()
    env1.reset()
    critical_obs = env1.step(
        TriageAction(action_type="allocate_ventilator", patient_id=0)
    )

    env2 = TriageEnvironment()
    env2.reset()
    moderate_obs = env2.step(
        TriageAction(action_type="allocate_ventilator", patient_id=2)
    )

    assert critical_obs.reward > moderate_obs.reward


def test_wait_penalty_when_urgent_patient_exists(env):
    obs = env.step(TriageAction(action_type="wait"))

    assert obs.reward < 0
    assert obs.message == "Waited one step"


def test_ignoring_critical_is_bad():
    env = TriageEnvironment()
    env.reset()

    moderate_reward = env.step(
        TriageAction(action_type="treat", patient_id=2)
    ).reward

    wait_env = TriageEnvironment()
    wait_env.reset()
    wait_reward = wait_env.step(TriageAction(action_type="wait")).reward

    assert moderate_reward > wait_reward


def test_medics_reset_after_step(env):
    env.step(TriageAction(action_type="treat", patient_id=0))

    assert env.state.resources.medics_available == 2


def test_patient_can_die_if_ignored():
    env = TriageEnvironment(max_steps=10)
    env.reset()

    for _ in range(10):
        env.step(TriageAction(action_type="wait"))
        if any(not p.alive for p in env.state.patients):
            break

    assert any(not p.alive for p in env.state.patients)


def test_patient_death_penalty():
    env = TriageEnvironment(max_steps=10)
    env.reset()

    total_reward = 0.0
    for _ in range(10):
        obs = env.step(TriageAction(action_type="wait"))
        total_reward += obs.reward

    assert total_reward < -20


def test_environment_runs_multiple_steps_without_crashing():
    env = TriageEnvironment(max_steps=15)
    env.reset()

    obs = None
    for _ in range(10):
        alive = [p for p in env.state.patients if p.alive]
        if not alive:
            break

        target = min(alive, key=lambda p: p.health)
        obs = env.step(TriageAction(action_type="treat", patient_id=target.id))

    assert obs is not None
    assert obs.step_count >= 1
    assert isinstance(obs.reward, float)


def test_reward_breakdown_present(env):
    obs = env.step(TriageAction(action_type="treat", patient_id=0))

    assert "reward_breakdown" in obs.metadata
    assert isinstance(obs.metadata["reward_breakdown"], dict)