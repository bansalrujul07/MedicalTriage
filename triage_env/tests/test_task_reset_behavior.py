from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


def test_reset_uses_task_config_shape():
    for task in ("task1", "task2", "task3"):
        env = TriageEnvironment(task=task)
        obs = env.reset(task=task)
        cfg = TASK_CONFIGS[task]

        assert obs.metadata["task"] == task
        assert len(obs.patients) == cfg.num_patients
        assert obs.resources.medics_available == cfg.medics_available
        assert obs.resources.ventilators_available == cfg.ventilators_available
