from triage_env.tasks import TASK_CONFIGS


def test_task_configs_exist_and_ordered_by_pressure():
    assert set(TASK_CONFIGS.keys()) == {"task1", "task2", "task3"}

    assert TASK_CONFIGS["task1"].num_patients < TASK_CONFIGS["task2"].num_patients
    assert TASK_CONFIGS["task2"].num_patients < TASK_CONFIGS["task3"].num_patients

    assert TASK_CONFIGS["task1"].deterioration_rate["critical"] < TASK_CONFIGS["task2"].deterioration_rate["critical"]
    assert TASK_CONFIGS["task2"].deterioration_rate["critical"] < TASK_CONFIGS["task3"].deterioration_rate["critical"]
