from triage_env.scripts import run_benchmark, run_llm_agent, run_random, run_rule_based, train_q_agent, train_rl


def test_script_modules_import():
    assert run_random is not None
    assert run_rule_based is not None
    assert run_llm_agent is not None
    assert train_q_agent is not None
    assert train_rl is not None
    assert run_benchmark is not None
