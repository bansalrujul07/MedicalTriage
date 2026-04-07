from triage_env.agents.llm_agent import LLMAgent
from triage_env.agents.parser import parse_llm_action
from triage_env.server.triage_env_environment import TriageEnvironment


def test_parse_llm_action_valid_json():
    action = parse_llm_action('{"action_type":"treat","patient_id":1,"reasoning":"ok"}')
    assert action.action_type == "treat"
    assert action.patient_id == 1


def test_parse_llm_action_invalid_payload_falls_back_wait():
    action = parse_llm_action("not-json")
    assert action.action_type == "wait"
    assert action.patient_id == -1


def test_llm_agent_fallback_on_malformed_response():
    env = TriageEnvironment(max_steps=5)
    obs = env.reset()

    def bad_llm(system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt, user_prompt
        return "I cannot output JSON"

    agent = LLMAgent(llm_callable=bad_llm)
    action = agent.act(obs)

    assert action.action_type in {"treat", "allocate_ventilator", "wait"}
    if action.action_type == "wait":
        assert action.patient_id == -1
