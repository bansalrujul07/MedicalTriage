from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.agents.llm_agent import LLMAgent


def mock_llm(system_prompt: str, user_prompt: str) -> str:
    # temporary placeholder until real API integration
    _ = system_prompt, user_prompt
    return '{"action_type": "treat", "patient_id": 0}'


def main():
    env = TriageEnvironment(max_steps=20)
    agent = LLMAgent(llm_callable=mock_llm)

    obs = env.reset()
    print("Initial Observation:")
    print(obs.model_dump())

    while not obs.done:
        action = agent.act(obs)
        print("\nAction:", action.model_dump())
        obs = env.step(action)
        print("Observation:", obs.model_dump())

    print("\nFinal State:")
    print(env.state.model_dump())


if __name__ == "__main__":
    main()