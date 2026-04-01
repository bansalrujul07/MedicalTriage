try:
    from triage_env.server.triage_env_environment import TriageEnvironment
    from triage_env.evaluation.rule_based_agent import RuleBasedAgent
except ImportError:
    from server.triage_env_environment import TriageEnvironment
    from evaluation.rule_based_agent import RuleBasedAgent


def main():
    env = TriageEnvironment(max_steps=20)
    agent = RuleBasedAgent()

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