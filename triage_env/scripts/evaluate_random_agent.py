try:
    from triage_env.server.triage_env_environment import TriageEnvironment
    from triage_env.agents.random_agent import RandomAgent
    from triage_env.evaluation.evaluator import evaluate
except ImportError:
    from server.triage_env_environment import TriageEnvironment
    from agents.random_agent import RandomAgent
    from evaluation.evaluator import evaluate


def main():
    env = TriageEnvironment(max_steps=20)
    agent = RandomAgent()

    metrics = evaluate(env, agent, episodes=10)

    print("\nFinal Evaluation Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()