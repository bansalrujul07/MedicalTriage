from triage_env.agents.random_agent import RandomAgent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.training.trainer import train


def main() -> None:
    env = TriageEnvironment(max_steps=20)
    agent = RandomAgent()

    summary = train(env, agent, episodes=10)

    print("\nFinal Summary:")
    print(summary)


if __name__ == "__main__":
    main()