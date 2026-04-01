from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.agents.rule_based_agent import RuleBasedAgent


def run_episode(agent, max_steps=20):
    env = TriageEnvironment(max_steps=max_steps)
    obs = env.reset()

    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)

    return {
        "total_reward": env.state.total_reward,
        "steps": env.state.step_count,
        "alive_patients": sum(1 for p in env.state.patients if p.alive),
    }


def main():
    agent = RuleBasedAgent()
    results = [run_episode(agent) for _ in range(5)]

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_alive = sum(r["alive_patients"] for r in results) / len(results)

    print("Baseline Evaluation")
    print("Episodes:", len(results))
    print("Average Reward:", avg_reward)
    print("Average Alive Patients:", avg_alive)
    print("All Results:", results)


if __name__ == "__main__":
    main()