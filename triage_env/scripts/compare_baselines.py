try:
    from triage_env.server.triage_env_environment import TriageEnvironment
    from triage_env.agents.random_agent import RandomAgent
    from triage_env.agents.rule_based_agent import RuleBasedAgent
    from triage_env.evaluation.evaluator import evaluate
except ImportError:
    from server.triage_env_environment import TriageEnvironment
    from agents.random_agent import RandomAgent
    from agents.rule_based_agent import RuleBasedAgent
    from evaluation.evaluator import evaluate


def main():
    print("Evaluating Random Agent...")
    random_env = TriageEnvironment(max_steps=20)
    random_agent = RandomAgent()
    random_metrics = evaluate(random_env, random_agent, episodes=10)

    print("\nEvaluating Rule-Based Agent...")
    rule_env = TriageEnvironment(max_steps=20)
    rule_agent = RuleBasedAgent()
    rule_metrics = evaluate(rule_env, rule_agent, episodes=10)

    print("\n===== Baseline Comparison =====")
    print("Random Agent:", random_metrics)
    print("Rule-Based Agent:", rule_metrics)


if __name__ == "__main__":
    main()