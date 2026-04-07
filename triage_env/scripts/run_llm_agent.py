import argparse
import logging
import os

from triage_env.agents.llm_agent import LLMAgent
from triage_env.config import get_runtime_config
from triage_env.evaluation.evaluator import run_single_episode
from triage_env.server.triage_env_environment import TriageEnvironment


logging.basicConfig(level=logging.INFO)


def main() -> None:
    runtime = get_runtime_config()
    parser = argparse.ArgumentParser(description="Run LLM triage agent")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default=runtime.default_task)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. LLMAgent will run in safe fallback mode.")

    env = TriageEnvironment(task=args.task)
    metrics = run_single_episode(env, LLMAgent())
    print("LLM agent episode metrics:")
    print(metrics)


if __name__ == "__main__":
    main()