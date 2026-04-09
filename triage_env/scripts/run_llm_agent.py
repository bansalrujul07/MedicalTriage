import argparse
import logging
import os

from triage_env.agents.llm_agent import LLMAgent
from triage_env.config import get_llm_config
from triage_env.config import get_runtime_config
from triage_env.evaluation.evaluator import run_single_episode
from triage_env.server.triage_env_environment import TriageEnvironment


logging.basicConfig(level=logging.INFO)


def main() -> None:
    runtime = get_runtime_config()
    llm_config = get_llm_config()
    parser = argparse.ArgumentParser(description="Run LLM triage agent")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default=runtime.default_task)
    args = parser.parse_args()

    if not llm_config.api_key:
        print("API_KEY is not set. LLMAgent may run in fallback mode.")
    if not llm_config.base_url:
        print("API_BASE_URL is not set. Requests may not route through the validator proxy.")

    env = TriageEnvironment(task=args.task)
    metrics = run_single_episode(env, LLMAgent())
    print("LLM agent episode metrics:")
    print(metrics)


if __name__ == "__main__":
    main()