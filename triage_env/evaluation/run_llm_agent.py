import logging
import sys

from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.agents.llm_agent import LLMAgent

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def mock_llm(system_prompt: str, user_prompt: str) -> str:
    # temporary placeholder until real API integration
    _ = system_prompt, user_prompt
    return '{"action_type": "treat", "patient_id": 0}'


def main():
    _configure_logging()
    env = TriageEnvironment(max_steps=20)
    agent = LLMAgent(llm_callable=mock_llm)

    obs = env.reset()
    LOGGER.info("Initial Observation: %s", obs.model_dump())

    while not obs.done:
        action = agent.act(obs)
        LOGGER.info("Action: %s", action.model_dump())
        obs = env.step(action)
        LOGGER.info("Observation: %s", obs.model_dump())

    LOGGER.info("Final State: %s", env.state.model_dump())


if __name__ == "__main__":
    main()