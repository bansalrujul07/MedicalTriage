try:
    from triage_env.agents.base_agent import BaseAgent
    from triage_env.agents.prompt_builder import observation_to_prompt
    from triage_env.agents.action_parser import parse_llm_action
    from triage_env.models import TriageAction, TriageObservation
except ImportError:
    from agents.base_agent import BaseAgent
    from agents.prompt_builder import observation_to_prompt
    from agents.action_parser import parse_llm_action
    from models import TriageAction, TriageObservation


class LLMAgent(BaseAgent):
    def __init__(self, llm_callable):
        self.llm_callable = llm_callable

    def act(self, observation: TriageObservation) -> TriageAction:
        prompt = observation_to_prompt(observation)
        raw_output = self.llm_callable(prompt)
        return parse_llm_action(raw_output)