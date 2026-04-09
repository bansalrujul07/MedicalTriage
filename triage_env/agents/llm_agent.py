import logging
import os
from typing import Callable

from openai import OpenAI

from triage_env.agents.base_agent import BaseAgent
from triage_env.agents.parser import parse_llm_action
from triage_env.agents.prompt_builder import build_system_prompt, build_user_prompt
from triage_env.config import LLMConfig, get_llm_config
from triage_env.models import TriageAction, TriageObservation

LOGGER = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """
    LLM-backed triage agent.

    Works with:
    - a mock callable passed from run_llm_agent.py
    - or the default internal fallback
    """

    def __init__(
        self,
        llm_callable: Callable[[str, str], str] | None = None,
        config: LLMConfig | None = None,
    ):
        self.config = config or get_llm_config()
        self.llm_callable = llm_callable
        self._client: OpenAI | None = None
        self._missing_key_warned = False
        # In validator context, never silently degrade to non-LLM behavior.
        self._strict_proxy_mode = bool(
            os.getenv("API_KEY", "").strip() or os.getenv("API_BASE_URL", "").strip()
        )

        if self.llm_callable is None and self.config.api_key:
            LOGGER.info("Initializing OpenAI-compatible client for model %s", self.config.model)
            self._client = OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
                base_url=self.config.base_url,
            )
        elif self.llm_callable is None:
            LOGGER.warning("No API key and no custom llm_callable provided; LLMAgent will use fallback policy")

    def act(self, observation: TriageObservation) -> TriageAction:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(observation)
        raw = self._query_llm(system_prompt, user_prompt)

        if raw is None:
            if self._strict_proxy_mode:
                raise RuntimeError("LLM response missing in strict proxy mode")
            return self._safe_fallback_action(observation)

        action = parse_llm_action(raw)
        if not self._is_valid_action(observation, action):
            LOGGER.warning("LLM action failed environment validation; falling back safely")
            return self._safe_fallback_action(observation)

        return action

    def _query_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        if self.llm_callable is not None:
            try:
                return self.llm_callable(system_prompt, user_prompt)
            except Exception as exc:  # pragma: no cover
                if self._strict_proxy_mode:
                    raise
                LOGGER.warning("Custom llm_callable failed: %s", exc)
                return None

        if self._client is None:
            if self._strict_proxy_mode:
                raise RuntimeError("OpenAI client is not initialized in strict proxy mode")
            if not self._missing_key_warned:
                LOGGER.warning("OpenAI API key missing; LLMAgent using fallback policy")
                self._missing_key_warned = True
            return None

        try:
            LOGGER.info("Making LLM API call to %s", self.config.model)
            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            LOGGER.info("OpenAI API call succeeded")
            content = response.choices[0].message.content
            return content or None
        except Exception as exc:  # pragma: no cover
            if self._strict_proxy_mode:
                raise
            LOGGER.warning("OpenAI request failed: %s", exc)
            return None

    def _is_valid_action(self, observation: TriageObservation, action: TriageAction) -> bool:
        alive_patients = {p.id: p for p in observation.patients if p.alive}

        if action.action_type == "wait":
            return action.patient_id == -1

        target = alive_patients.get(action.patient_id)
        if target is None:
            return False

        if action.action_type == "allocate_ventilator":
            if observation.resources.ventilators_available <= 0:
                return False
            if target.ventilated:
                return False

        return True

    def _safe_fallback_action(self, observation: TriageObservation) -> TriageAction:
        alive_patients = [p for p in observation.patients if p.alive]
        if not alive_patients:
            return TriageAction(action_type="wait", patient_id=-1)

        critical_unventilated = [
            p
            for p in alive_patients
            if p.severity in ("critical", "severe") and not p.ventilated
        ]
        if critical_unventilated and observation.resources.ventilators_available > 0:
            target = min(critical_unventilated, key=lambda p: p.health)
            return TriageAction(action_type="allocate_ventilator", patient_id=target.id)

        target = min(alive_patients, key=lambda p: p.health)
        return TriageAction(action_type="treat", patient_id=target.id)





