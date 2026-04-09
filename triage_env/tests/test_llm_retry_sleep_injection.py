from __future__ import annotations

from openai import APITimeoutError

from triage_env.agents.llm_agent import LLMAgent
from triage_env.config import LLMConfig
from triage_env.models import Resources, TriageObservation


class _TimeoutCompletions:
    def create(self, **_kwargs):
        raise APITimeoutError("simulated timeout")


class _TimeoutChat:
    def __init__(self):
        self.completions = _TimeoutCompletions()


class _TimeoutClient:
    def __init__(self):
        self.chat = _TimeoutChat()


def _empty_observation() -> TriageObservation:
    return TriageObservation(
        patients=[],
        resources=Resources(medics_available=0, ventilators_available=0),
        step_count=0,
        done=False,
        reward=0.0,
        message="test",
        metadata={},
    )


def test_llm_agent_retry_uses_injected_sleep_function():
    sleep_calls: list[float] = []

    agent = LLMAgent(
        config=LLMConfig(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1",
            temperature=0.0,
            max_tokens=32,
            timeout_seconds=1.0,
        ),
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )

    # Replace network client with deterministic timeout client.
    agent._client = _TimeoutClient()  # type: ignore[assignment]
    agent._max_attempts = 3
    agent._retry_delay_seconds = 0.01
    agent._strict_proxy_mode = False

    action = agent.act(_empty_observation())

    assert action.action_type == "wait"
    assert action.patient_id == -1
    assert len(sleep_calls) == 2
    assert all(seconds > 0 for seconds in sleep_calls)
