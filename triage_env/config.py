import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from triage_env.tasks import TASK_CONFIGS

# Load .env file from repository root
env_file = Path(__file__).resolve().parents[2] / ".env"
if env_file.exists():
    load_dotenv(env_file)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    provider: str  # 'openai' or 'groq'


@dataclass(frozen=True)
class RuntimeConfig:
    train_episodes: int
    eval_episodes: int
    seed: int
    default_task: str
    epsilon_start: float


def get_llm_config() -> LLMConfig:
    provider = os.getenv("TRIAGE_LLM_PROVIDER", "groq").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("TRIAGE_LLM_MODEL", "gpt-4.1-mini")
    else:  # groq
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("TRIAGE_LLM_MODEL", "llama-3.3-70b-versatile")
    
    return LLMConfig(
        api_key=api_key,
        model=model,
        temperature=_env_float("TRIAGE_LLM_TEMPERATURE", 0.0),
        max_tokens=_env_int("TRIAGE_LLM_MAX_TOKENS", 200),
        timeout_seconds=_env_float("TRIAGE_LLM_TIMEOUT", 20.0),
        provider=provider,
    )


def get_runtime_config() -> RuntimeConfig:
    default_task = os.getenv("TRIAGE_DEFAULT_TASK", "task2")
    if default_task not in TASK_CONFIGS:
        default_task = "task2"

    train_episodes = _env_int("TRIAGE_TRAIN_EPISODES", _env_int("TRIAGE_MAX_EPISODES", 200))
    eval_episodes = _env_int("TRIAGE_EVAL_EPISODES", 30)

    return RuntimeConfig(
        train_episodes=max(1, train_episodes),
        eval_episodes=max(1, eval_episodes),
        seed=_env_int("TRIAGE_SEED", 42),
        default_task=default_task,
        epsilon_start=_env_float("TRIAGE_EPSILON_START", 0.2),
    )
