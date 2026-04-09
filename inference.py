import asyncio
import json
import os
from typing import List, Optional

from openai import OpenAI

from triage_env.agents.parser import parse_llm_action
from triage_env.client import TriageEnv
from triage_env.models import TriageAction, TriageObservation

# Required by challenge spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE_NAME")
IMAGE_NAME = os.getenv("IMAGE_NAME")

# Environment/task controls
TASKS_ENV = os.getenv("TRIAGE_TASKS", "task1,task2,task3")
SINGLE_TASK = os.getenv("TRIAGE_TASK", os.getenv("MY_ENV_V4_TASK", "")).strip()
BENCHMARK = os.getenv("TRIAGE_BENCHMARK", "medicaltriage")
MAX_STEPS = int(os.getenv("TRIAGE_MAX_STEPS", "28"))
TEMPERATURE = float(os.getenv("TRIAGE_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("TRIAGE_MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("TRIAGE_SUCCESS_THRESHOLD", "0.50"))


SYSTEM_PROMPT = (
    "You are a medical triage policy. Return exactly one JSON object and no extra text. "
    "Schema: {\"action_type\":\"treat\"|\"allocate_ventilator\"|\"wait\",\"patient_id\":int|null}. "
    "Use wait with patient_id=-1 only when no safe/valid resource action exists."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _action_to_str(action: TriageAction) -> str:
    if action.action_type == "wait":
        return "wait()"
    return f"{action.action_type}({action.patient_id})"


def _build_user_prompt(step: int, observation: TriageObservation, history: List[str]) -> str:
    patient_rows = []
    for p in observation.patients:
        patient_rows.append(
            f"id={p.id}, severity={p.severity}, health={p.health:.1f}, "
            f"alive={p.alive}, ventilated={p.ventilated}, waiting_time={p.waiting_time}"
        )

    history_block = "\n".join(history[-6:]) if history else "none"
    return (
        f"Step={step}\n"
        f"Task={observation.metadata.get('task', 'unknown')}\n"
        f"Resources: medics={observation.resources.medics_available}, "
        f"ventilators={observation.resources.ventilators_available}\n"
        f"Patients:\n- " + "\n- ".join(patient_rows) + "\n"
        f"Recent actions:\n{history_block}\n"
        "Return only the JSON action now."
    )


def _select_action(client: OpenAI, step: int, obs: TriageObservation, history: List[str]) -> TriageAction:
    user_prompt = _build_user_prompt(step, obs, history)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        return TriageAction(action_type="wait", patient_id=-1)

    # Reuse repository parser to coerce partial/invalid model payloads safely.
    return parse_llm_action(text)


def _compute_score(last_obs: Optional[TriageObservation], rewards: List[float]) -> float:
    if last_obs is None:
        return 0.0

    alive = [p for p in last_obs.patients if p.alive]
    patient_count = max(1, len(last_obs.patients))
    survival_rate = len(alive) / patient_count
    avg_health_alive = (sum(p.health for p in alive) / len(alive)) if alive else 0.0

    # Score normalized to [0, 1]: blend survival and health quality.
    health_component = min(max(avg_health_alive / 100.0, 0.0), 1.0)
    reward_component = 0.0
    if rewards:
        clipped_rewards = [max(-150.0, min(150.0, r)) for r in rewards]
        reward_component = (sum(clipped_rewards) / (len(clipped_rewards) * 300.0)) + 0.5
        reward_component = min(max(reward_component, 0.0), 1.0)

    score = 0.55 * survival_rate + 0.35 * health_component + 0.10 * reward_component
    return min(max(score, 0.0), 1.0)


def _candidate_image_names() -> List[str]:
    """Return likely local Docker image names in priority order."""
    candidates = [
        LOCAL_IMAGE_NAME,
        DOCKER_IMAGE_NAME,
        IMAGE_NAME,
        "medicaltriage:latest",
        "medicaltriage:ci",
        "medicaltriage",
        "triage-env:latest",
    ]
    return [candidate for candidate in candidates if candidate]


async def _connect_environment() -> tuple[TriageEnv, str]:
    """Connect to the first available local Docker image."""
    last_error: Exception | None = None

    for candidate in _candidate_image_names():
        try:
            env = await TriageEnv.from_docker_image(candidate)
            return env, candidate
        except Exception as exc:  # pragma: no cover - depends on local docker availability
            last_error = exc

    if last_error is not None:
        raise SystemExit(
            "Unable to start a local OpenEnv container. Set LOCAL_IMAGE_NAME or build a supported image tag "
            "such as medicaltriage:latest. Last error: " + str(last_error)
        ) from last_error

    raise SystemExit(
        "Unable to start a local OpenEnv container. Set LOCAL_IMAGE_NAME or build a supported image tag "
        "such as medicaltriage:latest."
    )


def _resolve_tasks() -> List[str]:
    if SINGLE_TASK:
        return [SINGLE_TASK]

    raw = [part.strip() for part in TASKS_ENV.split(",") if part.strip()]
    allowed = {"task1", "task2", "task3"}
    tasks = [task for task in raw if task in allowed]
    return tasks or ["task1", "task2", "task3"]


async def _run_task(env: TriageEnv, client: OpenAI, task_name: str) -> dict:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False
    score = 0.0
    last_obs: Optional[TriageObservation] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    result = await env.reset(task=task_name)
    last_obs = result.observation

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        error_val: Optional[str] = None
        reward_val = 0.0
        done_val = False
        action = TriageAction(action_type="wait", patient_id=-1)

        try:
            action = _select_action(client, step, result.observation, history)
            result = await env.step(action)
            last_obs = result.observation

            reward_val = float(result.reward or 0.0)
            done_val = bool(result.done)
            error_meta = None
            if getattr(result.observation, "metadata", None):
                error_meta = result.observation.metadata.get("last_action_error")
            error_val = error_meta if error_meta else None
        except Exception as exc:
            reward_val = 0.0
            done_val = True
            error_val = str(exc)

        rewards.append(reward_val)
        steps_taken = step
        log_step(
            step=step,
            action=_action_to_str(action),
            reward=reward_val,
            done=done_val,
            error=error_val,
        )
        history.append(
            json.dumps(
                {
                    "step": step,
                    "action": _action_to_str(action),
                    "reward": round(reward_val, 2),
                    "done": done_val,
                }
            )
        )

        if done_val:
            break

    score = _compute_score(last_obs, rewards)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": round(score, 4),
        "steps": steps_taken,
        "success": success,
        "model": MODEL_NAME,
    }


async def main() -> None:
    api_key = OPENAI_API_KEY or HF_TOKEN
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env, image_name = await _connect_environment()
    tasks = _resolve_tasks()
    summary: List[dict] = []

    try:
        for task_name in tasks:
            print(
                f"[INFO] running baseline task={task_name} image={image_name} model={MODEL_NAME}",
                flush=True,
            )
            summary.append(await _run_task(env=env, client=client, task_name=task_name))

    finally:
        try:
            await env.close()
        except Exception:
            # Keep stdout contract strict: do not print non-[START|STEP|END] lines.
            pass

    average_score = sum(item["score"] for item in summary) / max(1, len(summary))
    print(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "tasks": summary,
                "average_score": round(average_score, 4),
                "reproducibility": {
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "api_base_url": API_BASE_URL,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
