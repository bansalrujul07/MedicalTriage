import asyncio
import os
from typing import List, Optional

from openai import OpenAI

from triage_env.agents.parser import parse_llm_action
from triage_env.client import TriageEnv
from triage_env.models import TriageAction, TriageObservation

# Required by challenge spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "").strip()

# Environment/task controls
TASK_NAME = os.getenv("TRIAGE_TASK", os.getenv("MY_ENV_V4_TASK", "task3")).strip()
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


async def _connect_environment() -> tuple[TriageEnv, str]:
    if not LOCAL_IMAGE_NAME:
        raise SystemExit("LOCAL_IMAGE_NAME is required")
    env = await TriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return env, LOCAL_IMAGE_NAME


async def _run_task(env: TriageEnv, client: OpenAI, task_name: str) -> tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False
    score = 0.0
    last_obs: Optional[TriageObservation] = None

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
        history.append(f"step={step} action={_action_to_str(action)} reward={reward_val:.2f} done={str(done_val).lower()}")

        if done_val:
            break

    score = _compute_score(last_obs, rewards)
    success = score >= SUCCESS_SCORE_THRESHOLD

    return success, steps_taken, score, rewards


async def main() -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    env: TriageEnv | None = None
    success = False
    steps_taken = 0
    score = 0.0
    rewards: List[float] = []

    try:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required")

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        env, _ = await _connect_environment()
        success, steps_taken, score, rewards = await _run_task(env=env, client=client, task_name=TASK_NAME)
    except Exception:
        success = False
        score = max(0.0, min(1.0, score))

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        # Always emit END exactly once, including on exception paths.
        log_end(success=success, steps=steps_taken, score=max(0.0, min(1.0, score)), rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
