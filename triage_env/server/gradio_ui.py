from __future__ import annotations

import threading
import traceback
from typing import Any

import gradio as gr

try:
    from ..models import TriageAction
    from .triage_env_environment import TriageEnvironment
except ModuleNotFoundError:
    from models import TriageAction
    from server.triage_env_environment import TriageEnvironment


_ENV_LOCK = threading.Lock()
_ENV: TriageEnvironment | None = None


def _ensure_env(task: str | None = None) -> TriageEnvironment:
    global _ENV
    if _ENV is None:
        initial_task = task or "task2"
        _ENV = TriageEnvironment(task=initial_task)
        _ENV.reset(task=initial_task)
    return _ENV


def _obs_payload(env: TriageEnvironment, observation: Any) -> dict[str, Any]:
    return {
        "observation": observation.model_dump(mode="json"),
        "state": env.state.model_dump(mode="json"),
    }


def _state_payload(env: TriageEnvironment) -> dict[str, Any]:
    return {
        "task": env.task_name,
        "step_count": env.state.step_count,
        "state": env.state.model_dump(mode="json"),
    }


def build_gradio_ui() -> gr.Blocks:
    with gr.Blocks(title="Medical Triage") as demo:
        gr.Markdown("## Medical Triage\nSimple controls for reset, step, and getState.")

        with gr.Row():
            task = gr.Dropdown(
                choices=["task1", "task2", "task3"],
                value="task2",
                label="Task",
            )
            action_type = gr.Dropdown(
                choices=["treat", "allocate_ventilator", "wait"],
                value="wait",
                label="Action Type",
            )
            patient_id = gr.Number(value=-1, precision=0, label="Patient ID")

        with gr.Row():
            reset_btn = gr.Button("Reset", variant="primary")
            step_btn = gr.Button("Step")
            get_state_btn = gr.Button("Get State")

        output = gr.JSON(label="Response")

        def on_reset(selected_task: str):
            try:
                with _ENV_LOCK:
                    env = _ensure_env(selected_task)
                    obs = env.reset(task=selected_task)
                    return _obs_payload(env, obs)
            except Exception as exc:
                return {"error": str(exc), "traceback": traceback.format_exc()}

        def on_step(
            selected_task: str,
            selected_action: str,
            selected_patient_id: float,
        ):
            try:
                with _ENV_LOCK:
                    env = _ensure_env(selected_task)
                    if env.task_name != selected_task:
                        env.reset(task=selected_task)

                    pid = -1 if selected_action == "wait" else int(selected_patient_id or 0)
                    action = TriageAction(action_type=selected_action, patient_id=pid)
                    obs = env.step(action)
                    return _obs_payload(env, obs)
            except Exception as exc:
                return {"error": str(exc), "traceback": traceback.format_exc()}

        def on_get_state():
            try:
                with _ENV_LOCK:
                    env = _ensure_env("task2")
                    return _state_payload(env)
            except Exception as exc:
                return {"error": str(exc), "traceback": traceback.format_exc()}

        reset_btn.click(on_reset, inputs=[task], outputs=[output], queue=False)
        step_btn.click(
            on_step,
            inputs=[task, action_type, patient_id],
            outputs=[output],
            queue=False,
        )
        get_state_btn.click(on_get_state, inputs=[], outputs=[output], queue=False)

        demo.load(on_get_state, inputs=[], outputs=[output], queue=False)

    return demo
