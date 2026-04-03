from openai import OpenAI
from triage_env.models import TriageAction
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from triage_env.server.triage_env_environment import TriageEnvironment
    from triage_env.agents.llm_agent import LLMAgent
except ImportError:
    from server.triage_env_environment import TriageEnvironment
    from agents.llm_agent import LLMAgent


api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Add it to the project root .env file.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


def real_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert medical triage decision system.\n\n"
                    "Your goal is to maximize overall patient survival, not just treat one patient repeatedly.\n\n"
                    "You must consider ALL patients before deciding:\n"
                    "- Critical patients (very low health) need urgent care\n"
                    "- But ignoring other patients can lead to deaths\n"
                    "- Balance treatment across patients\n"
                    "- Use ventilators when necessary\n\n"
                    "IMPORTANT RULES:\n"
                    "- Do NOT always choose the same patient\n"
                    "- Prevent deaths of other patients\n"
                    "- Think about long-term survival, not just immediate reward\n\n"
                    "Return EXACTLY one JSON object only.\n"
                    "No explanation, no text, only JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    print("\nRaw LLM Output:", repr(content))

    if content is None:
        raise ValueError("LLM returned empty content.")

    return content.strip()


def main():
    env = TriageEnvironment(max_steps=20)
    agent = LLMAgent(llm_callable=real_llm)

    obs = env.reset()
    print("Initial Observation:")
    print(obs.model_dump())

    while not obs.done:
        action = agent.act(obs)

        alive_patients = [p for p in obs.patients if p.alive]

        # Safety guard 1: never choose dead patient
        if action.patient_id != -1:
            valid_ids = {p.id for p in alive_patients}
            if action.patient_id not in valid_ids:
                if alive_patients:
                    target = min(alive_patients, key=lambda p: p.health)
                    action = TriageAction(action_type="treat", patient_id=target.id)
                else:
                    action = TriageAction(action_type="wait", patient_id=-1)

        # Safety guard 2: invalid ventilator choice -> fallback to treat most urgent alive
        if action.action_type == "allocate_ventilator":
            target_patient = next((p for p in alive_patients if p.id == action.patient_id), None)

            invalid_ventilator = (
                obs.resources.ventilators_available <= 0
                or target_patient is None
                or target_patient.ventilated
            )

            if invalid_ventilator:
                if alive_patients:
                    target = min(alive_patients, key=lambda p: p.health)
                    action = TriageAction(action_type="treat", patient_id=target.id)
                else:
                    action = TriageAction(action_type="wait", patient_id=-1)

        print("\nAction:", action.model_dump())
        obs = env.step(action)
        print("Observation:", obs.model_dump())

    print("\nFinal State:")
    print(env.state.model_dump())


if __name__ == "__main__":
    main()