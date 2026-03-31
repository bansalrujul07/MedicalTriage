from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.models import TriageAction


SEVERITY_PRIORITY = {
    "critical": 0,
    "severe": 1,
    "moderate": 2,
    "mild": 3,
}


def choose_rule_based_action(obs):
    alive_patients = [p for p in obs.patients if p.alive]

    if not alive_patients:
        return TriageAction(action_type="wait", patient_id=-1)

    alive_patients.sort(
        key=lambda p: (
            SEVERITY_PRIORITY[p.severity],
            p.health,
            -p.waiting_time,
        )
    )

    target = alive_patients[0]

    if (
        target.severity == "critical"
        and not target.ventilated
        and obs.resources.ventilators_available > 0
    ):
        return TriageAction(
            action_type="allocate_ventilator",
            patient_id=target.id,
        )

    return TriageAction(action_type="treat", patient_id=target.id)


def run_episode():
    env = TriageEnvironment()
    obs = env.reset()

    done = False
    while not done:
        action = choose_rule_based_action(obs)
        obs = env.step(action)
        done = obs.done

    return env.state.total_reward


def main():
    scores = [run_episode() for _ in range(20)]
    avg_score = sum(scores) / len(scores)

    print("Rule-based policy scores:", scores)
    print("Average rule-based score:", avg_score)


if __name__ == "__main__":
    main()