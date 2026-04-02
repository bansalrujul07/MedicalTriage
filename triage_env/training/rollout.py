def run_episode(env, agent):
    obs = env.reset()
    total_reward = 0.0

    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)
        total_reward += obs.reward

    return {
        "total_reward": total_reward,
        "steps": obs.step_count,
        "alive_patients": sum(1 for p in obs.patients if p.alive),
    }