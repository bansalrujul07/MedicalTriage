def compute_metrics(results):
    return {
        "episodes": len(results),
        "avg_reward": sum(r["total_reward"] for r in results) / len(results),
        "avg_alive_patients": sum(r["alive_patients"] for r in results) / len(results),
        "avg_steps": sum(r["steps"] for r in results) / len(results),
    }