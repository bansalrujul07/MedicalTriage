#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

QUICK=0
WITH_LLM=0
SKIP_TASK1=0
SKIP_TASK2=0
SKIP_TASK3=0
SKIP_BENCHMARK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --with-llm)
      WITH_LLM=1
      shift
      ;;
    --skip-task1)
      SKIP_TASK1=1
      shift
      ;;
    --skip-task2)
      SKIP_TASK2=1
      shift
      ;;
    --skip-task3)
      SKIP_TASK3=1
      shift
      ;;
    --skip-benchmark)
      SKIP_BENCHMARK=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--quick] [--with-llm] [--skip-task1] [--skip-task2] [--skip-task3] [--skip-benchmark]"
      exit 2
      ;;
  esac
done

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv/bin/python not found. Create venv first."
  exit 1
fi

PY=".venv/bin/python"

if [[ "$QUICK" -eq 1 ]]; then
  TASK1_EPISODES=150
  TASK1_EVAL_EPISODES=40
  TASK1_SEEDS=(11 22 33)
  TASK2_TRAIN_EPISODES=200
  TASK2_EVAL_EPISODES=15
  TASK3_TRAIN_EPISODES=300
  TASK3_EVAL_EPISODES=10
  BENCH_EPISODES=10
else
  TASK1_EPISODES=500
  TASK1_EVAL_EPISODES=100
  TASK1_SEEDS=(11 22 33 44 55)
  TASK2_TRAIN_EPISODES=500
  TASK2_EVAL_EPISODES=30
  TASK3_TRAIN_EPISODES=1000
  TASK3_EVAL_EPISODES=30
  BENCH_EPISODES=30
fi

TASK1_SEEDS_CSV="$(IFS=,; echo "${TASK1_SEEDS[*]}")"

echo "=== Robustness Pipeline Start ==="
date

echo
echo "[1/4] Running full tests"
"$PY" -m pytest -q

if [[ "$SKIP_TASK1" -eq 0 ]]; then
  echo
  echo "[2/4] Task 1 stability lock"
  "$PY" - <<PY
import random
import sys

from triage_env.agents.rl_agents import RLAgent
from triage_env.evaluation.evaluator import evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS
from triage_env.training.rollout import run_episode

TASK = "task1"
CFG = TASK_CONFIGS[TASK]
EPOCHS = ${TASK1_EPISODES}
EVAL_EPISODES = ${TASK1_EVAL_EPISODES}
SEEDS = [${TASK1_SEEDS_CSV}]

rows = []
for seed in SEEDS:
    random.seed(seed)
    agent = RLAgent()
    env = TriageEnvironment(task=TASK, max_steps=CFG.max_steps)
    for _ in range(EPOCHS):
        run_episode(env, agent, training=True, task=TASK)
    agent.epsilon = 0.0
    summary, _ = evaluate_agent(
        env_class=TriageEnvironment,
        agent=agent,
        task=TASK,
        num_episodes=EVAL_EPISODES,
        seed=seed,
        max_steps=CFG.max_steps,
    )
    rows.append((seed, summary))

print("seed | reward | critical_survival | success | invalid")
for seed, s in rows:
    print(
        f"{seed:>4} | {s['avg_total_reward']:.3f} | "
        f"{s['critical_survival_rate']:.3f} | {s['success_rate']:.3f} | {s['invalid_action_count']:.3f}"
    )

ok = all(s["critical_survival_rate"] >= 1.0 and s["success_rate"] >= 1.0 and s["invalid_action_count"] == 0 and s["avg_total_reward"] > 210 for _, s in rows)
if not ok:
    print("TASK1_GATE=FAIL")
    sys.exit(1)
print("TASK1_GATE=PASS")
PY
fi

if [[ "$SKIP_TASK2" -eq 0 ]]; then
  echo
  echo "[3/4] Task 2 progression"
  "$PY" -m triage_env.scripts.run_task2_progression \
    --train \
    --train-episodes "$TASK2_TRAIN_EPISODES" \
    --episodes "$TASK2_EVAL_EPISODES" \
    --output task2_progression_report.csv

  "$PY" - <<'PY'
import csv
import sys

with open("task2_progression_report.csv", newline="", encoding="utf-8") as f:
    rows = {r["agent_name"]: r for r in csv.DictReader(f)}

if "RLAgent" not in rows or "RuleBasedAgent" not in rows:
    print("TASK2_GATE=FAIL: missing RLAgent or RuleBasedAgent row")
    sys.exit(1)

rl = rows["RLAgent"]
rb = rows["RuleBasedAgent"]

crit = float(rl["critical_survival_rate"])
success = float(rl["success_rate"])
vent = float(rl["ventilator_utilization"])
invalid = float(rl["invalid_action_count"])
reward = float(rl["avg_total_reward"])
rb_reward = float(rb["avg_total_reward"])

print("RL task2 metrics:", {"reward": reward, "critical": crit, "success": success, "vent": vent, "invalid": invalid, "rule_based_reward": rb_reward})

ok = (
    0.85 <= crit <= 0.95
    and success >= 0.80
    and 0.20 <= vent <= 0.60
    and invalid == 0.0
    and reward > rb_reward
)

if not ok:
    print("TASK2_GATE=FAIL")
    sys.exit(1)
print("TASK2_GATE=PASS")
PY
fi

if [[ "$SKIP_TASK3" -eq 0 ]]; then
  echo
  echo "[4/5] Task 3 progression"
  "$PY" -m triage_env.scripts.run_task3_progression \
    --train \
    --train-episodes "$TASK3_TRAIN_EPISODES" \
    --episodes "$TASK3_EVAL_EPISODES" \
    --output task3_progression_report.csv

  TASK3_GATE_MODE="quick"
  if [[ "$QUICK" -eq 0 ]]; then
    TASK3_GATE_MODE="full"
  fi

  TASK3_GATE_MODE="$TASK3_GATE_MODE" "$PY" - <<'PY'
import csv
import os
import sys

with open("task3_progression_report.csv", newline="", encoding="utf-8") as f:
    rows = {r["agent_name"]: r for r in csv.DictReader(f)}

if "RLAgent" not in rows or "RuleBasedAgent" not in rows:
    print("TASK3_GATE=FAIL: missing RLAgent or RuleBasedAgent row")
    sys.exit(1)

rl = rows["RLAgent"]
rb = rows["RuleBasedAgent"]

success = float(rl["success_rate"])
crit = float(rl["critical_survival_rate"])
invalid = float(rl["invalid_action_count"])
reward = float(rl["avg_total_reward"])
rb_reward = float(rb["avg_total_reward"])
vent = float(rl["ventilator_utilization"])

mode = os.environ.get("TASK3_GATE_MODE", "full")
if mode == "quick":
    ok = success > 0.0 and invalid == 0.0 and reward > rb_reward
    gate = "TASK3_GATE_QUICK"
else:
  ok = success >= 0.40 and crit >= 0.60 and invalid == 0.0 and reward > rb_reward and vent >= 0.20
    gate = "TASK3_GATE_FULL"

print("RL task3 metrics:", {"reward": reward, "critical": crit, "success": success, "vent": vent, "invalid": invalid, "rule_based_reward": rb_reward})

if not ok:
    print(f"{gate}=FAIL")
    sys.exit(1)
print(f"{gate}=PASS")
PY
fi

if [[ "$SKIP_BENCHMARK" -eq 0 ]]; then
  echo
  echo "[5/5] Cross-task benchmark"
  AGENTS="RandomAgent,RuleBasedAgent,RLAgent,TrainedQAgent"
  if [[ "$WITH_LLM" -eq 1 ]]; then
    AGENTS="RandomAgent,RuleBasedAgent,LLMAgent,RLAgent,TrainedQAgent"
  fi

  "$PY" -m triage_env.scripts.run_benchmark \
    --tasks task1,task2,task3 \
    --agents "$AGENTS" \
    --episodes "$BENCH_EPISODES" \
    --output benchmark_final.csv

  "$PY" - <<'PY'
import csv
import sys

with open("benchmark_final.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

lookup = {(r["task"], r["agent_name"]): r for r in rows}

needed = [("task3", "RandomAgent"), ("task3", "RLAgent")]
missing = [k for k in needed if k not in lookup]
if missing:
    print("BENCH_GATE=FAIL: missing rows", missing)
    sys.exit(1)

r3 = float(lookup[("task3", "RLAgent")]["avg_total_reward"])
rr = float(lookup[("task3", "RandomAgent")]["avg_total_reward"])
print({"task3_rl_reward": r3, "task3_random_reward": rr})

if r3 <= rr:
    print("BENCH_GATE=FAIL: RLAgent should outperform RandomAgent on task3 reward")
    sys.exit(1)

print("BENCH_GATE=PASS")
PY
fi

echo
echo "=== Robustness Pipeline Completed Successfully ==="
