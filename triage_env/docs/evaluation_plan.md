# Evaluation Plan

## Goal

The environment will be evaluated using simple baseline policies before testing stronger agent policies. The purpose is to measure whether the environment produces meaningful differences in performance across strategies.

## Baselines

### 1. Random Policy
A random baseline will select from the available actions without strategic prioritization. This provides a lower-bound reference score.

### 2. Rule-Based Policy
A rule-based baseline will prioritize:
- critical patients first
- lower health patients first
- longer waiting patients when priorities are otherwise similar
- ventilator allocation for critical unventilated patients when resources are available

This provides a stronger non-learning baseline.

## Metrics

The following metrics should be tracked:

- total reward per episode
- average reward across multiple episodes
- number of surviving patients
- number of deceased patients
- episode length

## Reproducibility

To ensure reproducibility:
- fixed random seeds should be used for random-policy evaluation
- the same number of episodes should be run for each baseline
- evaluation scripts should print raw scores and averages

## Expected Outcome

The rule-based baseline should outperform the random baseline if the environment reward and transition logic are meaningful. This helps validate that the environment is useful for evaluating agent behavior.