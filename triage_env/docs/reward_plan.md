# Reward Plan

## Goal

The reward function should encourage the agent to make effective triage decisions under limited resources in mixed medical and military emergency scenarios.

## Reward Design Principles

The reward should provide meaningful signal throughout the episode, not just at the end. It should reward actions that improve patient outcomes and penalize wasteful or harmful decisions.

## Positive Reward Signals

- Improve a patient's health after treatment
- Prioritize severe and critical patients correctly
- Allocate ventilators to critical patients when available
- Keep more patients alive over time
- Use limited resources efficiently

## Negative Reward Signals

- Waiting when urgent intervention is needed
- Selecting poor targets while higher-priority patients deteriorate
- Wasting ventilators or medical attention
- Allowing preventable patient deaths
- Repeated ineffective actions

## Partial Progress

The reward should reflect gradual progress across the episode. For example:

- small positive reward for modest health improvement
- larger positive reward for stabilizing critical patients
- penalty when a patient's condition worsens
- large penalty when a patient dies

## Why This Matters

This reward structure makes the environment more useful for training and evaluating agents because it captures the quality of decisions over time rather than only final success or failure.