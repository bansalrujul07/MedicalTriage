# Teammate Task Distribution

## Overview

This project is developed collaboratively, with responsibilities divided across environment design, testing, evaluation, and documentation.

## Roles

### Environment Development (Rujul)
- Implements core environment logic (`reset`, `step`, `state`)
- Handles patient dynamics and resource updates
- Designs reward computation logic

### Model Definitions (Rujul)
- Defines action, observation, and state structures using Pydantic models
- Ensures compatibility with OpenEnv specifications

### Testing (Kanishka)
- Writes unit tests for:
  - reset functionality
  - step transitions
  - episode termination
- Ensures environment behaves correctly under different scenarios

### Evaluation (Kanishka)
- Implements baseline evaluation scripts:
  - random policy
  - rule-based policy
- Validates that environment produces meaningful performance differences

### Documentation (Kanishka)
- Updates README with:
  - project description
  - action space
  - observation space
  - reward and termination
- Creates:
  - reward_plan.md
  - evaluation_plan.md
  - teammate_tasks.md

## Goal

The goal of this task distribution is to ensure:
- clean separation of responsibilities
- faster development
- easier debugging and validation