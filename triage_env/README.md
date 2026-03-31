---
title: Triage Env Environment Server
emoji: 📺
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# CriticalOps Triage Environment

A real-world triage simulation environment combining both medical and military emergency response scenarios.

In this environment, an AI agent must make critical decisions under pressure using limited resources. The agent is responsible for prioritizing patients based on severity, allocating resources like medics and ventilators, and deciding when to act or wait.

The objective is to maximize survival rates and overall health outcomes while efficiently managing constrained resources in high-stakes situations.


## Action Space

The agent can take the following actions at each step:

- `treat` → Provide medical treatment to a selected patient
- `allocate_ventilator` → Assign a ventilator to a critical patient
- `wait` → Take no action and allow time to pass

Each action includes a `patient_id` indicating the target patient (if applicable).

These actions simulate real-world decision-making under constrained medical and operational conditions.


## Observation Space

At each step, the agent receives an observation containing:

- `patients` → A list of current patients in the scenario
- `resources` → Available medical resources such as medics and ventilators
- `step_count` → Current timestep in the episode
- `message` → Optional environment feedback message

Each patient includes information such as:

- `id`
- `severity` (`mild`, `moderate`, `severe`, `critical`)
- `health` (0 to 100)
- `waiting_time`
- `alive`
- `ventilated`

This observation design allows the agent to make decisions based on urgency, patient condition, and limited operational resources.


## Reward Function

The reward is designed to reflect the quality of decisions made by the agent over time.

- Positive reward for improving patient health
- Higher reward for treating severe or critical patients effectively
- Reward for successfully allocating ventilators to critical patients
- Penalty for inaction when patients require urgent care
- Penalty for poor decisions that lead to health deterioration or death
- Small penalty for inefficient use of limited resources

The reward is not binary — it provides continuous feedback throughout the episode to guide better decision-making.


## Termination

An episode ends when one of the following conditions is met:

- The maximum number of steps is reached
- All patients are no longer in a treatable state (e.g., stabilized or deceased)
- No meaningful actions remain for the agent

This ensures that each episode has a clear boundary while reflecting realistic operational constraints.

## Quick Start

The simplest way to use the Triage Env environment is through the `TriageEnv` class:

```python
from triage_env import TriageAction, TriageEnv

try:
    # Create environment from Docker image
    triage_envenv = TriageEnv.from_docker_image("triage_env-env:latest")

    # Reset
    result = triage_envenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = triage_envenv.step(TriageAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    triage_envenv.close()
```

That's it! The `TriageEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t triage_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
The agent selects one of the following actions:
- `treat` → Provide treatment to a selected patient
- `allocate_ventilator` → Assign ventilator to a critical patient
- `wait` → No action

Each action includes a `patient_id`.

---

### Observation
The agent receives:
- List of patients (with severity, health, status)
- Available resources (medics, ventilators)
- Step count
- Optional message

---

### Reward
The reward is shaped based on:
- Improvement in patient health
- Successful treatment of critical cases
- Efficient resource allocation
- Penalties for inaction or harmful decisions
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Triage Env environment server running, you can connect directly:

```python
from triage_env import TriageEnv

# Connect to existing server
triage_envenv = TriageEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = triage_envenv.reset()
result = triage_envenv.step(TriageAction(message="Hello!"))
```

Note: When connecting to an existing server, `triage_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from triage_env import TriageAction, TriageEnv

# Connect with context manager (auto-connects and closes)
with TriageEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(TriageAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TriageEnvironment,  # Pass class, not instance
    TriageAction,
    TriageObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from triage_env import TriageAction, TriageEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with TriageEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(TriageAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/triage_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
triage_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TriageEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── triage_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
