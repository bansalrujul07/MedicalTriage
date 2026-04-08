#!/usr/bin/env python3
"""
OpenEnv Submission Validator (Python version)

Equivalent to the provided validate-submission.sh:
1) Ping HF Space /reset
2) Run docker build
3) Run openenv validate

Usage:
    python validation.py <ping_url> [repo_dir]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DOCKER_BUILD_TIMEOUT = 600


@dataclass
class Colors:
    red: str = ""
    green: str = ""
    yellow: str = ""
    bold: str = ""
    nc: str = ""


def _colors() -> Colors:
    if sys.stdout.isatty():
        return Colors(
            red="\033[0;31m",
            green="\033[0;32m",
            yellow="\033[1;33m",
            bold="\033[1m",
            nc="\033[0m",
        )
    return Colors()


C = _colors()


def now_utc_hms() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_utc_hms()}] {message}")


def pass_msg(message: str) -> None:
    log(f"{C.green}PASSED{C.nc} -- {message}")


def fail_msg(message: str) -> None:
    log(f"{C.red}FAILED{C.nc} -- {message}")


def hint(message: str) -> None:
    print(f"  {C.yellow}Hint:{C.nc} {message}")


def stop_at(step_name: str) -> None:
    print()
    print(f"{C.red}{C.bold}Validation stopped at {step_name}.{C.nc} Fix the above before continuing.")
    raise SystemExit(1)


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return 124, output


def resolve_repo_dir(repo_dir_raw: str) -> Path:
    repo = Path(repo_dir_raw).expanduser().resolve()
    if not repo.exists() or not repo.is_dir():
        print(f"Error: directory '{repo_dir_raw}' not found")
        raise SystemExit(1)
    return repo


def normalize_ping_url(url: str) -> str:
    return url.rstrip("/")


def check_step1_ping(ping_url: str) -> None:
    log(f"{C.bold}Step 1/3: Pinging HF Space{C.nc} ({ping_url}/reset) ...")

    payload = b"{}"
    req = urllib.request.Request(
        f"{ping_url}/reset",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.getcode()
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception:
        code = 0

    if code == 200:
        pass_msg("HF Space is live and responds to /reset")
        return

    if code == 0:
        fail_msg("HF Space not reachable (connection failed or timed out)")
        hint("Check your network connection and that the Space is running.")
        hint(f"Try: curl -s -o /dev/null -w '%{{http_code}}' -X POST {ping_url}/reset")
        stop_at("Step 1")

    fail_msg(f"HF Space /reset returned HTTP {code} (expected 200)")
    hint("Make sure your Space is running and the URL is correct.")
    hint(f"Try opening {ping_url} in your browser first.")
    stop_at("Step 1")


def find_docker_context(repo_dir: Path) -> tuple[Path, Path] | None:
    root_docker = repo_dir / "Dockerfile"
    server_docker = repo_dir / "server" / "Dockerfile"
    nested_server_docker = repo_dir / "triage_env" / "server" / "Dockerfile"
    if root_docker.exists():
        return repo_dir, root_docker
    if server_docker.exists():
        return repo_dir, server_docker
    if nested_server_docker.exists():
        return repo_dir, nested_server_docker
    return None


def find_openenv_dir(repo_dir: Path) -> Path | None:
    """Find the directory containing openenv.yaml by checking common locations."""
    # Check root first
    if (repo_dir / "openenv.yaml").exists():
        return repo_dir
    
    # Check common subdirectories
    for subdir in ["triage_env", "env", "environment", "server"]:
        candidate = repo_dir / subdir
        if (candidate / "openenv.yaml").exists():
            return candidate
    
    # If not found, return None
    return None


def check_step2_docker_build(repo_dir: Path) -> None:
    log(f"{C.bold}Step 2/3: Running docker build{C.nc} ...")

    if shutil.which("docker") is None:
        fail_msg("docker command not found")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        stop_at("Step 2")

    docker_info = find_docker_context(repo_dir)
    if docker_info is None:
        fail_msg("No Dockerfile found in repo root, server/, or triage_env/server/")
        stop_at("Step 2")

    docker_context, dockerfile_path = docker_info

    log(f"  Found Dockerfile: {dockerfile_path}")
    log(f"  Build context:   {docker_context}")

    rc, output = run_command(
        ["docker", "build", "-f", str(dockerfile_path), str(docker_context)],
        timeout=DOCKER_BUILD_TIMEOUT,
    )

    if rc == 0:
        pass_msg("Docker build succeeded")
        return

    fail_msg(f"Docker build failed (timeout={DOCKER_BUILD_TIMEOUT}s)")
    tail = "\n".join((output or "").splitlines()[-20:])
    if tail:
        print(tail)
    stop_at("Step 2")


def check_step3_openenv_validate(repo_dir: Path) -> None:
    log(f"{C.bold}Step 3/3: Running openenv validate{C.nc} ...")

    if shutil.which("openenv") is None:
        fail_msg("openenv command not found")
        hint("Install it: pip install openenv-core")
        stop_at("Step 3")

    # Find the actual OpenEnv environment directory
    env_dir = find_openenv_dir(repo_dir)
    if env_dir is None:
        fail_msg("openenv.yaml not found in repo or common subdirectories (triage_env, env, environment, server)")
        hint(f"Make sure openenv.yaml is in {repo_dir} or a subdirectory like {repo_dir}/triage_env/")
        stop_at("Step 3")

    log(f"  Found openenv.yaml in: {env_dir}")

    rc, output = run_command(["openenv", "validate"], cwd=env_dir)

    if rc == 0:
        pass_msg("openenv validate passed")
        if output.strip():
            log(f"  {output.strip()}")
        return

    fail_msg("openenv validate failed")
    print(output)
    stop_at("Step 3")


def print_header(repo_dir: Path, ping_url: str) -> None:
    print()
    print(f"{C.bold}========================================{C.nc}")
    print(f"{C.bold}  OpenEnv Submission Validator{C.nc}")
    print(f"{C.bold}========================================{C.nc}")
    log(f"Repo:     {repo_dir}")
    log(f"Ping URL: {ping_url}")
    print()


def print_success_footer() -> None:
    print()
    print(f"{C.bold}========================================{C.nc}")
    print(f"{C.green}{C.bold}  All 3/3 checks passed!{C.nc}")
    print(f"{C.green}{C.bold}  Your submission is ready to submit.{C.nc}")
    print(f"{C.bold}========================================{C.nc}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenEnv submission validator (Python equivalent of validate-submission.sh)"
    )
    parser.add_argument("ping_url", help="Hugging Face Space URL, e.g. https://your-space.hf.space")
    parser.add_argument("repo_dir", nargs="?", default=".", help="Path to repository (default: current dir)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ping_url = normalize_ping_url(args.ping_url)
    repo_dir = resolve_repo_dir(args.repo_dir)
    os.environ["PING_URL"] = ping_url

    print_header(repo_dir, ping_url)

    check_step1_ping(ping_url)
    check_step2_docker_build(repo_dir)
    check_step3_openenv_validate(repo_dir)

    print_success_footer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
