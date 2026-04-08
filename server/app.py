"""Server entrypoint at repo root - forwards to triage_env.server.app"""
from triage_env.server.app import app, main as triage_main


def main() -> None:
    """Main entrypoint for running the server."""
    triage_main()


if __name__ == "__main__":
    main()
