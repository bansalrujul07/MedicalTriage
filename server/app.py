"""Server entrypoint at repo root - forwards to triage_env.server.app"""
from triage_env.server.app import app


def main() -> None:
    """Main entrypoint for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
