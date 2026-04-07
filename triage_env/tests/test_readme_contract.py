from pathlib import Path


def test_readme_mentions_action_contract():
    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "action_type" in readme
    assert "patient_id" in readme
    assert "echoed_message" not in readme
