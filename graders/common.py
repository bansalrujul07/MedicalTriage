from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make triage_env imports robust regardless of caller working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triage_env.graders.common import grade_task


def print_grader_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
