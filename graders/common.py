from __future__ import annotations

import json
from typing import Any

from triage_env.graders.common import grade_task


def print_grader_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
