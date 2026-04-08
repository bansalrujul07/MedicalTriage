# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triage Env Environment."""

from .models import TriageAction, TriageObservation

__all__ = [
    "TriageAction",
    "TriageObservation",
    "TriageEnv",
]


def __getattr__(name: str):
    if name == "TriageEnv":
        try:
            from .client import TriageEnv as _TriageEnv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TriageEnv client requires optional dependency 'openenv-core'. "
                "Install project dependencies to use triage_env.client features."
            ) from exc
        return _TriageEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
