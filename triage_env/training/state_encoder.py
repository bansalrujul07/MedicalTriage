"""State encoding utilities for RL agents in Medical Triage environment."""

import numpy as np
from typing import Any, Dict


def encode_observation(obs: Any) -> tuple:
    """
    Encode observation dictionary into a numerical state vector.
    
    Args:
        obs: Observation dictionary from environment step()
        
    Returns:
        Encoded, hashable tuple suitable for Q-table indexing
    """
    # Native triage observation object path (preferred).
    if hasattr(obs, "patients") and hasattr(obs, "resources"):
        patients = []
        for patient in getattr(obs, "patients", []):
            patients.append(
                (
                    int(getattr(patient, "id", -1)),
                    str(getattr(patient, "severity", "unknown")),
                    int(round(float(getattr(patient, "health", 0.0)))),
                    bool(getattr(patient, "alive", False)),
                    bool(getattr(patient, "ventilated", False)),
                )
            )

        resources = getattr(obs, "resources", None)
        medics_available = int(getattr(resources, "medics_available", 0)) if resources is not None else 0
        ventilators_available = int(getattr(resources, "ventilators_available", 0)) if resources is not None else 0
        step_count = int(getattr(obs, "step_count", 0))

        return (
            tuple(sorted(patients, key=lambda p: p[0])),
            medics_available,
            ventilators_available,
            step_count,
        )

    if isinstance(obs, dict):
        features = []
        
        # Extract triage score if present
        if "triage_score" in obs:
            features.append(float(obs["triage_score"]))
        
        # Extract vital signs if present
        if "vital_signs" in obs:
            vital = obs["vital_signs"]
            if isinstance(vital, dict):
                features.extend([
                    float(vital.get("heart_rate", 0)),
                    float(vital.get("blood_pressure_systolic", 0)),
                    float(vital.get("blood_pressure_diastolic", 0)),
                    float(vital.get("temperature", 0)),
                    float(vital.get("respiratory_rate", 0))
                ])
        
        # Extract symptoms if present
        if "symptoms" in obs:
            symptoms = obs["symptoms"]
            if isinstance(symptoms, list):
                features.append(float(len(symptoms)))
            elif isinstance(symptoms, dict):
                features.append(float(len(symptoms)))
        
        # Extract patient info if present
        if "patient_age" in obs:
            features.append(float(obs["patient_age"]))
        
        if not features:
            features = [0.0]
        return tuple(float(x) for x in features)

    if isinstance(obs, np.ndarray):
        return tuple(float(x) for x in obs.astype(np.float32).flatten().tolist())

    # Default fallback keeps contract hashable and deterministic.
    return (0.0,)
