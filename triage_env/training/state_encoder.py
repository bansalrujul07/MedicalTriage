"""State encoding utilities for RL agents in Medical Triage environment."""

import numpy as np
from typing import Any, Dict


def encode_observation(obs: Dict[str, Any]) -> np.ndarray:
    """
    Encode observation dictionary into a numerical state vector.
    
    Args:
        obs: Observation dictionary from environment step()
        
    Returns:
        Encoded state as numpy array suitable for RL agent input
    """
    # Extract relevant state features from observation
    # This is a simplified encoder that extracts numerical features
    
    if isinstance(obs, dict):
        # Convert dict values to numerical array
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
        
        # Return encoded state or zero vector if empty
        return np.array(features, dtype=np.float32) if features else np.zeros(1, dtype=np.float32)
    
    # If obs is already an array, return as-is
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    
    # Default: return empty encoded state
    return np.zeros(1, dtype=np.float32)
