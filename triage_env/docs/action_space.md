# Action Space

The environment currently supports three actions:

- treat(patient_id)
- allocate_ventilator(patient_id)
- wait

## Intent
These actions simulate a simple medical triage decision process:
- treat improves patient health
- allocate_ventilator reduces future health decay for severe cases
- wait applies no direct intervention and usually incurs penalty