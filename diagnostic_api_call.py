import os
import sys
import traceback

print("=" * 80, file=sys.stderr)
print("🔬 FULL API CALL DIAGNOSTIC", file=sys.stderr)
print("=" * 80, file=sys.stderr)

# 1. Check env vars
print(f"1️⃣  API_BASE_URL: {os.getenv('API_BASE_URL')}", file=sys.stderr)
print(f"2️⃣  API_KEY: {'SET' if os.getenv('API_KEY') else 'NOT SET'}", file=sys.stderr)

# 2. Import and create agent
try:
    from triage_env.agents.llm_agent import LLMAgent
    print("3️⃣  ✅ Imported LLMAgent", file=sys.stderr)
    
    agent = LLMAgent()
    print(f"4️⃣  ✅ Agent created: {type(agent).__name__}", file=sys.stderr)
    print(f"5️⃣  ✅ Client base_url: {agent._client.base_url if agent._client else 'NO CLIENT'}", file=sys.stderr)
    
except Exception as e:
    print(f"❌ STEP 2 FAILED: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# 3. Simulate observation and call act()
try:
    from triage_env.models import TriageObservation, Patient, Resources
    
    obs = TriageObservation(
        done=False,
        reward=0.0,
        patients=[
            Patient(id=0, severity="critical", health=20, ventilated=False, alive=True),
            Patient(id=1, severity="severe", health=50, ventilated=False, alive=True),
        ],
        resources=Resources(medics_available=2, ventilators_available=1),
        step_count=1,
        message="Test observation"
    )
    print("6️⃣  🎯 Calling agent.act()...", file=sys.stderr)
    
    action = agent.act(obs)
    
    print(f"7️⃣  ✅ act() returned: {action}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("🎉 FULL TEST PASSED - API call succeeded!", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
except Exception as e:
    print(f"❌ STEP 3 FAILED: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("💥 TEST FAILED - See error above", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    sys.exit(1)
