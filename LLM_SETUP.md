# OpenAI LLM Configuration Guide

## Quick Setup (2 steps)

### 1. Get Your API Key
Visit: https://platform.openai.com/api-keys

1. Click "Create new secret key"
2. Copy the key (you won't see it again)
3. Store it somewhere safe

### 2. Update `.env` File

Edit `/home/rujul/Documents/MedicalTriage/.env`:

```bash
OPENAI_API_KEY=sk-proj-your_actual_key_here_1234567890
```

Replace `sk-proj-your_actual_key_here_1234567890` with your real API key.

## Verify Setup

```bash
cd /home/rujul/Documents/MedicalTriage
python -m triage_env.scripts.run_llm_agent --task task1
```

### Expected Output (When API Key Works)

```
INFO: OpenAI API key detected; initializing LLM client for model gpt-4.1-mini
INFO: Making OpenAI API call to gpt-4.1-mini
INFO: OpenAI API call succeeded
EpisodeMetrics(...)
```

### If You See This (API Key Missing or Wrong)

```
WARNING: OPENAI_API_KEY missing; LLMAgent using fallback policy
```

**Fix:** Check your `.env` file again:
- API key starts with `sk-proj-`
- No quotes around the key
- No spaces before/after the key
- File is in the repository root folder

## Environment Variables Reference

| Variable | Default | Example |
|----------|---------|---------|
| OPENAI_API_KEY | (required) | sk-proj-abc123... |
| TRIAGE_LLM_MODEL | gpt-4.1-mini | gpt-4-turbo |
| TRIAGE_LLM_TEMPERATURE | 0.0 | 0.7 |
| TRIAGE_LLM_MAX_TOKENS | 200 | 500 |
| TRIAGE_LLM_TIMEOUT | 20 | 30 |

## Troubleshooting

### Issue: "Invalid API key"
**Fix:** Check that your key is correct and not expired. Generate a new one at https://platform.openai.com/api-keys

### Issue: "Rate limit exceeded"
**Fix:** Your API account has hit usage limits. Check your usage at https://platform.openai.com/account/usage

### Issue: "Model not found"
**Fix:** Change `TRIAGE_LLM_MODEL` in `.env` to a valid model like `gpt-4-turbo` or `gpt-3.5-turbo`

### Issue: ".env file not loading"
**Fix:** Make sure `.env` is in the root repository folder (`/home/rujul/Documents/MedicalTriage/.env`)

## Safety Notes

⚠️ **Never commit `.env` to git** — It contains your API key!
- The `.env` file is already in `.gitignore`
- Never share your API key
- Rotate old keys at https://platform.openai.com/api-keys

## Test All Agents with API

```bash
# Random agent (always works)
python -m triage_env.scripts.run_random --task task2

# Rule-based agent (always works)
python -m triage_env.scripts.run_rule_based --task task2

# LLM agent (requires API key)
python -m triage_env.scripts.run_llm_agent --task task2

# Benchmark all agents across tasks
python -m triage_env.scripts.run_benchmark --tasks task1,task2,task3 --agents RandomAgent,RuleBasedAgent,LLMAgent --episodes 1
```
