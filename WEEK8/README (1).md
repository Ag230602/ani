# Week 8 – Multimodal (Florida Data)

This folder contains ready-to-run scaffolds for:
- **Track A**: Voice-Interactive LLM (speech → transcription → QA → TTS)
- **Track B**: Conversational Data Visualization (NL → safe plot spec → matplotlib)
- **Track C**: Multimodal Router (speech / viz / QA) with guardrails + traces

## Quick Start
```bash
pip install -r <(python - <<'PY'
import json; j=json.load(open('/mnt/data/week8/env_week8.json')); print('\n'.join(j['pip']))
PY
)
```
Or install individually:
```bash
pip install SpeechRecognition openai-whisper gTTS pandas matplotlib streamlit pyttsx3
```

Run Streamlit demo:
```bash
streamlit run /mnt/data/week8/streamlit_app.py
```

Artifacts:
- Config: `week8_run_config.json`
- Env: `env_week8.json`
- Data (toy): `data/florida_metrics_sample.csv`
- Speech outputs: `audio_outputs/`
- Visual outputs: `visual_outputs/`
- Logs: `logs/` (trace JSONL)

> Replace the QA backend URL in `week8_run_config.json` with your Week 6/7 endpoint.
