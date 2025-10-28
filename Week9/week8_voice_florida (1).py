"""
week8_voice_florida.py
Track A: Voice → LLM → Voice for Florida Hurricane Irma (2017)

- STT: Whisper (preferred) with fallback to SpeechRecognition
- QA: Local Florida Hurricane Irma context (embedded passages)
- TTS: gTTS (fallback to pyttsx3)
- Logs: JSON trace with per-step latencies; saves TTS audio to /audio_outputs/

Usage:
    python week8_voice_florida.py /path/to/audio.wav [--ref "optional reference text for WER"]
    python week8_voice_florida.py /path/to/audio.wav --config week8_run_config.json
"""

from __future__ import annotations
import os, time, uuid, json, argparse, re, math
from pathlib import Path

# ✅ Use clean local folders (no hard-coded Windows paths!)
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_outputs"
LOG_DIR = BASE_DIR / "logs"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "audio": {
        "stt_engine": "whisper",          # whisper | speechrecognition
        "whisper_model": "base",          # base | small | medium | large
        "tts_engine": "gtts",             # gtts | pyttsx3
        "sample_rate": 16000
    }
}

# -----------------------------
# Florida Hurricane Irma context
# -----------------------------
IRMA_PASSAGES = [
    {"title": "Overview", "text": (
        "Hurricane Irma was a powerful Cape Verde hurricane in 2017. "
        "It peaked at Category 5 with 185 mph sustained winds. "
        "Irma impacted the northern Leeward Islands, the Bahamas, Cuba, and Florida."
    ), "tags": ["irma","overview","2017","atlantic","category 5","winds"]},
    {"title": "Florida Landfalls and Timeline", "text": (
        "Irma made landfall in the Florida Keys near Cudjoe Key on September 10, 2017 as a Category 4 hurricane. "
        "Later the same day, it made a second landfall at Marco Island as a Category 3, "
        "then tracked northward through the state, weakening but causing extensive impacts."
    ), "tags": ["florida","landfall","keys","cudjoe key","marco island","timeline"]},
    {"title": "Impacts in Florida", "text": (
        "Florida experienced destructive winds, storm surge flooding on both coasts, "
        "freshwater flooding from heavy rains, and massive power outages affecting millions."
    ), "tags": ["impacts","storm surge","flooding","power outages","evacuations"]},
    {"title": "Damages and Costs", "text": (
        "Losses from Irma were tens of billions of dollars, heavily concentrated in Florida. "
        "Housing, infrastructure, agriculture, and tourism sectors were affected."
    ), "tags": ["damages","costs","losses","billions","economy"]},
    {"title": "Preparedness and Lessons", "text": (
        "Lessons from Irma include the importance of early evacuation, hardened infrastructure, "
        "and better risk communication about surge, wind, and flooding."
    ), "tags": ["lessons","preparedness","evacuation","infrastructure","communication"]}
]

# -----------------------------
# Utilities
# -----------------------------
def _safe_audio_path(prefix="tts", ext="mp3") -> str:
    return str(AUDIO_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t.lower())).strip()

def wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)]/len(r) if len(r)>0 else 0.0

# -----------------------------
# Speech-to-Text
# -----------------------------
def transcribe_audio(audio_path: str, engine="whisper", whisper_model="base"):
    start = time.time()
    out = {"engine": engine, "text": "", "latency_s": None, "error": None}
    try:
        if engine == "whisper":
            import whisper
            model = whisper.load_model(whisper_model)
            res = model.transcribe(audio_path)
            out["text"] = res.get("text","").strip()
        else:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio_path) as src:
                audio = r.record(src)
            out["text"] = r.recognize_google(audio)
    except Exception as e:
        out["error"] = str(e)
    finally:
        out["latency_s"] = round(time.time() - start, 3)
    return out

# -----------------------------
# Simple QA (Local Irma Context)
# -----------------------------
def retrieve_passages(query, k=3):
    q = set(normalize_text(query).split())
    scored = []
    for p in IRMA_PASSAGES:
        text_tokens = set(normalize_text(p["text"]).split())
        tag_tokens = set(normalize_text(" ".join(p["tags"])).split())
        score = len(q & text_tokens) + 0.5*len(q & tag_tokens)
        scored.append((score, p))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [p for s,p in scored[:k] if s>0]

def answer_query_irma(query):
    start = time.time()
    ctx = retrieve_passages(query)
    if not ctx:
        ans = "Hurricane Irma (2017) severely impacted Florida with wind, surge, flooding, and outages."
    else:
        parts = [f"- {c['title']}: {c['text']}" for c in ctx]
        ans = "Here’s what’s relevant to your question about Hurricane Irma:\n" + "\n".join(parts)
    return {"answer": ans, "latency_s": round(time.time()-start,3), "context_used": [c["title"] for c in ctx]}

# -----------------------------
# Text-to-Speech
# -----------------------------
def synthesize_speech(text, engine="gtts"):
    start = time.time()
    out = {"path": None, "latency_s": None, "error": None}
    try:
        if engine == "gtts":
            from gtts import gTTS
            path = _safe_audio_path("gtts", "mp3")
            gTTS(text).save(path)
            out["path"] = path
        else:
            import pyttsx3
            path = _safe_audio_path("pyttsx3", "wav")
            tts = pyttsx3.init()
            tts.save_to_file(text, path)
            tts.runAndWait()
            out["path"] = path
    except Exception as e:
        out["error"] = str(e)
    finally:
        out["latency_s"] = round(time.time()-start,3)
    return out

# -----------------------------
# Full Pipeline
# -----------------------------
def voice_interaction_irma(audio_path, cfg_path=None, ref_text=None):
    cfg = DEFAULT_CONFIG
    if cfg_path and os.path.exists(cfg_path):
        try:
            loaded = json.load(open(cfg_path))
            if "audio" in loaded:
                cfg["audio"].update(loaded["audio"])
        except Exception:
            pass

    stt = transcribe_audio(audio_path, cfg["audio"]["stt_engine"], cfg["audio"]["whisper_model"])
    qa  = answer_query_irma(stt.get("text",""))
    tts = synthesize_speech(qa["answer"], cfg["audio"]["tts_engine"])

    trace = {
        "audio_in": audio_path,
        "steps": [ {"name":"stt", **stt}, {"name":"qa", **qa}, {"name":"tts", **tts} ],
        "total_latency_s": round(sum(s.get("latency_s",0) for s in [stt,qa,tts]),3)
    }

    with open(LOG_DIR/"trackA_florida_runs.jsonl","a") as f:
        f.write(json.dumps(trace)+"\n")

    return trace

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Voice → LLM → Voice (Florida Hurricane Irma)")
    ap.add_argument("audio", help="Path to your WAV/MP3/M4A audio file")
    ap.add_argument("--config", help="Optional JSON config", default=None)
    ap.add_argument("--ref", help="Optional reference text", default=None)
    args = ap.parse_args()

    res = voice_interaction_irma(args.audio, args.config, args.ref)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
