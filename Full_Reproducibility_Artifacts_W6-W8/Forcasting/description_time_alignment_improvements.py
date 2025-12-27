"""
Description–Time Alignment Improvements (Async Modular Pipeline)

What this implements (matches your paper text):
- Narrative descriptions are aligned with forecast timestamps.
- Each description is explicitly linked to a defined temporal window (valid_from, valid_to).
- Descriptions reference the forecast horizon (e.g., t+1) and the real-time event time.
- This reduces ambiguity and improves interpretability: users can trace each text explanation
  to the exact forecast state that produced it.

Outputs generated (downloadable after running):
- description_time_alignment_updates.png
- description_time_alignment_updates.csv

Run:
    pip install matplotlib
    python description_time_alignment_improvements.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Deque, Optional, List
from collections import deque
import math
import csv

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TimeSeriesPoint:
    ts: str          # ISO8601 timestamp (UTC recommended)
    value: float     # any scalar (e.g., wind speed, pressure, risk score, etc.)
    source: str = "stream"


@dataclass
class Forecast:
    forecast_ts: str  # when the forecast was produced
    horizon: int      # steps ahead (t+h)
    yhat: float       # prediction
    method: str       # model identifier


@dataclass
class AlignedDescription:
    forecast_ts: str
    horizon: int
    valid_from: str
    valid_to: str
    description: str


# -----------------------------
# Shared state (no re-init)
# -----------------------------
class RollingState:
    def __init__(self, window: int = 12):
        self.window = window
        self.raw: Deque[TimeSeriesPoint] = deque(maxlen=window)

        # logs for reproducibility
        self.points_log: List[TimeSeriesPoint] = []
        self.forecast_log: List[Forecast] = []
        self.desc_log: List[AlignedDescription] = []


# -----------------------------
# Module 1: Ingest (async)
# -----------------------------
async def ingest_points(out_q: asyncio.Queue, n: int = 60, delay_s: float = 0.12):
    """
    Simulates newly arriving time-series data points (real-time events).
    Replace this with IBTrACS streaming, sensors, API polling, etc.
    """
    base = 40.0
    for i in range(n):
        # drifting signal with periodic variation (deterministic)
        val = base + 0.18 * i + (math.sin(i / 4) * 0.9)

        pt = TimeSeriesPoint(
            ts=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            value=float(val),
            source="simulator"
        )
        await out_q.put(pt)
        await asyncio.sleep(delay_s)


# -----------------------------
# Module 2: Forecasting (async)
# -----------------------------
async def model_inference(in_q: asyncio.Queue, out_q: asyncio.Queue, state: RollingState):
    """
    Minimal next-step model:
    - estimates slope from last two observations
    - predicts y(t+1) = y(t) + slope
    Replace with LSTM/Transformer/ARIMA without changing the alignment module.
    """
    while True:
        pt: TimeSeriesPoint = await in_q.get()
        try:
            state.raw.append(pt)
            state.points_log.append(pt)

            if len(state.raw) < 2:
                continue

            v1, v2 = state.raw[-2].value, state.raw[-1].value
            slope = v2 - v1
            yhat = v2 + slope

            fc = Forecast(
                forecast_ts=pt.ts,
                horizon=1,
                yhat=float(yhat),
                method="naive_slope_step"
            )
            state.forecast_log.append(fc)
            await out_q.put(fc)

        finally:
            in_q.task_done()


# -----------------------------
# Module 3: Description–Time Alignment (async)
# -----------------------------
async def align_description(in_q: asyncio.Queue, state: RollingState, step_minutes: int = 15):
    """
    Generates a narrative description that is explicitly linked to:
    - the forecast timestamp (forecast_ts)
    - the forecast horizon (t+h)
    - a defined validity window [valid_from, valid_to]
    """
    while True:
        fc: Forecast = await in_q.get()
        try:
            t0 = datetime.fromisoformat(fc.forecast_ts.replace("Z", "+00:00"))
            t1 = t0 + timedelta(minutes=step_minutes)

            # A simple interpretability cue based on direction of change:
            # (compare latest observed to forecast, when possible)
            trend_note = "an expected increase"  # default
            if state.raw:
                last_obs = state.raw[-1].value
                if fc.yhat < last_obs:
                    trend_note = "an expected decrease"
                elif abs(fc.yhat - last_obs) < 1e-9:
                    trend_note = "a stable level"

            desc_text = (
                f"Forecast generated at {t0.strftime('%Y-%m-%d %H:%M:%S')} UTC predicts "
                f"{fc.yhat:.2f} for horizon t+{fc.horizon} (valid from {t0.strftime('%H:%M:%S')} "
                f"to {t1.strftime('%H:%M:%S')} UTC), indicating {trend_note} relative to the "
                f"most recent observed state."
            )

            aligned = AlignedDescription(
                forecast_ts=fc.forecast_ts,
                horizon=fc.horizon,
                valid_from=t0.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                valid_to=t1.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                description=desc_text
            )
            state.desc_log.append(aligned)

            # Publish (replace with DB write, websocket broadcast, UI panel update, etc.)
            print("[ALIGNED DESCRIPTION]", aligned.description)

        finally:
            in_q.task_done()


# -----------------------------
# Visualization + Export
# -----------------------------
def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def save_plot(state: RollingState, png_path: str = "description_time_alignment_updates.png") -> str:
    # Observed series
    x_obs = [_parse_ts(p.ts) for p in state.points_log]
    y_obs = [p.value for p in state.points_log]

    # Forecast series (computed upon each event)
    x_fc = [_parse_ts(f.forecast_ts) for f in state.forecast_log]
    y_fc = [f.yhat for f in state.forecast_log]

    fig = plt.figure()
    plt.plot(x_obs, y_obs, marker="o", linestyle="-", label="Observed (real-time events)")
    plt.plot(x_fc, y_fc, marker="x", linestyle="--", label="Forecast (t+1)")

    plt.xlabel("Time (UTC)")
    plt.ylabel("Signal value")
    plt.title("Description–Time Aligned Forecast Updates")
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path

def save_csv(state: RollingState, csv_path: str = "description_time_alignment_updates.csv") -> str:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["forecast_ts", "horizon", "yhat", "method", "valid_from", "valid_to", "description"])
        for fc, desc in zip(state.forecast_log, state.desc_log):
            w.writerow([
                fc.forecast_ts, fc.horizon, fc.yhat, fc.method,
                desc.valid_from, desc.valid_to, desc.description
            ])
    return csv_path


# -----------------------------
# Orchestrator
# -----------------------------
async def main():
    state = RollingState(window=12)

    q_ingest_to_model = asyncio.Queue(maxsize=2000)
    q_model_to_desc = asyncio.Queue(maxsize=2000)

    tasks = [
        asyncio.create_task(model_inference(q_ingest_to_model, q_model_to_desc, state)),
        asyncio.create_task(align_description(q_model_to_desc, state, step_minutes=15)),
        asyncio.create_task(ingest_points(q_ingest_to_model, n=60, delay_s=0.12)),
    ]

    # Wait for ingestion to finish, then drain queues
    await tasks[-1]
    await q_ingest_to_model.join()
    await q_model_to_desc.join()

    # Cancel background workers
    for t in tasks[:-1]:
        t.cancel()

    # Save artifacts
    png = save_plot(state, "description_time_alignment_updates.png")
    csvp = save_csv(state, "description_time_alignment_updates.csv")
    print(f"\nSaved plot: {png}")
    print(f"Saved CSV:  {csvp}")

    # Paper-ready figure caption
    caption = (
        "Figure X. Description–time alignment for real-time forecasting outputs. "
        "Observed time-series events are ingested continuously and used to compute next-step forecasts. "
        "For each forecast, a narrative description is generated and explicitly bound to the forecast "
        "timestamp and a defined validity window (valid_from–valid_to) for the specified horizon (t+1). "
        "This alignment ensures descriptive outputs remain synchronized with model state, reducing ambiguity "
        "and improving interpretability for real-time decision support."
    )
    print("\nSuggested caption:\n" + caption)

if __name__ == "__main__":
    asyncio.run(main())
