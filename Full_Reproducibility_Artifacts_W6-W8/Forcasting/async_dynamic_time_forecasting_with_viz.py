"""
Architecture Refinement for Dynamic Time-Based Forecasting (Async Modular Pipeline)

Modules (each runs asynchronously):
1) Ingest -> adds new time-series points
2) TemporalFeatureExtractor -> updates rolling features without full recompute
3) ModelInference -> produces next-step forecast from current features/state
4) ForecastUpdater -> publishes/updates the latest forecast snapshot

Key idea: Pipeline never reinitializes. Each module consumes new data/events and updates state.

Outputs generated (downloadable):
- async_forecast_updates_over_time.png
- async_forecast_updates_over_time.csv

Run:
    pip install matplotlib
    python async_dynamic_time_forecasting_with_viz.py
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Deque, Dict, Any, Optional, List
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
    ts: str          # ISO8601 string
    value: float     # any scalar (wind speed, pressure, etc.)
    source: str = "stream"


@dataclass
class FeatureVector:
    ts: str
    value: float
    mean_w: float
    std_w: float
    slope_w: float   # simple trend estimate


@dataclass
class Forecast:
    ts: str
    horizon: int
    yhat: float
    method: str


# -----------------------------
# Shared state (no re-init)
# -----------------------------
class RollingState:
    def __init__(self, window: int = 12):
        self.window = window
        self.raw: Deque[TimeSeriesPoint] = deque(maxlen=window)
        self.latest_features: Optional[FeatureVector] = None
        self.latest_forecast: Optional[Forecast] = None

        # Logs for visualization and reproducibility
        self.points_log: List[TimeSeriesPoint] = []
        self.features_log: List[FeatureVector] = []
        self.forecast_log: List[Forecast] = []


# -----------------------------
# Module 1: Temporal Feature Extraction (async)
# -----------------------------
class TemporalFeatureExtractor:
    def __init__(self, state: RollingState):
        self.state = state

    def _rolling_mean_std(self, xs):
        n = len(xs)
        if n == 0:
            return 0.0, 0.0
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        return mean, math.sqrt(var)

    def _slope(self, xs):
        # simple slope across the window (last - first) / (n-1)
        n = len(xs)
        if n < 2:
            return 0.0
        return (xs[-1] - xs[0]) / (n - 1)

    async def run(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        while True:
            pt: TimeSeriesPoint = await in_q.get()
            try:
                # Incremental update (no recompute of entire history beyond window)
                self.state.raw.append(pt)
                self.state.points_log.append(pt)

                xs = [p.value for p in self.state.raw]
                mean_w, std_w = self._rolling_mean_std(xs)
                slope_w = self._slope(xs)

                fv = FeatureVector(
                    ts=pt.ts,
                    value=pt.value,
                    mean_w=mean_w,
                    std_w=std_w,
                    slope_w=slope_w
                )
                self.state.latest_features = fv
                self.state.features_log.append(fv)

                # Emit to next stage (inference)
                await out_q.put(fv)

            finally:
                in_q.task_done()


# -----------------------------
# Module 2: Model Inference (async)
# -----------------------------
class ModelInference:
    """
    Placeholder model:
    Forecast next value using current value + slope.
    Replace with LSTM/Transformer/ARIMA/etc. without changing pipeline wiring.
    """
    def __init__(self, state: RollingState):
        self.state = state

    async def run(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        while True:
            fv: FeatureVector = await in_q.get()
            try:
                # lightweight inference (async-friendly)
                yhat = fv.value + fv.slope_w  # naive trend step

                fc = Forecast(
                    ts=fv.ts,
                    horizon=1,
                    yhat=float(yhat),
                    method="naive_trend_step"
                )
                await out_q.put(fc)

            finally:
                in_q.task_done()


# -----------------------------
# Module 3: Forecast Update/Publish (async)
# -----------------------------
class ForecastUpdater:
    def __init__(self, state: RollingState):
        self.state = state

    async def run(self, in_q: asyncio.Queue):
        while True:
            fc: Forecast = await in_q.get()
            try:
                # Update shared state (no reinitialization)
                self.state.latest_forecast = fc
                self.state.forecast_log.append(fc)

                # "Publish" action (replace with DB write, websocket broadcast, etc.)
                print("[FORECAST UPDATED]", asdict(fc))

            finally:
                in_q.task_done()


# -----------------------------
# Simulated Ingestion (async)
# -----------------------------
async def stream_points(out_q: asyncio.Queue, n: int = 60, delay_s: float = 0.15):
    """
    Simulates newly arriving time-series data points.
    Replace with real IBTrACS stream, sensor feed, or API ingestion.
    """
    base = 50.0
    for i in range(n):
        # drifting signal with periodic variation (deterministic, no randomness)
        val = base + 0.15 * i + (math.sin(i / 5) * 0.8)

        pt = TimeSeriesPoint(
            ts=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            value=float(val),
            source="simulator"
        )
        await out_q.put(pt)
        await asyncio.sleep(delay_s)


# -----------------------------
# Visualization + CSV export
# -----------------------------
def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def save_plot(state: RollingState, png_path: str) -> str:
    # Observations
    x_obs = [_parse_ts(p.ts) for p in state.points_log]
    y_obs = [p.value for p in state.points_log]

    # Forecasts (aligned to the same timestamps as observations: forecast is computed upon each new point)
    x_fc = [_parse_ts(f.ts) for f in state.forecast_log]
    y_fc = [f.yhat for f in state.forecast_log]

    fig = plt.figure()
    plt.plot(x_obs, y_obs, marker="o", linestyle="-", label="Observed stream")
    plt.plot(x_fc, y_fc, marker="x", linestyle="--", label="Next-step forecast (t+1)")

    plt.xlabel("Time (UTC)")
    plt.ylabel("Signal value")
    plt.title("Asynchronous Forecast Updates Over Time")

    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path

def save_csv(state: RollingState, csv_path: str) -> str:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "timestamp_utc", "value", "source_or_method"])
        for p in state.points_log:
            w.writerow(["observed", p.ts, p.value, p.source])
        for fc in state.forecast_log:
            w.writerow([f"forecast_h{fc.horizon}", fc.ts, fc.yhat, fc.method])
    return csv_path


# -----------------------------
# Orchestrator: wires modules together
# -----------------------------
async def main():
    state = RollingState(window=12)

    q_ingest_to_feat = asyncio.Queue(maxsize=2000)
    q_feat_to_inf = asyncio.Queue(maxsize=2000)
    q_inf_to_update = asyncio.Queue(maxsize=2000)

    feat = TemporalFeatureExtractor(state)
    inf = ModelInference(state)
    upd = ForecastUpdater(state)

    tasks = [
        asyncio.create_task(feat.run(q_ingest_to_feat, q_feat_to_inf)),
        asyncio.create_task(inf.run(q_feat_to_inf, q_inf_to_update)),
        asyncio.create_task(upd.run(q_inf_to_update)),
        asyncio.create_task(stream_points(q_ingest_to_feat, n=60, delay_s=0.15)),
    ]

    # Wait only for the stream to finish, then let queues drain
    await tasks[-1]
    await q_ingest_to_feat.join()
    await q_feat_to_inf.join()
    await q_inf_to_update.join()

    # Cancel workers gracefully
    for t in tasks[:-1]:
        t.cancel()

    # Export artifacts
    png_path = "async_forecast_updates_over_time.png"
    csv_path = "async_forecast_updates_over_time.csv"

    save_plot(state, png_path)
    save_csv(state, csv_path)

    print(f"\nSaved plot: {png_path}")
    print(f"Saved CSV log: {csv_path}")

    print("\nFinal latest features:", asdict(state.latest_features) if state.latest_features else None)
    print("Final latest forecast:", asdict(state.latest_forecast) if state.latest_forecast else None)


if __name__ == "__main__":
    asyncio.run(main())
