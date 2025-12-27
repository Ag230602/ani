"""
Real-time location ingestion (event-driven) + continuous forecasting demo.

What this implements (matches your paper text):
- Real-time location inputs are received continuously (WebSocket).
- An event-driven interface (async queue + worker) ingests spatiotemporal events.
- The forecasting module updates immediately and returns an updated prediction,
  keeping outputs spatially consistent with the latest geographic context.

How to run (two terminals):

Terminal 1 (server):
    pip install fastapi uvicorn websockets
    python realtime_location_forecast.py server

Terminal 2 (simulated stream client):
    python realtime_location_forecast.py client

WebSocket endpoint:
    ws://localhost:8000/ws/location
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# -----------------------------
# Data model for incoming events
# -----------------------------
@dataclass
class LocationEvent:
    timestamp: str  # ISO8601 string (UTC recommended)
    lat: float
    lon: float
    source: str = "client"


# -------------------------------------------
# Minimal forecasting module (replace later)
# -------------------------------------------
class SimpleTrackForecaster:
    """
    Naive forecaster:
    - maintains the last N observed points
    - estimates velocity from the last two points (delta lat/lon)
    - projects next K steps linearly
    """
    def __init__(self, history_size: int = 12):
        self.history_size = history_size
        self.history: List[LocationEvent] = []

    def update(self, evt: LocationEvent) -> None:
        self.history.append(evt)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def forecast(self, horizon_steps: int = 6, step_minutes: int = 60) -> Dict[str, Any]:
        if len(self.history) == 0:
            return {"status": "no_data", "last_observation": None, "forecast": []}

        if len(self.history) < 2:
            last = self.history[-1]
            return {"status": "warming_up", "last_observation": asdict(last), "forecast": []}

        p1, p2 = self.history[-2], self.history[-1]
        dlat = p2.lat - p1.lat
        dlon = p2.lon - p1.lon

        # Parse last observation time (fallback to now)
        try:
            base_time = datetime.fromisoformat(p2.timestamp.replace("Z", "+00:00"))
        except Exception:
            base_time = datetime.now(timezone.utc)

        preds = []
        for k in range(1, horizon_steps + 1):
            t = base_time + timedelta(minutes=k * step_minutes)
            preds.append({
                "timestamp": t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "lat": float(p2.lat + k * dlat),
                "lon": float(p2.lon + k * dlon),
                "method": "naive_linear_extrapolation"
            })

        return {
            "status": "ok",
            "last_observation": asdict(p2),
            "forecast": preds
        }


# -------------------------------------------
# Event-driven ingestion: async queue + worker
# -------------------------------------------
app = FastAPI(title="Real-time Location Ingestion + Forecasting")

event_queue: asyncio.Queue[LocationEvent] = asyncio.Queue(maxsize=5000)
forecaster = SimpleTrackForecaster(history_size=12)


async def ingestion_worker() -> None:
    while True:
        evt = await event_queue.get()
        try:
            forecaster.update(evt)
        finally:
            event_queue.task_done()


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(ingestion_worker())


@app.websocket("/ws/location")
async def ws_location(websocket: WebSocket) -> None:
    """
    Client sends JSON: {"timestamp": "...Z", "lat": 18.0, "lon": -45.0, "source": "simulator"}
    Server returns JSON with an updated forecast on every received event.
    """
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()

            evt = LocationEvent(
                timestamp=str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
                lat=float(payload["lat"]),
                lon=float(payload["lon"]),
                source=str(payload.get("source", "client")),
            )

            # Event-driven ingestion (queue) + immediate response
            await event_queue.put(evt)

            response = forecaster.forecast(horizon_steps=6, step_minutes=60)
            await websocket.send_json(response)

    except WebSocketDisconnect:
        return
    except Exception as e:
        # If anything goes wrong, send an error payload (keeps demo friendly)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Debug endpoint to inspect current queue/history/forecast."""
    return {
        "queue_size": int(event_queue.qsize()),
        "history": [asdict(x) for x in forecaster.history],
        "latest_forecast": forecaster.forecast()
    }


# -----------------------------
# Optional simulator client
# -----------------------------
async def run_client(ws_url: str = "ws://localhost:8000/ws/location") -> None:
    import websockets  # pip install websockets

    # Simple track (replace with IBTrACS points if desired)
    track = [
        (18.0, -45.0),
        (18.5, -46.0),
        (19.2, -47.5),
        (20.1, -49.0),
        (21.0, -50.5),
    ]

    async with websockets.connect(ws_url) as ws:
        for lat, lon in track:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "lat": lat,
                "lon": lon,
                "source": "simulator"
            }
            await ws.send(json.dumps(payload))

            resp = await ws.recv()
            resp = json.loads(resp)

            print("Status:", resp.get("status"))
            print("Last observation:", resp.get("last_observation"))
            f0 = (resp.get("forecast") or [None])[0]
            print("Forecast t+1:", f0)
            print("-" * 60)

            await asyncio.sleep(1.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "client"], help="Run as server or simulated client")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "server":
        uvicorn.run("realtime_location_forecast:app", host=args.host, port=args.port, reload=False)
    else:
        asyncio.run(run_client(f"ws://localhost:{args.port}/ws/location"))


if __name__ == "__main__":
    main()
