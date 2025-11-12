#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track A – Real-Time Video Loop (Multi‑Backend VLM)
--------------------------------------------------
Backends (select with --backend):
  - qwen2-vl         -> Qwen/Qwen2-VL-7B-Instruct
  - llava-onevision  -> llava-hf/llava-onevision-qwen2-7b
  - internvl2        -> OpenGVLab/InternVL2-8B
  - video-llava      -> LanguageBind/Video-LLaVA-7B

CPU-first, GPU-optional:
  * Auto-detects CUDA/MPS; otherwise runs on CPU (float32).
  * If a backend fails to load, falls back to a light stub so the app still runs.

Overlays & Metrics:
  * Flood/water-like regions (HSV + texture)
  * Debris/motion via optical flow + MOG2
  * Intensity deltas and histogram distance
  * Per-stage latency, FPS, Real-Time Factor (RTF)
  * CSV logging, optional MP4 recording

Usage examples:
  python realtime_video_loop_multi_backend.py --source 0 --backend qwen2-vl --batch 4 --stride 5 --out out_qwen2.mp4
  python realtime_video_loop_multi_backend.py --source path/to/file.mp4 --backend video-llava --batch 8 --stride 4 --vlm 1
"""
import argparse
import time
import os
import json
from collections import deque

import cv2
import numpy as np
import pandas as pd

# Torch/Transformers (optional for VLM; app still runs without them)
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    HF_OK = True
except Exception as _e:
    HF_OK = False
    torch = None
    AutoProcessor = AutoModelForCausalLM = None

# -----------------------------
# Device helper
# -----------------------------
def pick_device():
    if HF_OK:
        try:
            if torch.cuda.is_available():
                return "cuda"
            # Mac MPS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"

# -----------------------------
# VLM wrapper with multi-backend support
# -----------------------------
class VLMModel:
    """
    Wraps multiple HF backends behind a common .infer(frames_bgr) API.
    """
    MODEL_MAP = {
        "qwen2-vl":        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-onevision": "llava-hf/llava-onevision-qwen2-7b",
        "internvl2":       "OpenGVLab/InternVL2-8B",
        "video-llava":     "LanguageBind/Video-LLaVA-7B",
    }

    def __init__(self, enable=True, backend="qwen2-vl", max_new_tokens=64):
        self.enable = enable and HF_OK
        self.backend = backend
        self.model_name = self.MODEL_MAP.get(backend, self.MODEL_MAP["qwen2-vl"])
        self.max_new_tokens = max_new_tokens

        self.device = pick_device()
        self.model = None
        self.processor = None

        if not self.enable:
            print("[INFO] VLM disabled or Transformers not installed; using stub mode.")
            return

        dtype = None
        if self.device == "cuda":
            dtype = torch.float16
        elif self.device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32  # CPU

        try:
            print(f"[INFO] Loading backend '{self.backend}' -> {self.model_name} on {self.device} ...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device in ("cuda", "mps") else None,
            )
            if self.device == "cpu":
                self.model.to("cpu")
            self.model.eval()
            print("[INFO] Loaded model successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load {self.model_name} ({e}). Falling back to stub mode.")
            self.enable = False
            self.model = None
            self.processor = None

    def _build_prompt(self):
        if self.backend in ("video-llava",):
            return "Describe scene changes and signs of flooding/debris across these frames succinctly."
        elif self.backend in ("qwen2-vl","llava-onevision","internvl2"):
            return "In one or two sentences, summarize visible flooding, water rise, and debris motion across frames."
        return "Briefly summarize changes across frames."

    def infer(self, frames_bgr):
        t0 = time.perf_counter()
        if not self.enable or self.model is None or self.processor is None:
            # Stub reasoning: average brightness proxy
            avg_brightness = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames_bgr]))
            latency = time.perf_counter() - t0
            return {
                "vlm_enabled": False,
                "summary": f"Stub: avg brightness={avg_brightness:.1f}.",
                "avg_brightness": avg_brightness,
                "latency_s": latency
            }

        # Prepare inputs
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        prompt = self._build_prompt()

        try:
            if self.backend == "video-llava":
                # Videos API (list-of-frames acceptable for many processors)
                inputs = self.processor(text=[prompt], videos=[frames_rgb], return_tensors="pt")
            else:
                # Image-sequence as multimodal context (many processors accept 'images' list)
                inputs = self.processor(text=[prompt], images=[frames_rgb[0]], return_tensors="pt")
                # For simplicity we feed the key frame. Extend to multiple images if backend supports it.
            # Move tensors for non-CPU
            if self.device in ("cuda", "mps"):
                for k, v in inputs.items():
                    if hasattr(v, "to"):
                        inputs[k] = v.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                text_out = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            latency = time.perf_counter() - t0
            return {
                "vlm_enabled": True,
                "summary": text_out,
                "avg_brightness": np.nan,
                "latency_s": latency
            }
        except Exception as e:
            latency = time.perf_counter() - t0
            return {
                "vlm_enabled": False,
                "summary": f"Stub (VLM error: {e})",
                "avg_brightness": float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames_bgr])),
                "latency_s": latency
            }

# -----------------------------
# Vision analytics
# -----------------------------
def detect_water_like(hsv, gray, ksize=5):
    """Heuristic water/flooding mask via hue/saturation + low texture."""
    h, s, v = cv2.split(hsv)
    blue_mask = cv2.inRange(h, 90, 130)      # blue-ish hue (0-179 scale)
    low_sat   = cv2.inRange(s, 0, 60)        # gray-ish low saturation
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    lap_var = cv2.blur(lap * lap, (ksize, ksize))
    low_tex = (lap_var < 25).astype(np.uint8) * 255
    mask = cv2.bitwise_or(blue_mask, low_sat)
    mask = cv2.bitwise_and(mask, low_tex)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return mask

def bgr_to_luma(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def hist_distance(prev_gray, gray):
    hist_prev = cv2.calcHist([prev_gray],[0],None,[32],[0,256])
    hist_curr = cv2.calcHist([gray],[0],None,[32],[0,256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)
    return cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)

def draw_regions(frame, mask, color, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame, f"{label}:{int(area)}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0=webcam, else path/URL")
    ap.add_argument("--backend", type=str, default="qwen2-vl",
                    choices=["qwen2-vl","llava-onevision","internvl2","video-llava"],
                    help="Model backend for reasoning")
    ap.add_argument("--batch", type=int, default=4, help="frames per VLM batch")
    ap.add_argument("--stride", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--max_frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    ap.add_argument("--show", type=int, default=1, help="show live window")
    ap.add_argument("--save", type=int, default=1, help="write output video")
    ap.add_argument("--out", type=str, default="out_multi_backend.mp4", help="output mp4")
    ap.add_argument("--vlm", type=int, default=1, help="enable VLM (0 uses stub)")
    ap.add_argument("--log_csv", type=str, default="metrics_log_multi_backend.csv")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    # Open capture
    src = 0 if args.source.strip()=="0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, src_fps, (width, height))

    # Components
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    vlm = VLMModel(enable=bool(args.vlm), backend=args.backend, max_new_tokens=args.max_new_tokens)

    # Logs
    rows = []
    start_wall = time.perf_counter()
    total_processed = 0
    prev_gray_for_hist = None
    flow_prev_gray = None
    batch_buf = deque(maxlen=args.batch)
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % max(1, args.stride) != 0:
            continue

        t_start = time.perf_counter()
        gray = bgr_to_luma(frame)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Motion (MOG2) + clean
        t0 = time.perf_counter()
        fg = bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((5,5), np.uint8), iterations=1)
        motion_latency = time.perf_counter() - t0

        # Optical flow debris proxy
        t1 = time.perf_counter()
        debris_mask = np.zeros_like(fg)
        if flow_prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(flow_prev_gray, gray, None,
                                                0.5, 3, 21, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, debris_mask = cv2.threshold(mag_norm, 64, 255, cv2.THRESH_BINARY)
            debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        flow_prev_gray = gray.copy()
        flow_latency = time.perf_counter() - t1

        # Water-like mask
        t2 = time.perf_counter()
        water_mask = detect_water_like(hsv, gray)
        water_latency = time.perf_counter() - t2

        # Intensity changes
        hist_dist = np.nan
        mean_delta_l = np.nan
        if prev_gray_for_hist is not None:
            hist_dist = float(hist_distance(prev_gray_for_hist, gray))
            mean_delta_l = float(np.mean(cv2.absdiff(prev_gray_for_hist, gray)))
        prev_gray_for_hist = gray.copy()

        # Batch for VLM
        batch_buf.append(frame.copy())
        vlm_out = {"summary": "Not run", "latency_s": np.nan, "vlm_enabled": False}
        if len(batch_buf) == args.batch and vlm is not None:
            vlm_out = vlm.infer(list(batch_buf))
            batch_buf.clear()

        # Visualization
        vis = frame.copy()
        draw_regions(vis, fg, (0,255,0), "motion")
        draw_regions(vis, debris_mask, (0,0,255), "debris")
        draw_regions(vis, water_mask, (255,0,0), "flood")
        overlap = cv2.bitwise_and(water_mask, debris_mask)
        overlap_area = int(cv2.countNonZero(overlap))
        if overlap_area > 5000:
            cv2.putText(vis, "ALERT: Possible flood debris movement",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # Stats
        frame_latency = time.perf_counter() - t_start
        total_processed += 1
        elapsed = time.perf_counter() - start_wall
        proc_fps = total_processed / max(1e-6, elapsed)
        rtf = proc_fps / src_fps

        # HUD
        hud = [
            f"Frame {frame_idx} | FPS: {proc_fps:5.2f} | RTF: {rtf:4.2f}",
            f"motion {motion_latency*1000:5.1f}ms  flow {flow_latency*1000:5.1f}ms  water {water_latency*1000:5.1f}ms",
            f"meanΔL {mean_delta_l:.2f}  histDist {hist_dist:.3f}",
        ]
        if not np.isnan(vlm_out.get("latency_s", np.nan)):
            hud.append(f"VLM({ 'on' if vlm_out.get('vlm_enabled') else 'stub' }) {vlm_out['latency_s']*1000:.1f}ms")
        y = 20
        for line in hud:
            cv2.putText(vis, line, (12, height - 60 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            y += 18

        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow("Real-Time – Multi-Backend VLM", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Log
        rows.append({
            "frame_idx": frame_idx,
            "proc_time_s": frame_latency,
            "proc_fps": proc_fps,
            "rtf": rtf,
            "motion_latency_s": motion_latency,
            "flow_latency_s": flow_latency,
            "water_latency_s": water_latency,
            "mean_delta_l": mean_delta_l,
            "hist_bhattacharyya": hist_dist,
            "water_area_px": int(cv2.countNonZero(water_mask)),
            "debris_area_px": int(cv2.countNonZero(debris_mask)),
            "motion_area_px": int(cv2.countNonZero(fg)),
            "overlap_area_px": overlap_area,
            "vlm_summary": vlm_out.get("summary", ""),
            "vlm_latency_s": vlm_out.get("latency_s", np.nan),
            "vlm_enabled": vlm_out.get("vlm_enabled", False),
            "backend": args.backend,
        })

        if args.max_frames and total_processed >= args.max_frames:
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Write logs
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.log_csv, index=False)
        print(f"[INFO] Wrote per-frame metrics to: {args.log_csv}")

    # Summary JSON
    out_json = os.path.splitext(args.log_csv)[0] + "_summary.json"
    summary = {
        "total_frames": total_processed,
        "source_fps": src_fps,
        "avg_proc_fps": float(np.nanmean([r['proc_fps'] for r in rows])) if rows else 0.0,
        "avg_rtf": float(np.nanmean([r['rtf'] for r in rows])) if rows else 0.0,
        "backend": args.backend,
        "notes": "RTF = processed_fps / source_fps; RTF > 1.0 means faster than real-time."
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote run summary to: {out_json}")


if __name__ == "__main__":
    main()
