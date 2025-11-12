#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track A – Real-Time Video Loop (CPU-Compatible Video-LLaVA-7B)
---------------------------------------------------------------
Detect flooding, debris, and intensity changes, and run reasoning
across frames using Video-LLaVA-7B (CPU fallback mode).
"""
import argparse
import time
import os
import json
from collections import deque
import cv2
import numpy as np
import pandas as pd

# -----------------------------
# Optional VLM: Video-LLaVA-7B CPU
# -----------------------------
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class VLMModel:
    """
    Video-LLaVA-7B (CPU-compatible) reasoning backend.
    Falls back to stub mode if model not found or CPU too slow.
    """
    def __init__(self, enable=True):
        self.enable = enable
        self.model = None
        self.processor = None
        if not enable:
            print("[INFO] VLM disabled (stub mode).")
            return

        try:
            model_name = "LanguageBind/Video-LLaVA-7B"
            print(f"[INFO] Loading {model_name} on CPU (this may take a few minutes)...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            self.model.eval()
            print("[INFO] Video-LLaVA-7B loaded successfully (CPU mode).")
        except Exception as e:
            print(f"[WARN] Could not load Video-LLaVA-7B; using stub. Reason: {e}")
            self.enable = False

    def infer(self, frames_bgr):
        t0 = time.perf_counter()
        if not self.enable or self.model is None:
            avg_brightness = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames_bgr]))
            latency = time.perf_counter() - t0
            return {"vlm_enabled": False, "summary": "Stub mode", "avg_brightness": avg_brightness, "latency_s": latency}

        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        inputs = self.processor(text=["Describe changes or flooding in this video sequence."],
                                videos=[frames_rgb], return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64)
            text_out = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        latency = time.perf_counter() - t0
        return {"vlm_enabled": True, "summary": text_out, "avg_brightness": np.nan, "latency_s": latency}

# -----------------------------
# Vision analytics helpers
# -----------------------------
def detect_water_like(hsv, gray, ksize=5):
    h, s, v = cv2.split(hsv)
    blue_mask = cv2.inRange(h, 90, 130)
    low_sat = cv2.inRange(s, 0, 60)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    lap_var = cv2.blur(lap * lap, (ksize, ksize))
    low_tex = (lap_var < 25).astype(np.uint8) * 255
    mask = cv2.bitwise_or(blue_mask, low_sat)
    mask = cv2.bitwise_and(mask, low_tex)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return mask

def bgr_to_luma(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

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
    ap.add_argument("--source", type=str, default="0", help="0 for webcam or path to file/URL")
    ap.add_argument("--batch", type=int, default=4, help="frames per batch for VLM")
    ap.add_argument("--stride", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--out", type=str, default="out_llava_cpu.mp4")
    ap.add_argument("--vlm", type=int, default=1)
    args = ap.parse_args()

    src = 0 if args.source.strip()=="0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Could not open source: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, src_fps, (width,height))

    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    vlm = VLMModel(enable=bool(args.vlm))

    logs = []
    prev_gray, prev_hist = None, None
    flow_prev_gray = None
    buf = deque(maxlen=args.batch)
    frame_idx, total = -1, 0
    start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % args.stride != 0: continue

        t0 = time.perf_counter()
        gray = bgr_to_luma(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fg = bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        debris_mask = np.zeros_like(fg)
        if flow_prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(flow_prev_gray, gray, None,0.5,3,21,3,5,1.2,0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag_norm = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            _, debris_mask = cv2.threshold(mag_norm,64,255,cv2.THRESH_BINARY)
        flow_prev_gray = gray.copy()

        water_mask = detect_water_like(hsv, gray)
        hist_dist = mean_delta = np.nan
        if prev_gray is not None:
            hist_dist = float(hist_distance(prev_gray, gray))
            mean_delta = float(np.mean(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray.copy()

        buf.append(frame.copy())
        vlm_out = {"summary": "Not run", "latency_s": np.nan}
        if len(buf)==args.batch and vlm.enable:
            vlm_out = vlm.infer(list(buf))
            buf.clear()

        vis = frame.copy()
        draw_regions(vis, fg, (0,255,0), "motion")
        draw_regions(vis, debris_mask, (0,0,255), "debris")
        draw_regions(vis, water_mask, (255,0,0), "flood")
        overlap = cv2.bitwise_and(water_mask, debris_mask)
        overlap_area = int(cv2.countNonZero(overlap))
        if overlap_area>5000:
            cv2.putText(vis,"ALERT: Possible flood debris movement",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        latency = time.perf_counter()-t0
        total += 1
        elapsed = time.perf_counter()-start
        fps = total/max(1e-6,elapsed)
        rtf = fps/src_fps
        cv2.putText(vis,f"FPS:{fps:.2f} RTF:{rtf:.2f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        if args.show:
            cv2.imshow("Video-LLaVA-CPU",vis)
            if cv2.waitKey(1)&0xFF==27: break
        if writer: writer.write(vis)

        logs.append({
            "frame":frame_idx,"fps":fps,"rtf":rtf,
            "mean_delta":mean_delta,"hist_dist":hist_dist,
            "vlm_summary":vlm_out["summary"],"vlm_latency":vlm_out["latency_s"],
            "water_area":int(cv2.countNonZero(water_mask)),
            "debris_area":int(cv2.countNonZero(debris_mask)),
            "motion_area":int(cv2.countNonZero(fg)),
            "overlap_area":overlap_area
        })
        if args.max_frames and total>=args.max_frames: break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    pd.DataFrame(logs).to_csv("metrics_log_llava_cpu.csv",index=False)
    print("[INFO] Logs written to metrics_log_llava_cpu.csv")

if __name__=="__main__":
    main()
