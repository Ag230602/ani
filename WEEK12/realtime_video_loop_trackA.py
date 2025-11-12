#!/usr/bin/env python3

import argparse
import time
import os
import json
import platform
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import pandas as pd


# ----------------------------------------------------------
# OPTIONAL YOLO LOADER
# ----------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("YOLO not available — continuing without YOLO.")


# ----------------------------------------------------------
# VIDEO LLM / VLM STUB
# ----------------------------------------------------------
class VLMModel:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def infer(self, frames):
        if not self.enabled:
            return None

        t0 = time.perf_counter()
        avg_brightness = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]))
        time.sleep(0.01)  # simulate compute
        latency = time.perf_counter() - t0

        return {
            "avg_brightness": avg_brightness,
            "latency_s": latency
        }


# ----------------------------------------------------------
# WATER (FLOOD) DETECTOR
# ----------------------------------------------------------
def detect_water_like(hsv, gray):
    h, s, v = cv2.split(hsv)
    blue_mask = cv2.inRange(h, 90, 130)
    low_sat = cv2.inRange(s, 0, 60)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    lap2 = cv2.blur(lap * lap, (5,5))
    low_tex = (lap2 < 25).astype(np.uint8) * 255

    mask = cv2.bitwise_and(cv2.bitwise_or(blue_mask, low_sat), low_tex)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return mask


# ----------------------------------------------------------
# DRAW BOUNDING REGIONS
# ----------------------------------------------------------
def draw_regions(frame, mask, color, label):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame, f"{label}:{area:.0f}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ----------------------------------------------------------
# STORM-EYE HEURISTIC DETECTOR
# ----------------------------------------------------------
def detect_storm_eye(gray):
    # center crop
    h, w = gray.shape
    cy, cx = h//2, w//2
    eye_region = gray[cy-60:cy+60, cx-60:cx+60]
    if eye_region.size == 0:
        return False, 0

    bright = np.mean(eye_region)
    is_eye = bright < 80  # heuristic threshold
    return is_eye, bright


# ----------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="annotated.mp4")

    # Ablations
    parser.add_argument("--motion", type=int, default=1)
    parser.add_argument("--debris", type=int, default=1)
    parser.add_argument("--water", type=int, default=1)
    parser.add_argument("--storm_eye", type=int, default=1)
    parser.add_argument("--yolo", type=int, default=1)
    parser.add_argument("--vlm", type=int, default=1)

    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--show", type=int, default=1)
    args = parser.parse_args()


    # ----------------------------------------------------------
    # CREATE RUN FOLDER (REPRODUCIBILITY)
    # ----------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/{ts}"
    os.makedirs(run_dir, exist_ok=True)

    out_video_path = f"{run_dir}/{args.out}"
    csv_path = f"{run_dir}/metrics_log.csv"
    summary_path = f"{run_dir}/summary.json"
    config_path = f"{run_dir}/run_config.json"
    env_path = f"{run_dir}/environment.json"

    # Save config
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(env_path, "w") as f:
        json.dump({
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
        }, f, indent=2)


    # ----------------------------------------------------------
    # LOAD VIDEO INPUT
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_video_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W,H))


    # ----------------------------------------------------------
    # INITIALIZATIONS
    # ----------------------------------------------------------
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16)
    vlm = VLMModel(enabled=bool(args.vlm))

    yolo = None
    if args.yolo and YOLO_AVAILABLE:
        try:
            yolo = YOLO("yolov8n.pt")
        except:
            print("YOLO model failed; disabling YOLO.")
            yolo = None

    prev_gray = None
    prev_hist_gray = None
    batch_buf = deque(maxlen=args.batch)
    logs = []

    global_start = time.perf_counter()
    processed = 0
    frame_index = -1


    # ----------------------------------------------------------
    # PROCESS LOOP
    # ----------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % args.stride != 0:
            continue

        processed += 1
        frame_t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        overlay = frame.copy()


        # ----------------------------------------------------------
        # MOTION
        # ----------------------------------------------------------
        t_motion = 0
        if args.motion:
            t0 = time.perf_counter()
            motion_mask = bg.apply(frame)
            motion_mask = cv2.medianBlur(motion_mask, 5)
            t_motion = time.perf_counter() - t0
        else:
            motion_mask = np.zeros_like(gray)


        # ----------------------------------------------------------
        # OPTICAL-FLOW DEBRIS
        # ----------------------------------------------------------
        t_flow = 0
        debris_mask = np.zeros_like(gray)
        if args.debris and prev_gray is not None:
            t1 = time.perf_counter()
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5,3,21,3,5,1.2,0)
            mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag_norm = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            _, debris_mask = cv2.threshold(mag_norm, 64,255,cv2.THRESH_BINARY)
            t_flow = time.perf_counter() - t1

        prev_gray = gray.copy()


        # ----------------------------------------------------------
        # WATER / FLOOD DETECT
        # ----------------------------------------------------------
        t_water = 0
        if args.water:
            t2 = time.perf_counter()
            water_mask = detect_water_like(hsv, gray)
            t_water = time.perf_counter() - t2
        else:
            water_mask = np.zeros_like(gray)


        # ----------------------------------------------------------
        # STORM EYE DETECT
        # ----------------------------------------------------------
        t_eye = 0
        storm_eye_flag = False
        storm_eye_brightness = 0
        if args.storm_eye:
            t3 = time.perf_counter()
            storm_eye_flag, storm_eye_brightness = detect_storm_eye(gray)
            t_eye = time.perf_counter() - t3


        # ----------------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------------
        t_yolo = 0
        yolo_count = 0
        if args.yolo and yolo is not None:
            t4 = time.perf_counter()
            results = yolo(frame, verbose=False)
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1,y1,x2,y2 = box[:4].astype(int)
                    yolo_count += 1
                    cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)
            t_yolo = time.perf_counter() - t4


        # ----------------------------------------------------------
        # REGION DRAWING
        # ----------------------------------------------------------
        if args.motion: draw_regions(overlay, motion_mask, (0,255,0), "motion")
        if args.debris: draw_regions(overlay, debris_mask, (0,0,255), "debris")
        if args.water: draw_regions(overlay, water_mask, (255,0,0), "water")


        # ----------------------------------------------------------
        # OVERLAP ALERT
        # ----------------------------------------------------------
        overlap = cv2.bitwise_and(water_mask, debris_mask)
        overlap_area = int(cv2.countNonZero(overlap))
        if overlap_area > 5000:
            cv2.putText(overlay,"ALERT: Flood+Debris",(20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)


        # ----------------------------------------------------------
        # HIST CHANGE
        # ----------------------------------------------------------
        hist_dist = None
        mean_delta_l = None

        if prev_hist_gray is not None:
            hist_dist = float(cv2.compareHist(
                cv2.calcHist([prev_hist_gray],[0],None,[32],[0,256]),
                cv2.calcHist([gray],[0],None,[32],[0,256]),
                cv2.HISTCMP_BHATTACHARYYA
            ))
            mean_delta_l = float(np.mean(cv2.absdiff(prev_hist_gray, gray)))

        prev_hist_gray = gray.copy()


        # ----------------------------------------------------------
        # VLM INFERENCE (BATCH)
        # ----------------------------------------------------------
        vlm_latency = None
        vlm_avg = None

        if args.vlm:
            batch_buf.append(frame.copy())
            if len(batch_buf) == args.batch:
                res = vlm.infer(list(batch_buf))
                vlm_latency = res["latency_s"]
                vlm_avg = res["avg_brightness"]
                batch_buf.clear()


        # ----------------------------------------------------------
        # METRICS — FPS + RTF
        # ----------------------------------------------------------
        elapsed = time.perf_counter() - global_start
        fps_proc = processed / elapsed
        rtf = fps_proc / fps
        total_latency = time.perf_counter() - frame_t0

        cv2.putText(overlay, f"FPS:{fps_proc:.2f} RTF:{rtf:.2f}",
                    (10,H-20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)


        # ----------------------------------------------------------
        # WRITE FRAME
        # ----------------------------------------------------------
        writer.write(overlay)

        if args.show:
            cv2.imshow("Video Analyzer", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break


        # ----------------------------------------------------------
        # LOG ENTRY
        # ----------------------------------------------------------
        logs.append({
            "frame": frame_index,
            "latency_total_s": total_latency,
            "latency_motion_s": t_motion,
            "latency_flow_s": t_flow,
            "latency_water_s": t_water,
            "latency_yolo_s": t_yolo,
            "latency_storm_eye_s": t_eye,
            "vlm_latency_s": vlm_latency if vlm_latency else np.nan,
            "vlm_avg": vlm_avg if vlm_avg else np.nan,
            "fps": fps_proc,
            "rtf": rtf,
            "water_area": int(cv2.countNonZero(water_mask)),
            "debris_area": int(cv2.countNonZero(debris_mask)),
            "motion_area": int(cv2.countNonZero(motion_mask)),
            "overlap_area": overlap_area,
            "storm_eye_detected": int(storm_eye_flag),
            "storm_eye_brightness": float(storm_eye_brightness),
            "yolo_count": yolo_count,
            "hist_distance": hist_dist if hist_dist else np.nan,
            "mean_delta_l": mean_delta_l if mean_delta_l else np.nan
        })


    # ----------------------------------------------------------
    # SAVE LOGS
    # ----------------------------------------------------------
    df = pd.DataFrame(logs)
    df.to_csv(csv_path, index=False)

    summary = {
        "frames_processed": processed,
        "avg_fps": float(df.fps.mean()),
        "avg_rtf": float(df.rtf.mean()),
        "avg_water_area": float(df.water_area.mean()),
        "avg_debris_area": float(df.debris_area.mean()),
        "avg_overlap_area": float(df.overlap_area.mean()),
        "storm_eye_frequency": int(df.storm_eye_detected.sum()),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================")
    print(" ✔ Run Complete")
    print(f" ✔ Output Video : {out_video_path}")
    print(f" ✔ Metrics CSV  : {csv_path}")
    print(f" ✔ Summary JSON : {summary_path}")
    print("===================================\n")


if __name__ == "__main__":
    main()
