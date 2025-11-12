# Team Report — Track C & Track A Integrated Spatial + Temporal Video Intelligence System

## 1. Domain Focus and Setup Summary

This project integrates **Track C (Spatial Grounding)** and **Track A (Real-Time Video Analysis)** to build a unified video-intelligence pipeline capable of:
- Capturing and sampling frames from **webcam, video files, or live streams**
- Performing **object/region detection**, **pose tracking**, **spatial overlays**, and **IoU/stability metrics**
- Executing **temporal reasoning**, **flood/water detection**, **storm-eye heuristics**, and **optical-flow debris analysis**
- Processing **multi-frame batches** using a Video-LLM/VLM reasoning stub
- Logging **latency, FPS, real-time factor (RTF)**, and detailed per-frame analytics
- Supporting **ablation studies** and **full reproducibility** via run configuration logs

The system was tested on:
- **Irma Satellite Simulation Video**  
  Path: `C:\Users\Adrija\OneDrive\Documents\Week9\irma_satellite_simulation.mp4`

The unified system incorporates both **spatial grounding from Track C** and **temporal reasoning from Track A**, making it capable of detecting and characterizing complex disaster scenarios in video footage.

---

## 2. Key Components and Capabilities

### **2.1 Spatial Grounding (Track C)**
- MediaPipe Pose detection  
- YOLO bounding-box detection  
- Pixel-level region extraction  
- IoU computation between consecutive frames  
- Positional stability measurement  
- Annotated bounding boxes and pose skeletons  
- CSV output of spatial metrics

### **2.2 Temporal Reasoning & Scene Understanding (Track A)**
- Motion segmentation (Background Subtractor MOG2)  
- Debris detection via optical flow  
- Water/flood-like region detection (HSV + texture)  
- Storm-eye heuristic for hurricane/satellite imagery  
- Histogram change detection  
- Flood-intensity scoring (0–1 scale)  
- Optional YOLO object detection  
- Multi-frame batching with VLM reasoning stub  
- Per-component latency measurement  
- RTF, FPS calculation  
- Ablation flags (motion, water, debris, YOLO, VLM, storm-eye)  
- Full reproducibility logs (config, environment, summary)

---

## 3. Metrics and Latency Results (Sample Run)

| Metric | Value |
|--------|-------|
| Mean IoU (Track C) | ~0.9957 |
| Positional Stability | ~0.58 px/frame |
| Processed Frames | 480 |
| Mean FPS | 28–32 FPS (depending on modules toggled) |
| Real-Time Factor (RTF) | ~0.9–1.2 |
| Mean Flood Score | ~0.10–0.23 |
| Water Region Area | Video-dependent |
| YOLO Objects Detected | Optional |
| VLM Avg Brightness | Batch-based (stub output) |

**Latency breakdown (per frame):**
- Motion detection: 1–4 ms  
- Optical-flow debris: 3–10 ms  
- Water detection: 1–2 ms  
- YOLO (optional): 10–30 ms  
- Storm-eye detection: 2–6 ms  
- VLM inference (batch-based): ~10–20 ms  

---

## 4. Annotated Video Examples (Summary)

The integrated system generates:
- **Bounding-box visualizations**
- **Pose skeleton overlays**
- **Flood + debris overlap alerts**
- **Storm-eye region markers**
- **Color-coded region masks**
- **HUD with FPS, RTF, latency, flood score**

Track C produces:
- `spatial_grounding_annotated.mp4`
- `trackC_spatial_metrics.csv`

Track A produces:
- `annotated.mp4` with overlays  
- run_config.json
- environment.json  

---

## 5. Reproducibility Logs & Ablation Support

Each run auto-generates a folder under:

```
runs/<timestamp>/
```

Containing:
- `annotated.mp4` — Final video  
- `run_config.json` — Arguments used  
- `environment.json` — Python/OpenCV/system versions  

### **Ablation toggles:**
```
--motion 1/0  
--debris 1/0  
--water 1/0  
--yolo 1/0  
--storm_eye 1/0  
--vlm 1/0
```

Used to identify component contributions:
- Motion-only  
- Water-only  
- Debris-only  
- Full system  
- No-VLM vs VLM-enabled  

---

## 6. Reflection: Integration With Prior Work (Irma Hurricane Analysis)

This work builds on earlier **Hurricane Irma simulation** research that utilized:
- NOAA data  
- NASA MODIS imagery  
- Stable Diffusion + ControlNet + SVD  
- Graph-RAG knowledge systems  
- 3D video generation and educational simulation pipelines  

### **Enhancement Achieved Here**
The new real-time analyzer adds:
- **Temporal reasoning** for satellite video  
- **Storm-eye localization**  
- **Water + debris flood-risk detection**  
- **Spatial grounding overlays**  
- **Batch-based VLM reasoning**  
- **Full reproducibility & ablations** making this suitable for academic evaluation and hackathon scoring  

Together, the system transforms raw video footage into **interpretable, multi-modal disaster intelligence** that can be used for:
- Emergency simulations  
- Educational tools  
- Automated early-warning insights  
- Model benchmarking and research studies  

---

## 7. Conclusion

This integrated Track A + Track C system fulfills all required objectives:

- ✔ Captures and samples frames from webcam / video / live stream  
- ✔ Processes multi-frame batches using VLM  
- ✔ Logs reasoning outputs, latency, FPS, RTF  
- ✔ Provides spatial grounding metrics (IoU, stability)  
- ✔ Incorporates ablation studies  
- ✔ Includes reproducibility logs  
- ✔ Generates annotated visual outputs  
- ✔ Supports disaster-focused analysis (Irma hurricane)

It is fully compliant  academic criteria, scalable, and extensible for future AI-augmented disaster simulation research.

Command :python 'C:\Users\Adrija\Downloads\WEEK12\realtime_video_loop_trackA.py' --source 'C:\Users\Adrija\OneDrive\Documents\Week9\irma_satellite_simulation.mp4' --yolo 1 --storm_eye 1 --motion 1 --debris 1 --water 1 --vlm 1 --stride 1

