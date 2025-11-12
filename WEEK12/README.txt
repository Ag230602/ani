# 🌪️ AI-Driven Spatial + Temporal + Sequence Video Intelligence System  
### **Track A + Track B + Track C Integrated Pipeline**  
**Author:** Adrija Ghosh  
**Course:** Week 12 – Video LLM Intelligence  
**Project: Hurricane Irma Satellite Simulation Analysis**

---

## 📌 1. Project Overview

This project integrates all three tracks of the Video LLM Intelligence workflow:

| Track | Purpose |
|-------|---------|
| **Track A** | Real-time video processing, multi-frame batching, VLM reasoning, latency, FPS, RTF |
| **Track B** | Temporal reasoning: step order, duration, repetition, trend analysis |
| **Track C** | Spatial grounding: bounding boxes, IoU, stability, pose, flood & debris regions |

Together, these form a complete **spatiotemporal video analysis system** capable of interpreting hurricane satellite footage (e.g., Hurricane Irma), combining **region detection**, **temporal patterning**, and **sequence-level VLM reasoning**.

The system supports:
- Webcam input  
- Local MP4 videos  
- RTSP / livestream URLs  
- Disaster, weather, or general video sequences  

---

## 🧩 2. Track-Wise System Capabilities

### **2.1 Track A – Real-Time Video Analysis**
- Captures frames from webcam / video / streams  
- Frame sampling (`--stride`)  
- Multi-frame batching (`--batch`)  
- Video-LLM (VLM) reasoning on frame windows  
- Real-time overlays (HUD, FPS, RTF)  
- Component-level latency logging  
- Reproducibility logs: `run_config.json`, `environment.json`  

---

### **2.2 Track B – Temporal Reasoning & Sequence Understanding**

Track B adds temporal intelligence:

#### ✔ Step-Order Reasoning  
Detects if key hurricane phases occur in expected order:
1. Turbulence  
2. Debris motion  
3. Flood formation  
4. Eye stabilization  

#### ✔ Duration Measurement  
Tracks:
- Flood duration  
- Debris bursts  
- Storm-eye persistence  
- Motion-active periods  

#### ✔ Repetition Counting  
Counts repeated:
- Motion spikes  
- Flood pulses  
- Brightness fluctuations  

#### ✔ Temporal Windowing  
Each 8-frame window produces:
- Average brightness  
- Activity level  
- VLM window summary  
- Temporal smoothness/stability  

#### ✔ VLM Multi-Frame Reasoning  
The VLM produces:
- “Flood increasing”
- “Scene stabilizing”
- “Debris intensifying”
- “Brightness trending upward”



---

### **2.3 Track C – Spatial Grounding**
- YOLO object detection  
- Pose/keypoint detection (optional)  
- IoU between frames  
- Centroid stability  
- Flood/water region mask  
- Debris mask using optical flow  
- Spatial overlays and segmentation  
- Flood + debris composite alerts  

---

##  3. Input Video (Example – Hurricane Irma)
Used during testing:

