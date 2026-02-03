# AI-Driven 3D Hurricane Visualization for Decision Support


##  3D Visualization Demo Videos

### Hurricane Irma (2017) — 3D Path Visualization
[![Watch on YouTube](https://img.youtube.com/vi/ZvJ8jOmbHDE/0.jpg)](https://youtu.be/ZvJ8jOmbHDE)

A stylized reconstruction of Hurricane Irma’s trajectory used to ground the visualization pipeline and demonstrate spatial–temporal context.

---

### Uncertainty & Ensemble Spread Visualization
[![Watch on YouTube](https://img.youtube.com/vi/nTIp0jjtJEk/0.jpg)](https://youtu.be/nTIp0jjtJEk)

Demonstrates uncertainty-aware visualization using ensemble concepts (e.g., best-case, median, worst-case paths) rather than a single deterministic forecast.

---

### Recovery Rays & Impact Heatmap Demonstration
[![Watch on YouTube](https://img.youtube.com/vi/TCNdMnLFamw/0.jpg)](https://youtu.be/TCNdMnLFamw)

Illustrates post-impact recovery prioritization using spatial intensity cues (heatmaps and recovery rays) to support planning decisions.

---

##  How the Visualization Works (High Level)

- Forecast and risk indicators are converted into structured spatial inputs  
- Outputs are rendered into a **frame-based 3D environment**  
- Visual layers represent:
  - Storm motion and timing
  - Uncertainty envelopes
  - Impact intensity
  - Recovery prioritization cues  

The videos are **reproducible demonstrations**, not real-time simulations.

---

##  Data Sources (Demonstration Case)

- **HURDAT2 (NOAA)** – Historical best-track storm trajectory and timing  
- **ERA5 (ECMWF)** – Hazard context fields (wind, pressure, precipitation)  
- **Florida geographic reference layers** – Coastline and spatial grounding  

**Note:**  
This system does **not ingest raw satellite imagery**. All visuals are stylized, interpretable representations designed for decision communication.

---

##  Limitations

- Demonstration videos are **offline and pre-rendered**
- Not a real-time operational forecasting system
- Visuals represent **relative risk and uncertainty**, not exact physical outcomes
- Intended for **planning support**, not emergency command automation

---

##  Planned Extensions

- Near-real-time data ingestion  
- Interactive controls (time, viewpoint, scenario assumptions)  
- Ensemble-aware visualization (P10 / P50 / P90)  
- Cross-region scenario comparison demos  

---

---
