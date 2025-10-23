# Week 7 – Visual Generation and Deployment Integration

### Objective
To extend Graph-RAG with a visual generation module using Stable Diffusion, ControlNet, and Stable Video Diffusion (SVD).  
This pipeline creates educational videos depicting Hurricane Irma’s damage scenarios for Florida.

### Components
1. LoRA Fine-Tuning
   - Base: `runwayml/stable-diffusion-v1-5`  
   - Dataset: ≈ 80 Irma images from trusted news sources.  
   - Parameters: 300 steps, lr = 8e-5, batch = 1 (optimized for CPU runtime).
2. ControlNet Depth Conditioning
   - Model: `lllyasviel/sd-controlnet-depth`  
   - Ensures structural fidelity in buildings and flood zones via depth maps.
3. Stable Video Diffusion
   - Model: `stabilityai/stable-video-diffusion-img2vid-xt`  
   - Generates 12 frames at 8 FPS to animate damage progression.
4. Integration and Deployment
   - Backend: FastAPI serving `/query` and `/generate` endpoints.  
   - Frontend: Streamlit dashboard connecting RAG outputs to image/video generators.  
   - Agent Loop: Planner → Executor → Aggregator routes retrieval to generation automatically.

  Ablation Highlights
| Variant | Accuracy | Latency (s) | Image FID | Notes |
|----------|-----------|--------------|------------|--------|
| Baseline RAG | 0.70 | 3.2 | – | Text-only |
| Graph-RAG | 0.87 | 3.8 | – | Entity reasoning |
| Multi-Hop | 0.92 | 4.5 | – | Causal links |
| LoRA + ControlNet | 0.94 | 5.3 | 0.19 | Visual realism |
| SVD Final | 0.95 | 5.6 | 0.17 | Full multimodal pipeline |

Achievements
- CPU-compatible LoRA training without GPU dependence.  
- Dynamic streaming image fetcher to avoid disk overflow.  
- End-to-end integration with FastAPI backend.  
- Reproducibility files for evaluation and setup.

Files Included
- env_week7.json  
- week7_run_config.json  
- ablation_results_week7.csv  
- README.md

Next Steps
Week 8 adds a Voice-Interactive LLM and Conversational Visualization module for full educational interactivity.
