# Week 6 – Graph-RAG and Multi-Hop Retrieval

Purpose
Week 6 focused on building a Graph-based Retrieval-Augmented Generation (Graph-RAG) system that connects factual nodes (events, locations, causes, and responses) for Florida’s Hurricane Irma dataset.  
It replaced linear text retrieval with multi-hop reasoning to answer complex questions such as *“Which Florida counties received the highest FEMA relief after Irma?”

Architecture Overview
1. Entity Extraction
   - Parsed textual FEMA / NOAA records to identify entities like `County`, `Damage_Type`, `Event_Date`, and `Relief_Amount`.
2. Graph Construction
   - Constructed a `networkx` graph where nodes represent entities and edges represent causal or spatial links.
3. Multi-Hop QA
   - Used BFS traversal to gather multi-level evidence paths before passing context to the LLM (Graph-RAG + LLM).
4. Frontend
   - Built a Streamlit interface with five tabs: Query/Answer, Evidence, Reasoning Trace, Graph Visualization, and Florida Data.

Key Improvements
- Baseline RAG Accuracy: 0.70 → 0.92
- Added evidence-trace transparency.
- Graph-based caching reduced redundant retrieval calls by 35 %.
- Latency increased slightly (3.2 → 4.5 s) due to multi-hop traversal.

Evaluation Metrics
| Variant | Accuracy | Latency (s) | Notes |
|----------|-----------|--------------|--------|
| Baseline RAG | 0.70 | 3.2 | Single-hop textual retrieval |
| Graph-RAG | 0.87 | 3.8 | Entity link reasoning |
| Multi-Hop | 0.92 | 4.5 | Cause-effect reasoning |

Files Included
- env_week6.json – environment snapshot  
- week6_run_config.json – graph and retrieval settings  
- ablation_results_week6.csv – metrics table  
- README.md – this file  

Next Steps
Integrate visual data and agent control logic (handled in Week 7) to couple Graph-RAG retrieval with generative model pipelines.
