# Curriculum Deep Structure Analysis

Computational comparative analysis of undergraduate curriculum structures across China, USA, and Europe using multilingual embeddings and GPT-4o agents.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18284324.svg)](https://doi.org/10.5281/zenodo.18284324)

---

## Overview

This project measures hidden structural differences in university curricula using three computable metrics:

- **Structural Rigidity Index (F)** — proportion of compulsory credits
- **Semantic Density (ρ) & Modularity (Q)** — cosine similarity and community structure of embedding vectors
- **Topic Responsiveness Score (S)** — RAG retrieval density for emerging topics

Universities: **NJU, PKU** (China) · **MIT, Stanford** (USA) · **Heidelberg, LMU** (Europe)

---

## Repository Structure

```
Curriculum_Analysis_Project/
├── scripts/
│   ├── config.py                  # Paths and shared constants
│   ├── 01_data_processing.py      # PDF/HTML → cleaned JSONL corpus
│   ├── 02_build_index.py          # BGE-M3 embeddings → FAISS index
│   └── 03_run_experiment.py       # Core experiment: GPT-4o agents + semantic metrics
├── outputs/                       # Raw JSON results (created at runtime)
├── requirements.txt
└── README.md
```

> **Raw data & FAISS indices** are on Zenodo: [10.5281/zenodo.18284324](https://doi.org/10.5281/zenodo.18284324).  
> Download and extract into `data/` before running steps 1–2.

---

## Setup

```bash
git clone https://github.com/Jie-Ni/Curriculum_Analysis_Project.git
cd Curriculum_Analysis_Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set your OpenAI API key (required for step 3)
export OPENAI_API_KEY=sk-...   # Windows: set OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# 1. Convert raw documents to JSONL corpus
python scripts/01_data_processing.py

# 2. Build BGE-M3 embeddings and FAISS index
python scripts/02_build_index.py

# 3. Run all experiments → outputs/experiment_results.json
python scripts/03_run_experiment.py
```

Step 3 runs three GPT-4o agents and computes semantic metrics. Runtime is approximately 30–60 minutes depending on corpus size and API throughput. Results are written as raw JSON to `outputs/experiment_results.json`.

---

## Experiment Pipeline (`03_run_experiment.py`)

| Agent / Module | Task | Output field |
|----------------|------|--------------|
| **Structural Agent** | Extract compulsory vs total credits via RAG + GPT-4o | `structural` |
| **Cognitive Agent** | Classify learning objectives by Bloom's Taxonomy (bootstrapped, B=1000) | `cognitive_bloom` |
| **Topic Agent** | Measure retrieval density for 4 emerging topics | `topic_scores` |
| **Semantic Metrics** | Compute ρ (cosine density), Q (modularity), C (clustering) from vectors | `semantic` |

---

## Output Format

`outputs/experiment_results.json` contains:

```json
{
  "structural":       { "NJU": {"total": 160, "compulsory": 104, "F": 0.65}, ... },
  "cognitive_bloom":  { "NJU": {"Remember": 2.1, "Apply": 29.4, "Create": 13.4, ...}, ... },
  "topic_scores":     { "NJU": {"AI Ethics": 174.1, "Climate Change": 68.6, ...}, ... },
  "semantic":         { "NJU": {"rho": 0.62, "rho_ci_half": 0.03, "Q": 0.66, "C": 0.49}, ... }
}
```

---

## License

MIT — see [LICENSE](LICENSE).
