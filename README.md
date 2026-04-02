# Curriculum Deep Structure Analysis

Computational comparative analysis of undergraduate curriculum structures across China, USA, and Europe using multilingual embeddings and GPT-4o agents.

Based on the paper: *Deep Structures of Undergraduate Curricula: A Computational Comparative Study*.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Note on Zenodo DOI:** A Zenodo DOI will be minted upon publication. The placeholder link that previously appeared here has been removed.

---

## Overview

This project measures hidden structural differences in university curricula using three computable metrics:

- **Structural Rigidity Index (F)** -- proportion of compulsory credits
- **Semantic Density (rho) & Modularity (Q)** -- cosine similarity and community structure of embedding vectors
- **Topic Responsiveness Score (S)** -- RAG retrieval density for emerging topics (normalised by per-university chunk count)

Additional analyses:

- **Bloom's Taxonomy distribution** with bootstrap confidence intervals
- **Kruskal-Wallis between-region tests** on rho and S

Universities: **NJU, PKU** (China) | **MIT, Stanford** (USA) | **Heidelberg, LMU** (Europe)

---

## Repository Structure

```
CSEdu_fix/
├── scripts/
│   ├── config.py                  # Paths, seeds, constants, style
│   ├── 01_data_processing.py      # Raw .txt/.pdf -> cleaned JSONL corpus
│   ├── 02_build_index.py          # BGE-M3 embeddings -> FAISS index
│   └── 03_run_experiment.py       # GPT-4o agents + semantic metrics + stats
├── data/                          # Created at runtime
│   ├── raw/                       # Place raw files here: raw/{University}/*.txt
│   ├── processed/                 # corpus.jsonl
│   └── embeddings/                # FAISS index + metadata pickle
├── output/                        # experiment_results.json (created at runtime)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd CSEdu_fix

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set your OpenAI API key (required for step 3)
export OPENAI_API_KEY=sk-...   # Windows: set OPENAI_API_KEY=sk-...
```

---

## Data Preparation

Place raw curriculum documents under `data/raw/{University}/`:

```
data/raw/
├── NJU/
│   ├── course1.txt
│   └── syllabus.pdf
├── PKU/
│   ├── ...
├── MIT/
│   ├── ...
├── Stanford/
├── Heidelberg/
└── LMU/
```

Supported formats: `.txt` (always), `.pdf` (requires `pdfplumber`).

If no raw data is present, step 1 generates synthetic demo data.

---

## Usage

```bash
# 1. Convert raw documents to JSONL corpus
python scripts/01_data_processing.py

# 2. Build BGE-M3 embeddings and FAISS index
python scripts/02_build_index.py

# 3. Run all experiments -> output/experiment_results.json
python scripts/03_run_experiment.py
```

Step 3 runs three GPT-4o agents and computes semantic metrics with bootstrap CIs. Runtime is approximately 30-60 minutes depending on corpus size and API throughput.

---

## Reproducibility

All random seeds are set in `config.py` (Python `random`, NumPy, and PyTorch). GPT-4o is called with `temperature=0.0`. Bootstrap uses B=1000 iterations.

---

## Experiment Pipeline (`03_run_experiment.py`)

| Agent / Module | Task | Output field |
|---|---|---|
| **Structural Agent** | Extract compulsory vs total credits via RAG + GPT-4o | `structural` |
| **Cognitive Agent** | Classify learning objectives by Bloom's Taxonomy via RAG + GPT-4o (bootstrapped, B=1000) | `cognitive_bloom` |
| **Topic Agent** | Measure retrieval density for 4 emerging topics (normalised by corpus size) | `topic_scores` |
| **Semantic Metrics** | Compute rho, Q (on symmetrised mutual kNN graph), C with bootstrap percentile CIs | `semantic` |
| **Between-Region Tests** | Kruskal-Wallis H-test on rho and S across China / USA / Europe | `between_region_tests` |

---

## Output Format

`output/experiment_results.json` contains:

```json
{
  "structural": {
    "NJU": {"total": 160, "compulsory": 104, "F": 0.65},
    "...": "..."
  },
  "cognitive_bloom": {
    "NJU": {
      "distribution": {"Remember": 2.1, "Understand": 12.3, "Apply": 29.4, "...": "..."},
      "ci": {"Remember": {"ci_lower": 1.0, "ci_upper": 3.2}, "...": "..."}
    }
  },
  "topic_scores": {
    "NJU": {"AI Ethics": 174.1, "Climate Change": 68.6, "...": "..."}
  },
  "semantic": {
    "NJU": {
      "rho": 0.62, "rho_ci_lower": 0.59, "rho_ci_upper": 0.65,
      "Q": 0.66, "Q_ci_lower": 0.61, "Q_ci_upper": 0.71,
      "C": 0.49, "C_ci_lower": 0.44, "C_ci_upper": 0.54
    }
  },
  "between_region_tests": {
    "rho_kruskal_wallis": {"H": 4.32, "p": 0.115},
    "S_kruskal_wallis": {"H": 3.87, "p": 0.144}
  }
}
```

---

## License

MIT -- see [LICENSE](LICENSE).
