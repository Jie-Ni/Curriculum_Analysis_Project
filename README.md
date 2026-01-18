# The Deep Structure of Curriculum: A Computational Comparative Analysis

**A Computational Comparative Analysis of Chinese, American, and European Undergraduate Education using Large Language Models**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18284324.svg)](https://doi.org/10.5281/zenodo.18284324)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge)]()

</div>

---

## 📖 Abstract

While higher education globalization has led to superficial convergence in curriculum goals, the underlying pedagogical structures remain distinct. This study proposes a **"Computational Education Sociology"** framework to quantify curriculum rigidity, semantic density, and cognitive objectives.

Leveraging the **BGE-M3 multilingual embedding model** and **RAG (Retrieval-Augmented Generation)**, we analyzed core curriculum documents from six top-tier universities (**NJU, PKU, Stanford, MIT, Heidelberg, LMU**). 

Our findings reveal a **"Goal-Structure Paradox"**: while cognitive objectives are converging globally, structural typologies diverge significantly between the **"Deep Well"** model (Strong Framing, typical in China/Europe) and the **"Broad Base"** model (Weak Framing, typical in the USA).

---

## 💾 Data Availability

Due to repository size limits, the raw curriculum corpus and processed embedding indices are hosted on Zenodo.

> **Dataset DOI:** [10.5281/zenodo.18284324](https://doi.org/10.5281/zenodo.18284324)

Please download the dataset and extract the contents into the `data/` directory before running the analysis pipeline.

---

## 📂 Repository Structure

```text
Curriculum_Analysis/
├── data/
│   ├── raw/                   # Raw text files (Extract Zenodo data here)
│   ├── processed/             # Cleaned JSONL corpus
│   └── embeddings/            # FAISS vector indices and metadata
├── output/
│   ├── figures/               # Generated high-res figures (Main Fig 2, 3, 4)
│   └── tables/                # Statistical tables (CSV)
├── scripts/
│   ├── config.py              # Global configuration and constants
│   ├── 01_data_processing.py  # Data cleaning pipeline
│   ├── 02_build_index.py      # Vector embedding and indexing
│   ├── 03_plot_main_figures.py# Visualization generation (Main Figures)
│   └── 04_generate_tables.py  # Statistical table generation
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
cd Curriculum-Analysis
```

### 2. Create a virtual environment (Recommended)
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Pipeline

To reproduce the results presented in the paper, execute the scripts in the following order. All scripts should be run from the root directory.

### Step 1: Data Preparation
Ensure you have downloaded the data from Zenodo (DOI: 10.5281/zenodo.18284324) and placed it in `data/raw`.

Run the processing script to convert documents into standardized JSONL format:
```bash
python scripts/01_data_processing.py
```

### Step 2: Vector Index Construction
Generates semantic embeddings using `BAAI/bge-m3` and builds the FAISS retrieval index.
```bash
python scripts/02_build_index.py
```

### Step 3: Visualization (Main Figures)
Generates the three core composite figures used in the manuscript (Figures 2, 3, and 4), including the Topological Network and AI Diagnosis charts.
```bash
python scripts/03_plot_main_figures.py
```
* **Output:** High-resolution TIFF files will be saved in `output/figures/`.

### Step 4: Statistical Tables
Generates the supplementary data tables (Corpus stats, Bloom's taxonomy distribution, Network metrics, etc.).
```bash
python scripts/04_generate_tables.py
```
* **Output:** CSV files will be saved in `output/tables/`.

---

## 📊 Methodology & Key Metrics

We operationalized **Basil Bernstein’s code theory** into computable metrics.

| Metric | Definition | Key Finding |
| :--- | :--- | :--- |
| **Bernstein's F-Value** (Rigidity) | Ratio of compulsory credits to total credits. | **LMU/Heidelberg** (>0.75) > **NJU/PKU** (>0.58) > **Stanford** (0.38) |
| **Semantic Density** | Measured via pairwise Cosine Similarity of curriculum vectors. | **NJU** exhibits high internal cohesion ("Deep Well"), while **Stanford** shows high variance ("Broad Base"). |
| **Cognitive Depth** | Quantified using a multilingual lexicon based on Bloom's Taxonomy. | Cognitive isomorphism observed across regions, with high emphasis on **"Create"** and **"Analyze"**. |
| **AI Responsiveness** | Normalized density of emerging topics (e.g., "AI Ethics") within the corpus. | Varies by institution type and region. |

---



---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
