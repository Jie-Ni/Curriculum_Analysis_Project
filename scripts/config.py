"""
config.py
---------
Shared paths, constants, random seeds, and visualization style
for the Curriculum Deep Structure Analysis pipeline.
"""

import os
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# If torch is available, seed it too (used by sentence-transformers)
try:
    import torch

    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
except ImportError:
    pass


# ── Path Configurations ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"

OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, EMBEDDING_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
INPUT_CORPUS = PROCESSED_DIR / "corpus.jsonl"
INDEX_PATH = EMBEDDING_DIR / "edu_index.faiss"
META_PATH = EMBEDDING_DIR / "metadata.pkl"

# ── Model Config ─────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-m3"

# ── OpenAI / GPT-4o Config ──────────────────────────────────────────────────
GPT_MODEL = "gpt-4o"
GPT_TEMPERATURE = 0.0  # C1: deterministic outputs for reproducibility

# ── Experiment Constants ─────────────────────────────────────────────────────
UNIVERSITIES = ["NJU", "PKU", "MIT", "Stanford", "Heidelberg", "LMU"]
REGIONS = {
    "China": ["NJU", "PKU"],
    "USA": ["MIT", "Stanford"],
    "Europe": ["Heidelberg", "LMU"],
}
TOPICS = ["AI Ethics", "Climate Change", "Data Science", "Digital Humanities"]
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

B = 1000  # C1: bootstrap iterations — matches README claim
N = 200  # chunks per institution per bootstrap round
TOP_K = 5  # RAG retrieval depth
SIM_THR = 0.75  # minimum cosine similarity for RAG
KAPPA = 1000  # scaling constant for S_topic


# ── Visualization Style ─────────────────────────────────────────────────────
def set_style():
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 14


COLORS = {
    "Region_A": "#d6404e",
    "Region_B": "#4a7bb7",
    "Region_C": "#55a868",
    "NJU": "#d6404e",
    "PKU": "#c0392b",
    "Heidelberg": "#4a7bb7",
    "LMU": "#3498db",
    "Stanford": "#55a868",
    "MIT": "#2ecc71",
}
