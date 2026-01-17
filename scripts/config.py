import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Path Configurations ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # New: Folder for raw PDFs/Txts
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

# Model Config
MODEL_NAME = 'BAAI/bge-m3'


# --- Visualization Style (Same as before) ---
def set_style():
    sns.set_theme(style="white")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14


COLORS = {
    "Region_A": "#d6404e",
    "Region_B": "#4a7bb7",
    "Region_C": "#55a868",
    "NJU": "#d6404e", "PKU": "#c0392b",
    "Heidelberg": "#4a7bb7", "LMU": "#3498db",
    "Stanford": "#55a868", "MIT": "#2ecc71"
}