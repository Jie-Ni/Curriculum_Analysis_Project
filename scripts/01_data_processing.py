"""
01_data_processing.py
---------------------
Reads raw curriculum documents from data/raw/{University}/ and converts them
into a standardised JSONL corpus at data/processed/corpus.jsonl.

Supported input formats:
  - .txt  (plain text)
  - .pdf  (via pdfplumber, if installed)    [M4]

Chinese text is segmented with jieba (if available) to improve downstream
embedding quality.                           [m4]
"""

import json
import re
import sys
from pathlib import Path

import config  # seeds are set on import


# ── Optional PDF support (M4) ───────────────────────────────────────────────
try:
    import pdfplumber

    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print(
        "[WARN] pdfplumber not installed. PDF files will be skipped. "
        "Install with: pip install pdfplumber"
    )

# ── Optional Chinese segmentation (m4) ──────────────────────────────────────
try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

# Universities known to have Chinese-language curricula
CHINESE_UNIS = {"NJU", "PKU"}


def clean_text(text: str) -> str:
    """Remove HTML tags, collapse whitespace, strip non-printable chars."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = "".join(c for c in text if c.isprintable())
    return text.strip()


def segment_chinese(text: str) -> str:
    """Insert spaces between Chinese tokens using jieba, if available."""
    if HAS_JIEBA:
        return " ".join(jieba.cut(text))
    return text


def read_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
    return "\n".join(pages_text)


def process_raw_data() -> None:
    """
    Walk data/raw/{University}/ folders. Read .txt and .pdf files,
    clean them, and write corpus.jsonl.
    """
    print(f"Processing raw data from {config.RAW_DATA_DIR} ...")

    # If no raw data exists, generate demo data for demonstration
    if not any(config.RAW_DATA_DIR.iterdir()):
        print("No raw data found. Generating synthetic demo data ...")
        generate_demo_data()
        return

    documents = []

    for uni_folder in sorted(config.RAW_DATA_DIR.iterdir()):
        if not uni_folder.is_dir():
            continue
        university_name = uni_folder.name
        print(f"  Processing {university_name} ...")

        for file_path in sorted(uni_folder.iterdir()):
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".txt":
                    raw_content = file_path.read_text(encoding="utf-8")
                elif suffix == ".pdf":
                    if not HAS_PDF:
                        print(f"    Skipping PDF (no pdfplumber): {file_path.name}")
                        continue
                    raw_content = read_pdf(file_path)
                else:
                    continue  # skip unsupported formats

                cleaned = clean_text(raw_content)

                # Chinese segmentation for CN universities
                if university_name in CHINESE_UNIS:
                    cleaned = segment_chinese(cleaned)

                if len(cleaned) > 50:
                    documents.append(
                        {
                            "university": university_name,
                            "filename": file_path.name,
                            "content": cleaned,
                            "length": len(cleaned),
                        }
                    )
            except Exception as e:
                print(f"    Error reading {file_path}: {e}")

    output_path = config.INPUT_CORPUS
    print(f"Saving {len(documents)} documents to {output_path} ...")

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Data processing complete. {len(documents)} documents written.")


def generate_demo_data() -> None:
    """Generate synthetic data when no raw files are present."""
    demo_docs = []
    for uni in config.UNIVERSITIES:
        content = (
            f"This is a sample curriculum document for {uni}. "
            f"It includes core modules on Computer Science and Social Theory. "
            f"Students must complete 120 credits. Assessment includes exams and projects."
        )
        for i in range(50):
            demo_docs.append(
                {
                    "university": uni,
                    "filename": f"demo_{uni}_{i}.txt",
                    "content": content,
                    "length": len(content),
                }
            )

    with open(config.INPUT_CORPUS, "w", encoding="utf-8") as f:
        for doc in demo_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Synthetic demo data generated: {len(demo_docs)} documents.")


if __name__ == "__main__":
    process_raw_data()
