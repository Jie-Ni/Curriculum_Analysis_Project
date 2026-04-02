"""
02_build_index.py
-----------------
Reads the JSONL corpus, chunks documents, generates BGE-M3 embeddings,
builds a FAISS inner-product index, and saves both the index and a
metadata pickle.

C3 fix: metadata stores the **full** chunk text under the key "content"
(not a truncated 200-char "text" preview) so that retrieve() in
03_run_experiment.py can pass real context to GPT-4o.
"""

import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config  # seeds are set on import


def load_corpus() -> list[dict]:
    """Load the processed JSONL corpus."""
    print(f"Loading corpus from {config.INPUT_CORPUS} ...")
    documents = []
    if config.INPUT_CORPUS.exists():
        with open(config.INPUT_CORPUS, "r", encoding="utf-8") as f:
            for line in f:
                documents.append(json.loads(line))
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into word-level sliding-window chunks.

    For Chinese text (already segmented by jieba in step 01), the space-
    delimited tokens work correctly here.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_vector_index() -> None:
    """Chunking -> Embedding -> FAISS indexing pipeline."""
    # 1. Load data
    raw_docs = load_corpus()
    if not raw_docs:
        print("Error: No data found. Run 01_data_processing.py first.")
        return

    # 2. Chunk documents
    print("Chunking documents ...")
    processed_chunks: list[str] = []
    metadata: list[dict] = []

    for doc in raw_docs:
        text_chunks = chunk_text(doc["content"])
        for chunk in text_chunks:
            processed_chunks.append(chunk)
            # C3 fix: store FULL chunk text under "content"
            metadata.append(
                {
                    "university": doc["university"],
                    "filename": doc["filename"],
                    "content": chunk,  # full text, NOT truncated
                }
            )

    print(f"Total chunks created: {len(processed_chunks)}")

    # 3. Generate embeddings
    print(f"Loading model: {config.MODEL_NAME} ...")
    model = SentenceTransformer(config.MODEL_NAME, device="cpu")

    print("Generating embeddings (this may take a while) ...")
    embeddings = model.encode(
        processed_chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # 4. Build FAISS index (Inner Product = cosine similarity on normalised vecs)
    print("Building FAISS index ...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))

    # 5. Save artefacts
    print(f"Saving index to {config.INDEX_PATH} ...")
    faiss.write_index(index, str(config.INDEX_PATH))

    print(f"Saving metadata to {config.META_PATH} ...")
    with open(config.META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Indexing complete. {len(processed_chunks)} vectors stored.")


if __name__ == "__main__":
    build_vector_index()
