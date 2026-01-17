import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config


def load_corpus():
    """Loads the processed JSONL corpus."""
    print(f"Loading corpus from {config.INPUT_CORPUS}...")
    documents = []
    if config.INPUT_CORPUS.exists():
        with open(config.INPUT_CORPUS, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits long text into smaller chunks with overlap to preserve context.
    Simple sliding window approach.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def build_vector_index():
    """
    Main pipeline: Chunking -> Embedding -> FAISS Indexing.
    """
    # 1. Load Data
    raw_docs = load_corpus()
    if not raw_docs:
        print("Error: No data found. Run 01_data_processing.py first.")
        return

    # 2. Prepare Chunks
    print("Chunking documents...")
    processed_chunks = []
    metadata = []

    for doc in raw_docs:
        text_chunks = chunk_text(doc['content'])
        for chunk in text_chunks:
            processed_chunks.append(chunk)
            # Store metadata for retrieval later
            metadata.append({
                "university": doc['university'],
                "filename": doc['filename'],
                "text": chunk[:200] + "..."  # Store preview
            })

    print(f"Total chunks created: {len(processed_chunks)}")

    # 3. Generate Embeddings
    print(f"Loading model: {config.MODEL_NAME}...")
    model = SentenceTransformer(config.MODEL_NAME, device='cpu')

    print("Generating embeddings (this may take time)...")
    # Batch encoding for efficiency
    embeddings = model.encode(processed_chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    # 4. Build FAISS Index
    print("Building FAISS index...")
    d = embeddings.shape[1]  # Dimension (e.g., 1024 for BGE-M3)
    index = faiss.IndexFlatIP(d)  # Inner Product (Cosine Similarity because normalized)
    index.add(embeddings)

    # 5. Save Artifacts
    print(f"Saving index to {config.INDEX_PATH}...")
    faiss.write_index(index, str(config.INDEX_PATH))

    print(f"Saving metadata to {config.META_PATH}...")
    with open(config.META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Indexing complete. Ready for analysis.")


if __name__ == "__main__":
    build_vector_index()