"""
03_run_experiment.py
--------------------
Core experiment pipeline. Runs three GPT-4o agents against the FAISS index
and computes all primary metrics. Results are saved as raw JSON to outputs/.

Agents:
  - Structural Agent  : extract compulsory-vs-total credit ratios  → F-values
  - Cognitive Agent   : classify learning objectives by Bloom level → distributions
  - Topic Agent       : measure retrieval density for emerging topics → S-scores

Semantic metrics (rho, Q, C) are computed directly from the embedding vectors.

Requirements: OPENAI_API_KEY env variable must be set.
"""

import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import kneighbors_graph
from community import community_louvain  # python-louvain
import networkx as nx

import config

# ── Setup ─────────────────────────────────────────────────────────────────────
client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL   = "gpt-4o"
TEMP    = 0.1                # deterministic reasoning
TOP_K   = 5                  # RAG retrieval depth
SIM_THR = 0.75               # minimum cosine similarity for RAG

UNIS    = ["NJU", "PKU", "MIT", "Stanford", "Heidelberg", "LMU"]
TOPICS  = ["AI Ethics", "Climate Change", "Data Science", "Digital Humanities"]
BLOOM   = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

B       = 100      # bootstrap iterations (use 1000 in production)
N       = 200      # chunks per institution per bootstrap round
KAPPA   = 1000     # scaling constant for S_topic


# ── Data loading ──────────────────────────────────────────────────────────────
def load_resources():
    with open(config.INPUT_CORPUS, "r", encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f]

    index    = faiss.read_index(str(config.INDEX_PATH))
    with open(config.META_PATH, "rb") as f:
        metadata = pickle.load(f)

    embedder = SentenceTransformer(config.MODEL_NAME, device="cpu")
    return corpus, index, metadata, embedder


def chunks_for(uni, corpus):
    return [c for c in corpus
            if uni.lower() in c["university"].lower()]


# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve(query_vec, index, metadata, uni, k=TOP_K, threshold=SIM_THR):
    """Return top-k chunks for a given university above the similarity threshold."""
    query_vec = query_vec / np.linalg.norm(query_vec)
    D, I = index.search(query_vec.reshape(1, -1), k * 10)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if dist < threshold:
            continue
        if uni.lower() not in metadata[idx]["university"].lower():
            continue
        results.append(metadata[idx]["content"])
        if len(results) >= k:
            break
    return results


def gpt(system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMP,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Agent 1: Structural Agent → F-values ─────────────────────────────────────
STRUCT_SYS = (
    "You are an expert curriculum analyst. "
    "Given university curriculum text, extract two numbers: "
    "(1) total graduation credits required, (2) compulsory/mandatory credits. "
    "Reply in JSON only: {\"total\": <int>, \"compulsory\": <int>}. "
    "If a value cannot be determined, use null."
)

def run_structural_agent(corpus, index, metadata, embedder):
    print("\n[Structural Agent]")
    query = embedder.encode(
        ["compulsory credits mandatory requirements graduation total"],
        normalize_embeddings=True,
    )[0]

    results = {}
    for uni in UNIS:
        chunks = retrieve(query, index, metadata, uni)
        if not chunks:
            print(f"  {uni}: no relevant chunks found")
            results[uni] = None
            continue
        context = "\n---\n".join(chunks)
        raw = gpt(STRUCT_SYS,
                  f"University: {uni}\n\nCurriculum text:\n{context}")
        try:
            data = json.loads(raw)
            total = data.get("total")
            comp  = data.get("compulsory")
            f_val = round(comp / total, 4) if total and comp else None
            results[uni] = {"total": total, "compulsory": comp, "F": f_val}
            print(f"  {uni}: F = {f_val}  (comp={comp}, total={total})")
        except Exception as e:
            print(f"  {uni}: parse error — {e}\n  raw: {raw[:120]}")
            results[uni] = {"raw": raw}
    return results


# ── Agent 2: Cognitive Agent → Bloom distributions ───────────────────────────
BLOOM_SYS = (
    "You are an expert in Bloom's Taxonomy. "
    "Given a curriculum text excerpt, classify the primary cognitive demand "
    "of each learning objective into exactly one of: "
    "Remember, Understand, Apply, Analyze, Evaluate, Create. "
    "Return a JSON object with those six keys and integer counts, e.g. "
    "{\"Remember\": 2, \"Understand\": 5, \"Apply\": 8, ...}."
)

def run_cognitive_agent(corpus, embedder):
    print("\n[Cognitive Agent]")
    query_vec = embedder.encode(
        ["learning objectives outcomes skills students will be able to"],
        normalize_embeddings=True,
    )[0]

    results = {}
    for uni in UNIS:
        pool = chunks_for(uni, corpus)
        if not pool:
            results[uni] = None
            continue

        # Bootstrap: average Bloom distribution over B rounds
        agg = defaultdict(list)
        for _ in range(B):
            sample = random.sample(pool, min(N, len(pool)))
            context = "\n\n".join(c["content"] for c in sample[:8])  # token budget
            raw = gpt(BLOOM_SYS, f"Curriculum text:\n{context}")
            try:
                counts = json.loads(raw)
                total  = sum(counts.values()) or 1
                for lvl in BLOOM:
                    agg[lvl].append(counts.get(lvl, 0) / total * 100)
            except Exception:
                pass  # skip failed parse

        dist = {lvl: round(float(np.mean(agg[lvl])), 2) for lvl in BLOOM}
        results[uni] = dist
        top = max(dist, key=dist.get)
        print(f"  {uni}: dominant = {top} ({dist[top]:.1f}%)")
    return results


# ── Agent 3: Topic Responsiveness → S-scores ─────────────────────────────────
def run_topic_agent(index, metadata, embedder):
    print("\n[Topic Agent]")
    uni_ids = defaultdict(set)
    for i, m in enumerate(metadata):
        for uni in UNIS:
            if uni.lower() in m["university"].lower():
                uni_ids[uni].add(i)

    results = {}
    for uni in UNIS:
        results[uni] = {}
        total = len(uni_ids[uni]) or 1
        for topic in TOPICS:
            q = embedder.encode([topic], normalize_embeddings=True)
            D, I = index.search(q, 500)
            hits = sum(
                1 for dist, idx in zip(D[0], I[0])
                if idx in uni_ids[uni] and dist > SIM_THR
            )
            s = round(hits / total * KAPPA, 1)
            results[uni][topic] = s
        print(f"  {uni}: {results[uni]}")
    return results


# ── Semantic metrics: rho, Q, C ───────────────────────────────────────────────
def compute_semantic_metrics(index, metadata):
    print("\n[Semantic Metrics]")
    all_vecs = index.reconstruct_n(0, index.ntotal)

    uni_vecs = defaultdict(list)
    for i, m in enumerate(metadata):
        for uni in UNIS:
            if uni.lower() in m["university"].lower():
                uni_vecs[uni].append(all_vecs[i])

    results = {}
    for uni in UNIS:
        vecs = np.array(uni_vecs[uni])
        if len(vecs) < 5:
            results[uni] = None
            continue

        # Bootstrap density (rho)
        rho_list = []
        for _ in range(B):
            idx = np.random.choice(len(vecs), min(N, len(vecs)), replace=False)
            v   = vecs[idx]
            v   = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            sims = (v @ v.T)[np.triu_indices(len(v), k=1)]
            rho_list.append(float(np.mean(sims)))
        rho = round(float(np.mean(rho_list)), 4)
        rho_ci = round(float(np.percentile(rho_list, 97.5) -
                              np.percentile(rho_list, 2.5)) / 2, 4)

        # Modularity Q and Clustering C on kNN graph
        sample_idx = np.random.choice(len(vecs), min(300, len(vecs)), replace=False)
        sv = vecs[sample_idx]
        sv = sv / (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-9)
        A  = kneighbors_graph(sv, n_neighbors=5, mode="connectivity",
                              include_self=False)
        G  = nx.from_scipy_sparse_array(A)
        partition = community_louvain.best_partition(G)
        Q = round(community_louvain.modularity(partition, G), 4)
        C = round(nx.average_clustering(G), 4)

        results[uni] = {"rho": rho, "rho_ci_half": rho_ci, "Q": Q, "C": C}
        print(f"  {uni}: rho={rho}, Q={Q}, C={C}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    corpus, index, metadata, embedder = load_resources()

    out = {
        "structural":     run_structural_agent(corpus, index, metadata, embedder),
        "cognitive_bloom": run_cognitive_agent(corpus, embedder),
        "topic_scores":   run_topic_agent(index, metadata, embedder),
        "semantic":       compute_semantic_metrics(index, metadata),
    }

    out_dir = Path(config.FIGURES_DIR).parent
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
