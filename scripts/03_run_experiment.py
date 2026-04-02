"""
03_run_experiment.py
--------------------
Core experiment pipeline.  Runs three GPT-4o agents against the FAISS index
and computes all primary metrics.  Results are saved as raw JSON to output/.

Agents
------
  Structural Agent : extract compulsory-vs-total credit ratios        -> F-values
  Cognitive Agent  : classify learning objectives by Bloom level      -> distributions
  Topic Agent      : measure retrieval density for emerging topics    -> S-scores

Semantic metrics (rho, Q, C) are computed directly from embedding vectors.

Statistical tests
-----------------
  - Bootstrap CIs (percentile method) for rho, Q, C, and Bloom       [C4, M3]
  - Kruskal-Wallis between-region test on rho and S                   [M6]

Requirements
------------
  OPENAI_API_KEY environment variable must be set.
"""

import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from scipy import stats
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import kneighbors_graph
from community import community_louvain  # python-louvain
import networkx as nx

import config  # seeds are set on import

# ── Validate API key early (m7) ─────────────────────────────────────────────
_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    print(
        "ERROR: OPENAI_API_KEY environment variable is not set.\n"
        "Please set it before running this script:\n"
        "  export OPENAI_API_KEY=sk-...          # Linux / macOS\n"
        "  set OPENAI_API_KEY=sk-...             # Windows cmd\n"
        "  $env:OPENAI_API_KEY = 'sk-...'        # PowerShell"
    )
    sys.exit(1)

client = OpenAI(api_key=_api_key)

# ── Aliases from config ─────────────────────────────────────────────────────
MODEL = config.GPT_MODEL
TEMP = config.GPT_TEMPERATURE  # C1: 0.0 for determinism
TOP_K = config.TOP_K
SIM_THR = config.SIM_THR
UNIS = config.UNIVERSITIES
TOPICS = config.TOPICS
BLOOM = config.BLOOM_LEVELS
REGIONS = config.REGIONS
B = config.B  # 1000 bootstrap iterations
N = config.N
KAPPA = config.KAPPA


# ── Data loading ─────────────────────────────────────────────────────────────
def load_resources():
    with open(config.INPUT_CORPUS, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]

    index = faiss.read_index(str(config.INDEX_PATH))
    with open(config.META_PATH, "rb") as f:
        metadata = pickle.load(f)

    embedder = SentenceTransformer(config.MODEL_NAME, device="cpu")
    return corpus, index, metadata, embedder


def chunks_for(uni: str, corpus: list[dict]) -> list[dict]:
    return [c for c in corpus if uni.lower() in c["university"].lower()]


# ── RAG retrieval (C2 + C3 fix) ─────────────────────────────────────────────
def retrieve(
    query_vec: np.ndarray,
    index,
    metadata: list[dict],
    uni: str,
    k: int = TOP_K,
    threshold: float = SIM_THR,
) -> list[str]:
    """
    Return top-k chunks for *uni* above the similarity threshold.

    C2: Uses the FAISS index for real vector-similarity retrieval.
    C3: Reads metadata["content"] (full chunk text, stored by 02_build_index).
    """
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    D, I = index.search(query_vec.reshape(1, -1).astype(np.float32), k * 10)
    results: list[str] = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if dist < threshold:
            continue
        meta = metadata[idx]
        if uni.lower() not in meta["university"].lower():
            continue
        # C3: use "content" key; fall back to "text" for legacy indices
        text = meta.get("content") or meta.get("text", "")
        if text:
            results.append(text)
        if len(results) >= k:
            break
    return results


def gpt(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMP,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Helpers: bootstrap percentile CI (C4) ────────────────────────────────────
def percentile_ci(samples: list[float], alpha: float = 0.05) -> dict:
    """
    Return mean and (lower, upper) percentile bootstrap CI.

    C4 fix: we report proper asymmetric (lower, upper) bounds,
    not a symmetric half-width.
    """
    arr = np.array(samples)
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return {"mean": float(np.mean(arr)), "ci_lower": round(lower, 6), "ci_upper": round(upper, 6)}


# ── Agent 1: Structural Agent -> F-values ───────────────────────────────────
STRUCT_SYS = (
    "You are an expert curriculum analyst. "
    "Given university curriculum text, extract two numbers: "
    "(1) total graduation credits required, (2) compulsory/mandatory credits. "
    'Reply in JSON only: {"total": <int>, "compulsory": <int>}. '
    "If a value cannot be determined, use null."
)


def run_structural_agent(corpus, index, metadata, embedder) -> dict:
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
        raw = gpt(STRUCT_SYS, f"University: {uni}\n\nCurriculum text:\n{context}")
        try:
            data = json.loads(raw)
            total = data.get("total")
            comp = data.get("compulsory")
            f_val = round(comp / total, 4) if total and comp else None
            results[uni] = {"total": total, "compulsory": comp, "F": f_val}
            print(f"  {uni}: F = {f_val}  (comp={comp}, total={total})")
        except Exception as e:
            print(f"  {uni}: parse error -- {e}\n  raw: {raw[:120]}")
            results[uni] = {"raw": raw}
    return results


# ── Agent 2: Cognitive Agent -> Bloom distributions (C2 fix) ─────────────────
BLOOM_SYS = (
    "You are an expert in Bloom's Taxonomy. "
    "Given a curriculum text excerpt, classify the primary cognitive demand "
    "of each learning objective into exactly one of: "
    "Remember, Understand, Apply, Analyze, Evaluate, Create. "
    "Return a JSON object with those six keys and integer counts, e.g. "
    '{"Remember": 2, "Understand": 5, "Apply": 8, ...}.'
)


def run_cognitive_agent(corpus, index, metadata, embedder) -> dict:
    """
    C2 fix: the Cognitive Agent now uses FAISS-based RAG retrieval
    (via query_vec) to fetch relevant chunks, instead of randomly
    filtering the corpus by university name.
    """
    print("\n[Cognitive Agent]")
    query_vec = embedder.encode(
        ["learning objectives outcomes skills students will be able to"],
        normalize_embeddings=True,
    )[0]

    results = {}
    for uni in UNIS:
        # C2: use RAG retrieval with the learning-objectives query
        chunks = retrieve(query_vec, index, metadata, uni, k=TOP_K * 4, threshold=SIM_THR * 0.9)
        if not chunks:
            # Fallback: use corpus-level chunks
            pool = chunks_for(uni, corpus)
            chunks = [c["content"] for c in pool]

        if not chunks:
            results[uni] = None
            continue

        # Bootstrap: average Bloom distribution over B rounds
        agg = defaultdict(list)
        for _ in range(B):
            sample = random.sample(chunks, min(N, len(chunks)))
            context = "\n\n".join(sample[:8])  # token budget
            raw = gpt(BLOOM_SYS, f"Curriculum text:\n{context}")
            try:
                counts = json.loads(raw)
                total = sum(counts.values()) or 1
                for lvl in BLOOM:
                    agg[lvl].append(counts.get(lvl, 0) / total * 100)
            except Exception:
                pass  # skip failed parse

        # C4: proper percentile CI for Bloom distributions
        dist = {}
        dist_ci = {}
        for lvl in BLOOM:
            if agg[lvl]:
                info = percentile_ci(agg[lvl])
                dist[lvl] = round(info["mean"], 2)
                dist_ci[lvl] = {"ci_lower": info["ci_lower"], "ci_upper": info["ci_upper"]}
            else:
                dist[lvl] = 0.0
                dist_ci[lvl] = {"ci_lower": 0.0, "ci_upper": 0.0}

        results[uni] = {"distribution": dist, "ci": dist_ci}
        top = max(dist, key=dist.get)
        print(f"  {uni}: dominant = {top} ({dist[top]:.1f}%)")
    return results


# ── Agent 3: Topic Responsiveness -> S-scores (M1 fix) ──────────────────────
def run_topic_agent(index, metadata, embedder) -> dict:
    """
    M1 fix: S is normalised by per-university chunk count to remove
    corpus-size confounding.
    """
    print("\n[Topic Agent]")
    uni_ids: dict[str, set[int]] = defaultdict(set)
    for i, m in enumerate(metadata):
        for uni in UNIS:
            if uni.lower() in m["university"].lower():
                uni_ids[uni].add(i)

    results = {}
    for uni in UNIS:
        results[uni] = {}
        n_uni = len(uni_ids[uni])
        if n_uni == 0:
            for topic in TOPICS:
                results[uni][topic] = 0.0
            continue

        for topic in TOPICS:
            q = embedder.encode([topic], normalize_embeddings=True).astype(np.float32)
            D, I = index.search(q, min(500, index.ntotal))
            hits = sum(
                1
                for dist, idx in zip(D[0], I[0])
                if idx in uni_ids[uni] and dist > SIM_THR
            )
            # M1: normalise by per-university chunk count
            s = round(hits / n_uni * KAPPA, 1)
            results[uni][topic] = s
        print(f"  {uni}: {results[uni]}")
    return results


# ── Semantic metrics: rho, Q, C  (C4, C5, M3 fixes) ────────────────────────
def compute_semantic_metrics(index, metadata) -> dict:
    """
    C4: percentile CIs instead of symmetric half-width.
    C5: symmetrise kNN graph (mutual kNN) before computing modularity Q.
    M3: bootstrap CIs for Q and C as well.
    """
    print("\n[Semantic Metrics]")
    all_vecs = index.reconstruct_n(0, index.ntotal)

    uni_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
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

        # Normalise once
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs_normed = vecs / norms

        # ── Bootstrap rho (semantic density) ────────────────────────
        rho_list: list[float] = []
        for _ in range(B):
            idx = np.random.choice(len(vecs_normed), min(N, len(vecs_normed)), replace=False)
            v = vecs_normed[idx]
            sims = (v @ v.T)[np.triu_indices(len(v), k=1)]
            rho_list.append(float(np.mean(sims)))
        rho_info = percentile_ci(rho_list)

        # ── Bootstrap Q (modularity) and C (clustering) ────────────
        q_list: list[float] = []
        c_list: list[float] = []
        sample_size = min(300, len(vecs_normed))
        for _ in range(B):
            sample_idx = np.random.choice(len(vecs_normed), sample_size, replace=False)
            sv = vecs_normed[sample_idx]

            # Build kNN graph
            A = kneighbors_graph(sv, n_neighbors=5, mode="connectivity", include_self=False)

            # C5: symmetrise -> mutual kNN (require both directions)
            A_mutual = A.multiply(A.T)

            G = nx.from_scipy_sparse_array(A_mutual)
            # Remove isolated nodes from mutual kNN
            isolates = list(nx.isolates(G))
            G.remove_nodes_from(isolates)

            if G.number_of_nodes() < 3 or G.number_of_edges() < 1:
                continue

            partition = community_louvain.best_partition(G, random_state=config.GLOBAL_SEED)
            q_list.append(community_louvain.modularity(partition, G))
            c_list.append(nx.average_clustering(G))

        q_info = percentile_ci(q_list) if q_list else {"mean": None, "ci_lower": None, "ci_upper": None}
        c_info = percentile_ci(c_list) if c_list else {"mean": None, "ci_lower": None, "ci_upper": None}

        results[uni] = {
            "rho": round(rho_info["mean"], 4),
            "rho_ci_lower": round(rho_info["ci_lower"], 4),
            "rho_ci_upper": round(rho_info["ci_upper"], 4),
            "Q": round(q_info["mean"], 4) if q_info["mean"] is not None else None,
            "Q_ci_lower": round(q_info["ci_lower"], 4) if q_info["ci_lower"] is not None else None,
            "Q_ci_upper": round(q_info["ci_upper"], 4) if q_info["ci_upper"] is not None else None,
            "C": round(c_info["mean"], 4) if c_info["mean"] is not None else None,
            "C_ci_lower": round(c_info["ci_lower"], 4) if c_info["ci_lower"] is not None else None,
            "C_ci_upper": round(c_info["ci_upper"], 4) if c_info["ci_upper"] is not None else None,
        }
        print(
            f"  {uni}: rho={results[uni]['rho']} "
            f"[{results[uni]['rho_ci_lower']}, {results[uni]['rho_ci_upper']}], "
            f"Q={results[uni]['Q']}, C={results[uni]['C']}"
        )
    return results


# ── Between-region statistical tests (M6) ───────────────────────────────────
def between_region_tests(semantic_results: dict, topic_results: dict) -> dict:
    """
    M6: Kruskal-Wallis H-test across the three regions (China, USA, Europe)
    for rho and aggregated S-scores.
    """
    print("\n[Between-Region Tests]")
    tests = {}

    # -- rho across regions --
    region_rho: dict[str, list[float]] = {}
    for region, unis in REGIONS.items():
        vals = []
        for u in unis:
            if semantic_results.get(u) and semantic_results[u].get("rho") is not None:
                vals.append(semantic_results[u]["rho"])
        region_rho[region] = vals

    groups_rho = [v for v in region_rho.values() if len(v) > 0]
    if len(groups_rho) >= 2 and all(len(g) > 0 for g in groups_rho):
        try:
            stat, p = stats.kruskal(*groups_rho)
            tests["rho_kruskal_wallis"] = {"H": round(stat, 4), "p": round(p, 6)}
            print(f"  rho Kruskal-Wallis: H={stat:.4f}, p={p:.6f}")
        except ValueError as e:
            tests["rho_kruskal_wallis"] = {"error": str(e)}
            print(f"  rho Kruskal-Wallis skipped: {e}")
    else:
        tests["rho_kruskal_wallis"] = {"error": "insufficient data (need >= 2 groups)"}
        print("  rho Kruskal-Wallis skipped: insufficient data")

    # -- mean S across regions --
    region_s: dict[str, list[float]] = {}
    for region, unis in REGIONS.items():
        vals = []
        for u in unis:
            if topic_results.get(u):
                mean_s = np.mean(list(topic_results[u].values()))
                vals.append(mean_s)
        region_s[region] = vals

    groups_s = [v for v in region_s.values() if len(v) > 0]
    if len(groups_s) >= 2 and all(len(g) > 0 for g in groups_s):
        try:
            stat, p = stats.kruskal(*groups_s)
            tests["S_kruskal_wallis"] = {"H": round(stat, 4), "p": round(p, 6)}
            print(f"  S Kruskal-Wallis: H={stat:.4f}, p={p:.6f}")
        except ValueError as e:
            tests["S_kruskal_wallis"] = {"error": str(e)}
            print(f"  S Kruskal-Wallis skipped: {e}")
    else:
        tests["S_kruskal_wallis"] = {"error": "insufficient data"}
        print("  S Kruskal-Wallis skipped: insufficient data")

    return tests


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    # M2: re-seed at entry to guarantee reproducibility even if imported
    random.seed(config.GLOBAL_SEED)
    np.random.seed(config.GLOBAL_SEED)

    corpus, index, metadata, embedder = load_resources()

    structural = run_structural_agent(corpus, index, metadata, embedder)
    cognitive = run_cognitive_agent(corpus, index, metadata, embedder)
    topic = run_topic_agent(index, metadata, embedder)
    semantic = compute_semantic_metrics(index, metadata)
    region_tests = between_region_tests(semantic, topic)

    out = {
        "structural": structural,
        "cognitive_bloom": cognitive,
        "topic_scores": topic,
        "semantic": semantic,
        "between_region_tests": region_tests,
    }

    # Output path consistency: use OUTPUT_DIR from config
    out_path = config.OUTPUT_DIR / "experiment_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
