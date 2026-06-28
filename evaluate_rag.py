"""
evaluate_rag.py
===============
Compare baseline (no RAG) vs RAG-enhanced retrieval on fixed test queries.

Generates visualizations in visualizations/:
  - rag_retrieval_accuracy.png
  - baseline_vs_rag_comparison.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kb_updater
import vector_store

VIS_DIR = Path("visualizations")
TEST_QUERIES_FILE = Path("knowledge_base/eval_queries.json")
SIMILARITY_CUTOFF = float(os.getenv("KB_SIMILARITY_CUTOFF", "0.55"))


def load_test_queries() -> list[dict]:
    with open(TEST_QUERIES_FILE, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(queries: list[dict]) -> dict:
    hits = 0
    distances = []
    per_query = []

    for item in queries:
        results = vector_store.query(item["query"], n_results=1)
        if not results:
            per_query.append({"query": item["query"], "hit": False, "distance": None})
            continue

        best = results[0]
        source_match = item["expected_source"] in best["source"]
        within_cutoff = best["distance"] <= SIMILARITY_CUTOFF
        success = source_match and within_cutoff
        if success:
            hits += 1
        distances.append(best["distance"])
        per_query.append(
            {
                "query": item["query"],
                "hit": success,
                "distance": best["distance"],
                "source": best["source"],
            }
        )

    accuracy = hits / len(queries) if queries else 0.0
    return {
        "accuracy": accuracy,
        "hits": hits,
        "total": len(queries),
        "distances": distances,
        "per_query": per_query,
    }


def plot_retrieval_accuracy(eval_result: dict) -> None:
    VIS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Retrieved", "Missed"]
    sizes = [eval_result["hits"], eval_result["total"] - eval_result["hits"]]
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].pie(
        sizes,
        labels=labels,
        autopct="%1.0f%%",
        colors=colors,
        startangle=90,
    )
    axes[0].set_title(f"RAG Retrieval Accuracy (cutoff={SIMILARITY_CUTOFF})")

    distances = eval_result["distances"]
    if distances:
        axes[1].hist(distances, bins=8, color="#3498db", edgecolor="white")
        axes[1].axvline(
            SIMILARITY_CUTOFF,
            color="#e67e22",
            linestyle="--",
            label=f"Cutoff ({SIMILARITY_CUTOFF})",
        )
        axes[1].set_xlabel("Cosine distance (lower = more similar)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Top-1 Retrieval Distance Distribution")
        axes[1].legend()

    plt.tight_layout()
    out = VIS_DIR / "rag_retrieval_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Eval] Saved {out}")


def _load_source_chunks() -> dict[str, list[str]]:
    """Load raw text chunks per source label without embeddings."""
    import kb_updater as kb

    chunks_by_source: dict[str, list[str]] = {}
    if not TEST_QUERIES_FILE.parent.joinpath("sources.json").exists():
        return chunks_by_source

    with open(TEST_QUERIES_FILE.parent / "sources.json", encoding="utf-8") as f:
        sources = json.load(f)

    for source in sources:
        if not source.get("enabled", True):
            continue
        label = source.get("description", "?")
        try:
            if source.get("type", "file") == "file":
                raw = kb.sanitize_text(kb.fetch_file(source["path"]))
            elif source.get("type") == "url":
                raw = kb.sanitize_text(kb.fetch_url(source["url"]))
            else:
                continue
            chunks_by_source[label] = kb.chunk_text(raw)
        except Exception:
            continue
    return chunks_by_source


def _baseline_keyword_hit(query: str, expected_source: str, chunks_by_source: dict[str, list[str]]) -> bool:
    """Naive baseline: >=50% of query tokens must appear literally in one chunk."""
    stopwords = {
        "what", "how", "does", "the", "a", "an", "is", "are", "do", "i", "in", "to",
        "for", "of", "and", "me", "about", "can", "when", "this", "at", "on", "my",
        "you", "your", "its", "it", "or", "be", "with", "from", "that", "will", "get",
        "after", "work", "reach", "explain", "tell", "does", "have", "there", "any",
    }
    tokens = [t for t in query.lower().split() if t not in stopwords and len(t) > 2]
    if not tokens:
        return False

    matched_source = None
    for label, chunks in chunks_by_source.items():
        if expected_source in label:
            matched_source = chunks
            break
    if not matched_source:
        return False

    threshold = max(1, int(len(tokens) * 0.5 + 0.5))
    for chunk in matched_source:
        text_lower = chunk.lower()
        overlap = sum(1 for t in tokens if t in text_lower)
        if overlap >= threshold:
            return True
    return False


def plot_baseline_vs_rag(eval_result: dict, queries: list[dict], chunks_by_source: dict[str, list[str]]) -> None:
    """Compare naive keyword overlap vs semantic RAG hit rates."""
    VIS_DIR.mkdir(exist_ok=True)

    keyword_hits = sum(
        int(_baseline_keyword_hit(q["query"], q["expected_source"], chunks_by_source))
        for q in queries
    )
    rag_hits = eval_result["hits"]
    total = eval_result["total"]

    methods = ["Baseline\n(keyword heuristic)", "RAG\n(semantic vector search)"]
    accuracies = [
        keyword_hits / total if total else 0,
        rag_hits / total if total else 0,
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=["#95a5a6", "#2980b9"], width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Retrieval success rate")
    ax.set_title("Baseline vs RAG-Enhanced Knowledge Retrieval")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.0%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    out = VIS_DIR / "baseline_vs_rag_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Eval] Saved {out}")


def main():
    print("[Eval] Seeding knowledge base...")
    kb_updater.run_update(full_reindex=True)

    queries = load_test_queries()
    print(f"[Eval] Running retrieval evaluation on {len(queries)} queries...")
    result = evaluate_retrieval(queries)
    chunks_by_source = _load_source_chunks()

    print(f"[Eval] RAG accuracy: {result['accuracy']:.0%} ({result['hits']}/{result['total']})")
    if result["distances"]:
        print(f"[Eval] Mean distance: {np.mean(result['distances']):.3f}")

    plot_retrieval_accuracy(result)
    plot_baseline_vs_rag(result, queries, chunks_by_source)

    baseline_hits = sum(
        int(_baseline_keyword_hit(q["query"], q["expected_source"], chunks_by_source))
        for q in queries
    )
    print(f"[Eval] Baseline keyword accuracy: {baseline_hits / len(queries):.0%} ({baseline_hits}/{len(queries)})")
    print("[Eval] Done.")


if __name__ == "__main__":
    main()
