"""
Standalone script: fetch papers from arXiv API and build ChromaDB index.
Run once before launching the Streamlit app:
    python build_kb.py
"""
import os
import sys
import data_fetcher
from vector_store import ArxivVectorStore

DATA_PATH = "arxiv_cs_papers.json"
PAPERS_PER_CAT = 150  # 150 x 10 categories = ~1500 papers


def build(force: bool = False):
    # Step 1: Fetch papers
    if force or not os.path.exists(DATA_PATH):
        print("Step 1/2: Fetching CS papers from arXiv API...")
        print("  (This will take ~2 minutes due to rate limiting)")
        papers = data_fetcher.build_dataset(DATA_PATH, papers_per_category=PAPERS_PER_CAT)
    else:
        print(f"Step 1/2: Loading cached papers from {DATA_PATH}...")
        papers = data_fetcher.load_dataset(DATA_PATH)
        print(f"  -> Loaded {len(papers)} papers")

    if not papers:
        print("ERROR: No papers loaded.")
        sys.exit(1)

    # Step 2: Build vector index
    print("\nStep 2/2: Building ChromaDB vector index...")
    store = ArxivVectorStore()
    store.populate(papers, force=force)
    print(f"  -> Done! {store.count()} papers indexed.")


if __name__ == "__main__":
    force = "--force" in sys.argv or not os.path.exists(DATA_PATH)
    build(force=force)
