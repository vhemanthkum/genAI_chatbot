"""
Standalone script to build the MedQuAD vector database.
Run this ONCE before launching the Streamlit app:
    python build_kb.py
"""
import os
import sys
import shutil
import data_ingestion
from vector_store import MedicalKnowledgeBase

def build(force=True):
    csv_path = "medquad_subset.csv"
    chroma_path = "./chroma_db"

    # Step 1: Parse all MedQuAD XML files
    if force or not os.path.exists(csv_path):
        print("Step 1/2: Parsing MedQuAD XML files...")
        df = data_ingestion.parse_medquad_data("MedQuAD", subset=None)
        if len(df) == 0:
            print("ERROR: No QA pairs found. Check MedQuAD directory.")
            sys.exit(1)
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved {len(df):,} QA pairs to {csv_path}")
    else:
        print(f"Step 1/2: Skipping parse ('{csv_path}' already exists).")

    # Step 2: Wipe old DB and rebuild
    print("Step 2/2: Building vector database...")
    if os.path.exists(chroma_path):
        print(f"  -> Deleting old database at {chroma_path}")
        shutil.rmtree(chroma_path)

    kb = MedicalKnowledgeBase()
    kb.populate(csv_path)
    print(f"  -> Done! {kb.collection.count():,} entries in the database.")

if __name__ == "__main__":
    build(force="--force" in sys.argv or not os.path.exists("medquad_subset.csv"))
