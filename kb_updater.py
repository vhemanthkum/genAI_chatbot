"""
kb_updater.py
=============
Dynamic Knowledge Base updater for the Multi-Modal Assistant.

Reads knowledge_base/sources.json, fetches each enabled source (local file
or URL), chunks the text, and upserts embeddings into ChromaDB.

Usage:
  python kb_updater.py              # one-shot full re-index
  start_background_scheduler()        # periodic background updates (used by app.py)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Callable, List, Optional

import requests
import schedule
from bs4 import BeautifulSoup

import vector_store

SOURCES_FILE = os.getenv("KB_SOURCES_FILE", "knowledge_base/sources.json")
CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "50"))
KB_UPDATE_INTERVAL = int(os.getenv("KB_UPDATE_INTERVAL_MINUTES", "30"))
KB_SIMILARITY_CUTOFF = float(os.getenv("KB_SIMILARITY_CUTOFF", "0.55"))

_last_updated: Optional[datetime] = None
_on_update_callback: Optional[Callable[[datetime], None]] = None

_UNICODE_REPLACEMENTS = {
    "\u2192": "->",
    "\u2190": "<-",
    "\u2026": "...",
    "\u2014": "--",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "*",
}


def sanitize_text(text: str) -> str:
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("ascii", errors="ignore").decode("ascii")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    sub = para[i : i + chunk_size]
                    if sub:
                        chunks.append(sub)
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) >= 20]


def fetch_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def fetch_url(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "MultimodalAssistant-KBUpdater/1.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def run_update(full_reindex: bool = False) -> dict:
    """Ingest all enabled sources into the vector store."""
    global _last_updated

    print(f"\n[KB Updater] Starting update at {datetime.now():%Y-%m-%d %H:%M:%S}")

    if not os.path.exists(SOURCES_FILE):
        print(f"[KB Updater] WARNING: {SOURCES_FILE} not found. Skipping.")
        return {"sources_processed": 0, "chunks_added": 0, "errors": []}

    with open(SOURCES_FILE, encoding="utf-8") as f:
        sources = json.load(f)

    if full_reindex:
        vector_store.clear()
        print("[KB Updater] Full re-index: cleared old vectors.")

    stats = {"sources_processed": 0, "chunks_added": 0, "errors": []}

    for source in sources:
        if not source.get("enabled", True):
            continue

        src_type = source.get("type", "file")
        label = source.get("description", source.get("path", source.get("url", "?")))

        try:
            print(f"[KB Updater] Fetching [{src_type}] {label} ...")

            if src_type == "file":
                raw_text = sanitize_text(fetch_file(source["path"]))
            elif src_type == "url":
                raw_text = sanitize_text(fetch_url(source["url"]))
            else:
                print(f"[KB Updater] Unknown source type: {src_type}. Skipping.")
                continue

            chunks = chunk_text(raw_text)
            added = vector_store.add_documents(chunks, source=label)
            print(f"[KB Updater]   -> {added} chunk(s) upserted.")
            stats["sources_processed"] += 1
            stats["chunks_added"] += added

        except Exception as exc:
            msg = f"[KB Updater] ERROR on '{label}': {exc}"
            print(msg)
            stats["errors"].append(msg)

    _last_updated = datetime.now()

    if _on_update_callback:
        try:
            _on_update_callback(_last_updated)
        except Exception:
            pass

    total = vector_store.count()
    print(f"[KB Updater] Done. Total vectors in store: {total}")
    print(f"[KB Updater] Stats: {stats}")
    return stats


def _scheduler_loop():
    schedule.every(KB_UPDATE_INTERVAL).minutes.do(run_update)
    while True:
        schedule.run_pending()
        time.sleep(30)


def start_background_scheduler(
    on_update: Optional[Callable[[datetime], None]] = None,
) -> threading.Thread:
    """Launch a daemon thread that refreshes the KB on a fixed interval."""
    global _on_update_callback
    _on_update_callback = on_update

    thread = threading.Thread(target=_initial_then_schedule, daemon=True)
    thread.start()
    return thread


def _initial_then_schedule():
    run_update()
    _scheduler_loop()


def get_last_updated() -> Optional[datetime]:
    return _last_updated


def search_kb(
    query_text: str,
    n_results: int = 3,
    max_distance: float = KB_SIMILARITY_CUTOFF,
) -> Optional[str]:
    """Return the best matching KB chunk if similarity is above threshold."""
    results = vector_store.query(query_text, n_results=n_results)
    if not results:
        return None

    best = results[0]
    if best["distance"] <= max_distance:
        return best["text"]
    return None


def search_kb_multi(
    query_text: str,
    n_results: int = 3,
    max_distance: float = KB_SIMILARITY_CUTOFF,
) -> List[dict]:
    """Return all KB hits within the similarity threshold."""
    results = vector_store.query(query_text, n_results=n_results)
    return [r for r in results if r["distance"] <= max_distance]


if __name__ == "__main__":
    stats = run_update(full_reindex=True)
    print("\n[KB Updater] Standalone run complete.")
    print(f"  Sources processed : {stats['sources_processed']}")
    print(f"  Chunks added      : {stats['chunks_added']}")
    if stats["errors"]:
        print("  Errors:")
        for err in stats["errors"]:
            print(f"    - {err}")
    else:
        print("  No errors.")
