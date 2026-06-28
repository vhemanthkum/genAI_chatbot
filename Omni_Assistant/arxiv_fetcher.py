import requests
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# CS sub-categories to fetch
CS_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE",
    "cs.IR", "cs.HC", "cs.SE", "cs.DB", "cs.CR"
]

NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}


def fetch_papers(category: str, max_results: int = 200, start: int = 0) -> list:
    """Fetch papers from arXiv API for a given CS category."""
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    try:
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Failed to fetch {category}: {e}")
        return []

    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        try:
            paper_id_raw = entry.find("atom:id", NS).text.strip()
            paper_id = paper_id_raw.split("/abs/")[-1]
            title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
            published = entry.find("atom:published", NS).text.strip()[:10]
            authors = [a.find("atom:name", NS).text for a in entry.findall("atom:author", NS)]
            cats_elem = entry.findall("atom:category", NS)
            categories = [c.get("term") for c in cats_elem]

            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors[:5],
                "categories": categories,
                "published": published,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "primary_category": category
            })
        except Exception:
            continue

    return papers


def build_dataset(output_path: str = "arxiv_cs_papers.json",
                  papers_per_category: int = 200) -> list:
    """Fetch papers from all CS categories and save to JSON."""
    all_papers = []
    seen_ids = set()

    for cat in CS_CATEGORIES:
        print(f"  Fetching {papers_per_category} papers from {cat}...")
        papers = fetch_papers(cat, max_results=papers_per_category)
        for p in papers:
            if p["id"] not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p["id"])
        print(f"    -> Got {len(papers)} papers (total unique: {len(all_papers)})")
        time.sleep(3)  # Respect arXiv rate limits

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_papers)} papers to {output_path}")
    return all_papers


def load_dataset(path: str = "arxiv_cs_papers.json") -> list:
    """Load the cached dataset."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    papers = build_dataset()
    print(f"Total papers fetched: {len(papers)}")

import pandas as pd
import urllib.parse

class ArxivFetcher:
    def fetch_papers(self, search_query: str, max_results: int = 3) -> pd.DataFrame:
        """
        Fetches papers matching a general search query and returns a pandas DataFrame.
        """
        query_encoded = urllib.parse.quote(search_query)
        params = {
            "search_query": f"all:{query_encoded}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        try:
            resp = requests.get(ARXIV_API_URL, params=params, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            
            papers = []
            for entry in root.findall("atom:entry", NS):
                title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
                papers.append({"Title": title, "Summary": summary})
                
            return pd.DataFrame(papers)
        except Exception as e:
            print(f"Failed to fetch from arXiv: {e}")
            return pd.DataFrame()
