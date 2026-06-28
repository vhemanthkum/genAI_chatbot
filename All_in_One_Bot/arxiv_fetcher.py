import requests
import xml.etree.ElementTree as ET
import pandas as pd
import urllib.parse

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}

class ArxivFetcher:
    """
    Fetches real-time academic papers from arXiv based on dynamic search queries.
    """
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
                title_elem = entry.find("atom:title", NS)
                summary_elem = entry.find("atom:summary", NS)
                
                if title_elem is not None and summary_elem is not None:
                    title = title_elem.text.strip().replace("\n", " ")
                    summary = summary_elem.text.strip().replace("\n", " ")
                    papers.append({"Title": title, "Summary": summary})
                
            return pd.DataFrame(papers)
        except Exception as e:
            # Silent fallback to avoid crashing the UI pipeline
            return pd.DataFrame()
