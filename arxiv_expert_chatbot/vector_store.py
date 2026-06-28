import chromadb
from chromadb.utils import embedding_functions
import json
import os


class ArxivVectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "arxiv_cs"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def populate(self, papers: list, force: bool = False):
        """Index paper abstracts into ChromaDB."""
        if force:
            self.reset()
        elif self.count() > 0:
            print(f"Collection already has {self.count()} papers.")
            return

        print(f"Indexing {len(papers)} papers into ChromaDB...")
        BATCH = 500
        for i in range(0, len(papers), BATCH):
            batch = papers[i:i + BATCH]
            docs, metas, ids = [], [], []
            for p in batch:
                # Embed: title + abstract for better search
                doc_text = f"{p['title']}. {p['abstract']}"
                docs.append(doc_text)
                metas.append({
                    "title": p["title"],
                    "authors": ", ".join(p.get("authors", [])[:3]),
                    "categories": ", ".join(p.get("categories", [])),
                    "primary_category": p.get("primary_category", ""),
                    "published": p.get("published", ""),
                    "url": p.get("url", ""),
                    "abstract": p["abstract"][:1000]  # store first 1000 chars
                })
                ids.append(f"paper_{p['id'].replace('/', '_')}")
            self.collection.add(documents=docs, metadatas=metas, ids=ids)
            print(f"  Indexed batch {i // BATCH + 1} ({min(i + BATCH, len(papers))}/{len(papers)})")

        print(f"Done! {self.count()} papers indexed.")

    def search(self, query: str, n: int = 5, category_filter: str = None) -> list:
        """Semantic search with optional category filter."""
        if self.count() == 0:
            return []

        where = None
        if category_filter and category_filter != "All":
            where = {"primary_category": {"$eq": category_filter}}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n, self.count()),
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n, self.count()),
                include=["documents", "metadatas", "distances"]
            )

        papers = []
        if results and results["metadatas"]:
            for i, meta in enumerate(results["metadatas"][0]):
                papers.append({
                    **meta,
                    "distance": results["distances"][0][i],
                    "relevance": max(0, round((1 - results["distances"][0][i] / 2) * 100, 1))
                })
        return papers

    def get_all_metadata(self) -> list:
        """Return all paper metadata for visualization."""
        if self.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        return result.get("metadatas", [])
