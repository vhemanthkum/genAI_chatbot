import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

class MedicalKnowledgeBase:
    def __init__(self, persist_directory="./chroma_db", collection_name="medquad_qa"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Using a lightweight sentence-transformer model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def reset_collection(self):
        """Deletes and recreates the collection using ChromaDB API (no file deletion needed)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        print("Collection reset successfully.")
        
    def populate(self, csv_path="medquad_subset.csv", force_reload=False):
        """Populates the vector DB. Use force_reload=True to rebuild from scratch."""
        if force_reload:
            self.reset_collection()
        elif self.collection.count() > 0:
            print(f"Collection already contains {self.collection.count()} items.")
            return

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return
            
        print("Loading data into ChromaDB. This might take a minute...")
        df = pd.read_csv(csv_path)
        
        # We will embed the questions, and return the answers as metadata/documents
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            documents.append(str(row['question']))
            metadatas.append({
                "answer": str(row['answer']),
                "source": str(row['source_file'])
            })
            ids.append(f"qa_{idx}")
            
            # Batch insert every 1000 items to avoid memory issues
            if len(documents) >= 1000:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents, metadatas, ids = [], [], []
                
        # Insert remaining
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        print(f"Successfully populated DB with {self.collection.count()} QA pairs.")
        
    def retrieve_answer(self, query, n_results=3, distance_threshold=1.2):
        """Retrieves top n relevant QA pairs for a query. Returns empty if best match is too far."""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_data = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                dist = results['distances'][0][i]
                # Only include results within the relevance threshold
                if dist <= distance_threshold:
                    retrieved_data.append({
                        "question": results['documents'][0][i],
                        "answer": results['metadatas'][0][i]['answer'],
                        "source": results['metadatas'][0][i]['source'],
                        "distance": dist,
                        "confidence": max(0, round((1 - dist / distance_threshold) * 100, 1))
                    })
        return retrieved_data

if __name__ == "__main__":
    # Test initialization and population
    kb = MedicalKnowledgeBase()
    kb.populate()
    
    # Test retrieval
    res = kb.retrieve_answer("What are the symptoms of Glaucoma?")
    for r in res:
        print(f"Q: {r['question']}")
        print(f"Source: {r['source']}")
        print("---")
