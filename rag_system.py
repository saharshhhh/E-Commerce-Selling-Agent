import numpy as np
from sentence_transformers import SentenceTransformer
from data_loader import load_docs
from typing import List, Dict

class RAGSystem:
    def __init__(self, docs_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunks = load_docs(docs_dir)
        self.texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.model.encode(self.texts, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten() / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results

if __name__ == "__main__":
    rag = RAGSystem("sample_data/docs/")
    query = "What is the return policy for electronics?"
    results = rag.search(query)
    for res in results:
        print(f"Score: {res['score']:.4f} | Source: {res['chunk']['source']}")
        print(f"Text: {res['chunk']['text'][:200]}...\n")
