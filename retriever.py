import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRetriever:
    def __init__(self, index_path, meta_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, k=5):
        q = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)

        out = []
        for score, idx in zip(D[0], I[0]):
            out.append({
                "score": float(score),
                "meta": self.meta[idx],
                "text_snippet": "[snippet not stored]"
            })
        return out
