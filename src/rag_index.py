import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("/home/shivsharan/nl2shell-project/data/faiss_train.index")
META_PATH = Path("/home/shivsharan/nl2shell-project/data/faiss_train_meta.json")

class RAGRetriever:
    def __init__(self, data_path: str = "/home/shivsharan/nl2shell-project/data/train.jsonl"):
        print("Initializing SentenceTransformer on CPU...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.data_path = Path(data_path)
        self.examples = []
        self.index = None
        self._load_or_build()

    def _load_or_build(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            print("Loading cached FAISS index...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.examples = json.load(f)
            return

        print("Building FAISS index from training data...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.examples.append({"nl": item["nl"], "bash": item["bash"]})

        corpus = [ex["nl"] for ex in self.examples]
        embeddings = self.encoder.encode(corpus, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) # Inner product with normalized vectors = cosine similarity
        self.index.add(embeddings.astype(np.float32))

        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.examples, f)
        print(f"Index built with {len(self.examples)} vectors.")

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.examples):
                results.append(self.examples[idx])
        return results

if __name__ == "__main__":
    retriever = RAGRetriever()
    res = retriever.retrieve("count lines in file.txt", top_k=5)
    print("Test retrieval:")
    for r in res:
        print("  ->", r)

