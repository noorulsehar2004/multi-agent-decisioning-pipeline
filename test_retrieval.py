import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
index = faiss.read_index('compliance_index.faiss')
with open('compliance_data.pkl', 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
metadata = data['metadata']
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
def search(query, k=3):
    query_embedding = embed_model.encode([query])
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_norm = query_norm.astype('float32')
    scores, indices = index.search(query_norm, k)
    
    for rank, idx in enumerate(indices[0]):
        print(f"[{rank+1}] {metadata[idx]['source']} (score: {scores[0][rank]:.3f})")
        print(f"    {chunks[idx][:200]}")
        print()

print("=== Query: 'what does adverse action notice require' ===")
search("what does adverse action notice require")
print("\n=== Query: 'can age be used in credit decisions' ===")
search("can age be used in credit decisions")