import os
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=800, overlap=150):
    #Simple recursive chunking:split on paragraph breaks, respecting a target size
    paragraphs=[p.strip() for p in text.split('\n\n') if p.strip()]
    chunks=[]
    current=""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current += "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks

def build_compliance_index():
    doc_folder='compliance_docs'
    all_chunks=[]
    all_metadata=[]
    for filename in os.listdir(doc_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(doc_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = chunk_text(text)
            doc_name = filename.replace('.txt', '').replace('_', ' ').title()
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({'source': doc_name, 'filename': filename})
            
            print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding chunks...")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings_norm.astype('float32')

    dimension = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_norm)

    faiss.write_index(index, 'compliance_index.faiss')
    with open('compliance_data.pkl', 'wb') as f:
        pickle.dump({'chunks': all_chunks, 'metadata': all_metadata}, f)

    print(f"\nSaved compliance_index.faiss and compliance_data.pkl ({index.ntotal} vectors)")

if __name__ == "__main__":
    build_compliance_index()