import math
from collections import Counter

def preprocess_documents(documents):
    vocab = set()
    doc_term_freqs = []
    
    for doc in documents:
        terms = doc.lower().split()
        term_freqs = Counter(terms)
        doc_term_freqs.append(term_freqs)
        vocab.update(terms)
    
    return sorted(vocab), doc_term_freqs

def compute_idf(vocab, doc_term_freqs):
    N = len(doc_term_freqs)
    idf = {}
    
    for term in vocab:
        df = sum(1 for doc_freqs in doc_term_freqs if term in doc_freqs)
        idf[term] = math.log(N / (df + 1))  # Add 1 smoothing
        
    return idf

def compute_tf_idf_vector(terms, vocab, term_freqs, idf):
    vector = [0] * len(vocab)
    
    term_counts = Counter(terms)
    
    max_freq = max(term_counts.values()) if term_counts else 1
    
    for term in term_counts:
        if term in vocab:
            idx = vocab.index(term)
            tf = 0.5 + 0.5 * (term_counts[term] / max_freq)  # Augmented (0.5-1) & Normalized TF
            vector[idx] = tf * idf[term]
    
    return vector

def compute_document_vectors(vocab, doc_term_freqs, idf):
    doc_vectors = []
    
    for term_freqs in doc_term_freqs:
        terms = list(term_freqs.elements())
        vector = compute_tf_idf_vector(terms, vocab, term_freqs, idf)
        doc_vectors.append(vector)
    
    return doc_vectors

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def search(query, vocab, doc_vectors, idf, documents, top_k=5):
    query_terms = query.lower().split()
    query_freqs = Counter(query_terms)
    query_vector = compute_tf_idf_vector(query_terms, vocab, query_freqs, idf)
    
    similarities = []
    for doc_id, doc_vector in enumerate(doc_vectors):
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def print_document_vectors(vocab, doc_vectors, documents):
    print("\nDocument Vectors:")
    print("Term".ljust(15), end="")
    for i in range(len(documents)):
        print(f"D{i}".rjust(8), end="")
    print()
    print("-" * (15 + 8 * len(documents)))
    
    for term_idx, term in enumerate(vocab):
        print(term.ljust(15), end="")
        for doc_vector in doc_vectors:
            print(f"{doc_vector[term_idx]:.3f}".rjust(8), end="")
        print()

def print_search_results(query, results, documents):
    print(f"\nSearch Results for '{query}':")
    for doc_id, sim in results:
        print(f"Doc {doc_id} ({sim:.3f}): {documents[doc_id]}")

docs = [
    "information retrieval system",
    "database search query",
    "information system database",
    "web search engine",
    "query processing system"
]

# Build the model
vocab, doc_term_freqs = preprocess_documents(docs)
idf = compute_idf(vocab, doc_term_freqs)
doc_vectors = compute_document_vectors(vocab, doc_term_freqs, idf)

# Print document vectors
print_document_vectors(vocab, doc_vectors, docs)

queries = [
    "information system",
    "search database",
    "query processing",
    "web engine"
]

print("\nSearch Results:")
for query in queries:
    results = search(query, vocab, doc_vectors, idf, docs)
    print_search_results(query, results, docs)
    print()
