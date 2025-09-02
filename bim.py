import math

def preprocess_documents(docs):
    vocab_set = set()
    processed = []
    
    for doc in docs:
        tokens = set(doc.lower().split())
        processed.append(tokens)
        vocab_set.update(tokens)
    
    vocab = sorted(vocab_set)
    
    binary_matrix = []
    for doc_tokens in processed:
        row = [1 if term in doc_tokens else 0 for term in vocab]
        binary_matrix.append(row)
    
    return vocab, binary_matrix

def phase1_estimate(query_terms, vocab, binary_matrix):
    N_d = len(binary_matrix)
    estimates = {}
    
    for term in query_terms:
        if term not in vocab:
            continue
        
        term_idx = vocab.index(term)
        
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)
        
        # Phase I estimates
        p_k = 0.5 # random chance 
        q_k_simple = d_k / N_d if N_d > 0 else 0  # Simple estimation
        q_k = (d_k + 0.5) / (N_d + 1)  # With smoothing
        
        estimates[term] = {
            'd_k': d_k,
            'p_k': p_k,
            'q_k': q_k,
            'q_k_simple': q_k_simple
        }
    
    return estimates

def phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs):
    N_d = len(binary_matrix)
    N_r = len(relevant_docs)
    estimates = {}
    
    for term in query_terms:
        if term not in vocab:
            continue
        
        term_idx = vocab.index(term)
        
        # Relevant documents containing the term
        r_k = sum(1 for doc_id in relevant_docs 
                 if binary_matrix[doc_id][term_idx] == 1)

        # Total documents containing the term
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)
        
        # Phase II estimates

        # without smoothing
        p_k_simple = r_k / N_r if N_r > 0 else 0
        q_k_simple = (d_k - r_k) / (N_d - N_r) if (N_d - N_r) > 0 else 0

        # with smoothing
        p_k = (r_k + 0.5) / (N_r + 1)
        q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)
        
        estimates[term] = {
            'r_k': r_k,
            'd_k': d_k,
            'p_k': p_k,
            'p_k_simple': p_k_simple,
            'q_k': q_k,
            'q_k_simple': q_k_simple
        }
    
    return estimates

def calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix):
    rsv = 0
    
    for term in query_terms:
        if term not in estimates:
            continue
        
        term_idx = vocab.index(term)
        p_k = estimates[term]['p_k']
        q_k = estimates[term]['q_k']
        
        if binary_matrix[doc_id][term_idx] == 1:
            # Document contains term
            if p_k > 0 and q_k > 0:
                rsv += math.log(p_k / q_k)
        else:
            # Document doesn't contain term
            if p_k < 1 and q_k < 1:
                rsv += math.log((1 - p_k) / (1 - q_k))
    
    return rsv

def search_phase1(query, docs, top_k=5):
    query_terms = query.lower().split()
    vocab, binary_matrix = preprocess_documents(docs)
    estimates = phase1_estimate(query_terms, vocab, binary_matrix)
    
    # Calculate RSV for each document
    doc_scores = []
    for doc_id in range(len(docs)):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))
    
    # Sort by RSV and return top k
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]

def search_phase2(query, docs, relevant_docs, top_k=5):
    query_terms = query.lower().split()
    vocab, binary_matrix = preprocess_documents(docs)
    estimates = phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs)
    
    # Calculate RSV for each document
    doc_scores = []
    for doc_id in range(len(docs)):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))
    
    # Sort by RSV and return top k
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]

def print_estimates(query_terms, docs, relevant_docs=None):
    vocab, binary_matrix = preprocess_documents(docs)
    
    if relevant_docs is None:
        print("=== PHASE I ESTIMATES ===")
        estimates = phase1_estimate(query_terms, vocab, binary_matrix)
        for term, est in estimates.items():
            print(f"{term}: d_k={est['d_k']}, p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")
    else:
        print("=== PHASE II ESTIMATES ===")
        estimates = phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs)
        for term, est in estimates.items():
            print(f"{term}: r_k={est['r_k']}, d_k={est['d_k']}")
            print(f"      p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")

docs = [
    "information retrieval system",
    "database search query",
    "information system database",
    "web search engine",
    "query processing system"
]

query = "information system"
query_terms = query.split()

print("=== PHASE I (No Relevance Info) ===")
print_estimates(query_terms, docs)
results1 = search_phase1(query, docs)
print(f"Phase I Results: {results1}")

print("\n=== PHASE II (With Relevance Feedback) ===")
relevant_docs = [0, 2]  # Assume docs 0,2 are relevant
print_estimates(query_terms, docs, relevant_docs)
results2 = search_phase2(query, docs, relevant_docs)
print(f"Phase II Results: {results2}")

# If considering query for phase 1 :
'''
    tr_k = log[ (Nd - dk) / dk ]

    For similarity, find cross-prod of query weights and tr_k for each doc
'''

# If considering query for phase 2:
'''
    tr_k = (A/C) / (B/D)
    A - rk + 0.5
    B - dk - rk + 0.5
    C - Nr - rk + 0.5
    D - Nd - Nr - (dk - rk) + 0.5

    For similarity, find cross-prod of query weights and tr_k for each doc
'''