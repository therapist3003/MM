def build_matrix(documents):
    all_terms = set()
    processed_docs = []
    
    for doc in documents:
        tokens = doc.lower().split()
        processed_docs.append(tokens)
        all_terms.update(tokens)
    
    vocab = sorted(all_terms)
    
    term_doc_matrix = []
    for term in vocab:
        row = []
        for doc_tokens in processed_docs:
            row.append(1 if term in doc_tokens else 0)
        term_doc_matrix.append(row)
    
    return vocab, term_doc_matrix

def get_term_vector(term, vocab, term_doc_matrix):
    term = term.lower()
    if term not in vocab:
        return [0] * len(term_doc_matrix[0])
    
    term_idx = vocab.index(term)
    return term_doc_matrix[term_idx]

def boolean_and(term1, term2, vocab, term_doc_matrix):
    vec1 = get_term_vector(term1, vocab, term_doc_matrix)
    vec2 = get_term_vector(term2, vocab, term_doc_matrix)
    return [a & b for a, b in zip(vec1, vec2)]

def boolean_or(term1, term2, vocab, term_doc_matrix):
    vec1 = get_term_vector(term1, vocab, term_doc_matrix)
    vec2 = get_term_vector(term2, vocab, term_doc_matrix)
    return [a | b for a, b in zip(vec1, vec2)]

def boolean_not(term, vocab, term_doc_matrix):
    vec = get_term_vector(term, vocab, term_doc_matrix)
    return [1 - x for x in vec]

def search(query, vocab, term_doc_matrix):
    query = query.lower().strip()
    
    # Single term
    if ' ' not in query:
        result_vector = get_term_vector(query, vocab, term_doc_matrix)
    
    # AND query
    elif ' and ' in query:
        term1, term2 = [t.strip() for t in query.split(' and ')]
        result_vector = boolean_and(term1, term2, vocab, term_doc_matrix)
    
    # OR query
    elif ' or ' in query:
        term1, term2 = [t.strip() for t in query.split(' or ')]
        result_vector = boolean_or(term1, term2, vocab, term_doc_matrix)
    
    # NOT query
    elif ' not ' in query:
        term1, term2 = [t.strip() for t in query.split(' not ')]
        vec1 = get_term_vector(term1, vocab, term_doc_matrix)
        vec2 = boolean_not(term2, vocab, term_doc_matrix)
        result_vector = [a & b for a, b in zip(vec1, vec2)]
    
    else:
        return []
    
    return [i for i, val in enumerate(result_vector) if val == 1]

def print_matrix(vocab, term_doc_matrix, documents):
    print("\nTerm-Document Matrix:")
    print("Term".ljust(15), end="")
    for i in range(len(documents)):
        print(f"D{i}".rjust(4), end="")
    print()
    print("-" * (15 + 4 * len(documents)))
    
    # Print matrix rows
    for term, row in zip(vocab, term_doc_matrix):
        print(term.ljust(15), end="")
        for val in row:
            print(f"{val}".rjust(4), end="")
        print()

docs = [
    "information retrieval system",
    "database search query",
    "information system database",
    "web search engine",
    "query processing system"
]

vocab, term_doc_matrix = build_matrix(docs)
print_matrix(vocab, term_doc_matrix, docs)

print("\nSearch Results:")
queries = [
    "information",
    "information and system",
    "search or query",
    "system not database"
]

for query in queries:
    results = search(query, vocab, term_doc_matrix)
    print(f"'{query}': {results}")
    # Print matching documents
    print("Matching documents:")
    for doc_id in results:
        print(f"  Doc {doc_id}: {docs[doc_id]}")
    print()
