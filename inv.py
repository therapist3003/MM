def build_index(docs):
    index = {}
    for i, doc in enumerate(docs):
        for term in set(doc.lower().split()):
            if term not in index:
                index[term] = []
            index[term].append(i)
    return index

def get_posting_list(index, term):
    return index.get(term.lower(), [])

def AND(list1, list2):
    return [x for x in list1 if x in list2]

def OR(list1, list2):
    return sorted(set(list1 + list2))

def NOT(posting_list, total_docs):
    all_docs = list(range(total_docs))
    return [x for x in all_docs if x not in posting_list]

def optimize_terms(index, terms, operation='and'):
    """
    Sort terms by posting list length for optimal processing
    Returns terms sorted by posting list length
    """
    term_lengths = [(term, len(get_posting_list(index, term))) for term in terms]
    
    if operation == 'and':
        # For AND: process shortest lists first (fewer intersections)
        return [term for term, _ in sorted(term_lengths, key=lambda x: x[1])]
    else:  # OR
        # For OR: process longest lists first (build result faster)
        return [term for term, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]

def search(index, query, total_docs):
    q = query.lower()
    
    if ' and ' in q:
        terms = [t.strip() for t in q.split(' and ')]
        # Optimize: shortest posting lists first
        terms = optimize_terms(index, terms, 'and')
        result = get_posting_list(index, terms[0])
        for term in terms[1:]:
            result = AND(result, get_posting_list(index, term))
            if not result:  # Early termination
                break
        return result
    
    elif ' or ' in q:
        terms = [t.strip() for t in q.split(' or ')]
        # Optimize: longest posting lists first
        terms = optimize_terms(index, terms, 'or')
        result = get_posting_list(index, terms[0])
        for term in terms[1:]:
            result = OR(result, get_posting_list(index, term))
        return result
    
    elif ' not ' in q:
        pos, neg = q.split(' not ')
        pos_list = get_posting_list(index, pos.strip())
        neg_list = get_posting_list(index, neg.strip())
        return AND(pos_list, NOT(neg_list, total_docs))
    
    else:
        return get_posting_list(index, q)

def print_search_results(docs, result_ids):
    """
    Print the actual documents matching the search results
    """
    print("\nMatching documents:")
    for doc_id in result_ids:
        print(f"Doc {doc_id}: {docs[doc_id]}")

docs = [
    "cat dog bird",
    "dog bird",
    "cat mouse",
    "bird eagle",
    "mouse cat"
]

index = build_index(docs)
print("Inverted Index:", index)

# Example queries
queries = [
    "cat and bird",
    "cat or dog",
    "bird not cat",
    "mouse"
]

print("\nQuery Results:")
for query in queries:
    print(f"\nQuery: '{query}'")
    if ' and ' in query:
        terms = [t.strip() for t in query.split(' and ')]
        print("Posting list sizes:")
        for term in terms:
            print(f"  {term}: {len(get_posting_list(index, term))} docs")
        print(f"Optimized order: {optimize_terms(index, terms, 'and')}")
    
    results = search(index, query, len(docs))
    print(f"Matching document IDs: {results}")
    print_search_results(docs, results)
