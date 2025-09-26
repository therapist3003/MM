import numpy as np
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0

    cosine_sim = dot_product / (norm1 * norm2)

    print("Cosine similarity:", cosine_sim)
    return cosine_sim

def preprocess_text(text, stop_words=None):
    if stop_words is None:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    text = text.lower()
    words = re.findall(r'\w+', text)
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def build_term_doc_matrix(docs, stop_words=None):
    tokenized_docs = [preprocess_text(doc, stop_words) for doc in docs]
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    matrix = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(tokenized_docs):
        for word in doc:
            matrix[vocab_index[word], j] += 1

    return matrix, vocab

def lsi_query(docs, query, k=2, stop_words=None):
    term_doc_matrix, vocab = build_term_doc_matrix(docs, stop_words)

    svd = TruncatedSVD(n_components=k, random_state=42)
    doc_vectors = svd.fit_transform(term_doc_matrix.T)  # (docs x k)

    query_tokens = preprocess_text(query, stop_words)
    query_vec = np.zeros((len(vocab),))
    for word in query_tokens:
        if word in vocab:
            query_vec[vocab.index(word)] += 1

    query_reduced = svd.transform(query_vec.reshape(1, -1))

    sims = cosine_similarity(query_reduced, doc_vectors)[0]
    return sims

documents = [
    "The cat sat on the mat",
    "Dogs and cats are friends",
    "I love playing with my dog"
]

query = "cat and dog"

scores = lsi_query(documents, query, k=2)
print("Similarity scores:", scores)
print("Most relevant document index:", np.argmax(scores))
