
import numpy as np
import math
import random

def build_shingle_matrix(docs, k=2):
    shingles = set()
    doc_shingles = []

    for doc in docs:
        words = doc.lower().split()
        s = {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}
        shingles |= s
        doc_shingles.append(s)

    shingles = list(shingles)
    shingle_index = {s: i for i, s in enumerate(shingles)}

    M = np.zeros((len(shingles), len(docs)), dtype=int)
    for j, s_set in enumerate(doc_shingles):
        for s in s_set:
            M[shingle_index[s], j] = 1

    return M, shingles



def minhash_signature(M, num_hashes=100):
    n_shingles, n_docs = M.shape
    sig = np.full((num_hashes, n_docs), np.inf)

    p = 2**31 - 1
    hash_funcs = [(random.randint(1, p-1), random.randint(0, p-1)) for _ in range(num_hashes)]

    for r in range(n_shingles):
        row = M[r]
        for h, (a, b) in enumerate(hash_funcs):
            hash_val = (a*r + b) % p
            for c in range(n_docs):
                if row[c] == 1:
                    sig[h, c] = min(sig[h, c], hash_val)
    return sig

def minhash_signature_permutation(M, num_permutations=100):
    n_shingles, n_docs = M.shape
    sig = np.full((num_permutations, n_docs), np.inf)

    for p in range(num_permutations):
        perm = np.random.permutation(n_shingles)
        for c in range(n_docs):
            for idx in perm:
                if M[idx, c] == 1:
                    sig[p, c] = idx
                    break
    return sig

def jaccard_from_signatures(sig, i, j):
    return np.mean(sig[:, i] == sig[:, j])

def tokenize(doc):
    return doc.lower().split()

def build_count_matrix(docs):
    vocab = {}
    tokenized_docs = []
    for doc in docs:
        tokens = tokenize(doc)
        tokenized_docs.append(tokens)
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    V = len(vocab)
    D = len(docs)
    count_matrix = np.zeros((V, D), dtype=float)

    for j, tokens in enumerate(tokenized_docs):
        for t in tokens:
            i = vocab[t]
            count_matrix[i, j] += 1

    return count_matrix, vocab

def compute_tfidf(count_matrix):
    V, D = count_matrix.shape
    tfidf = np.zeros((V, D))

    df = np.count_nonzero(count_matrix > 0, axis=1)

    for j in range(D):
        for i in range(V):
            tf = count_matrix[i, j] / (np.sum(count_matrix[:, j]) + 1e-10)
            idf = math.log((D + 1) / (df[i] + 1)) + 1
            tfidf[i, j] = tf * idf

    return tfidf

def cosine_similarity(A):
    D = A.shape[1]
    sim = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            num = np.dot(A[:, i], A[:, j])
            denom = (np.linalg.norm(A[:, i]) * np.linalg.norm(A[:, j]) + 1e-10)
            sim[i, j] = num / denom
    return sim

def euclidean_distance(A):
    D = A.shape[1]
    dist = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            dist[i, j] = np.linalg.norm(A[:, i] - A[:, j])
    return dist


docs = [
    "the quick brown fox",
    "the quick brown dog",
    "the fast brown fox"
]

print("\n=== MinHash (Binary Shingle Matrix) ===")
M, shingles = build_shingle_matrix(docs, k=3)
print("Binary Shingle-Doc Matrix:\n", M)

sig = minhash_signature(M, num_hashes=50)

# Hash function method
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = jaccard_from_signatures(sig, i, j)
        print(f"Estimated Jaccard similarity (Doc {i}, Doc {j}): {sim:.3f}")


# Permutation method
sig_perm = minhash_signature_permutation(M, num_permutations=50)
print("\n--- Using Permutation Method ---")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = jaccard_from_signatures(sig_perm, i, j)
        print(f"Estimated Jaccard similarity (Doc {i}, Doc {j}): {sim:.3f}")

print("\n=== TF-IDF (Cosine & Euclidean) ===")
count_matrix, vocab = build_count_matrix(docs)
tfidf = compute_tfidf(count_matrix)

cos_sim = cosine_similarity(tfidf)
euc_dist = euclidean_distance(tfidf)

print("Cosine Similarity:\n", cos_sim)
print("Euclidean Distances:\n", euc_dist)
