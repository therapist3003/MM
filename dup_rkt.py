import random
import re
import hashlib

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def get_shingles(words, k=3):
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

def stable_hash(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

def minhash_random_permutation(shingle_sets, num_perms=50):
    universe = list(set().union(*shingle_sets))
    signatures = []
    for s in shingle_sets:
        sig = []
        for _ in range(num_perms):
            perm = universe.copy()
            random.shuffle(perm)
            for shingle in perm:
                if shingle in s:
                    sig.append(shingle)
                    break
        signatures.append(sig)
    return signatures

def minhash_affine(shingle_sets, hash_funcs):
    signatures = []
    for s in shingle_sets:
        sig = []
        for h in hash_funcs:
            sig.append(min(h(stable_hash(shingle)) for shingle in s))
        signatures.append(sig)
    return signatures

def make_simple_hash_funcs():
    return [
        lambda x: (x + 2) % 7,
        lambda x: (3 * x + 1) % 7,
        lambda x: (2 * x + 3) % 7
    ]
def estimate_jaccard(sig1, sig2):
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def detect_duplicates(documents, query=None, k=3, method="affine", num_perms=50):
    shingle_sets = [get_shingles(preprocess(doc), k) for doc in documents]

    if method == "random_perm":
        doc_signatures = minhash_random_permutation(shingle_sets, num_perms=num_perms)
    else:
        hash_funcs = make_simple_hash_funcs()
        doc_signatures = minhash_affine(shingle_sets, hash_funcs)

    duplicates = []
    n = len(documents)
    for i in range(n):
        for j in range(i+1, n):
            sim = estimate_jaccard(doc_signatures[i], doc_signatures[j])
            if sim > 0.5:  # threshold for duplicates
                duplicates.append((i, j, sim))

    if query:
        query_shingles = get_shingles(preprocess(query), k)
        if method == "random_perm":
            query_sig = minhash_random_permutation([query_shingles], num_perms=num_perms)[0]
        else:
            query_sig = minhash_affine([query_shingles], hash_funcs)[0]
        sims = [estimate_jaccard(query_sig, doc_sig) for doc_sig in doc_signatures]
        best_idx = sims.index(max(sims))
        return duplicates, sims, best_idx

    return duplicates

documents = [
    "The cat sat on the mat.",
    "A cat was sitting on the mat.",
    "Dogs are great pets.",
    "The cat sat on the mat." 
]
query = "cat and dog on the mat"

duplicates, sims, best_doc = detect_duplicates(documents, query, method="affine")
print("Affine Hash Functions:")
print("Duplicate pairs (doc_i, doc_j, sim):", duplicates)
print("Query similarity scores:", [f"{s:.2f}" for s in sims])
print("Most relevant document index:", best_doc)

duplicates_rp, sims_rp, best_doc_rp = detect_duplicates(documents, query, method="random_perm", num_perms=50)
print("\nRandom Permutations:")
print("Duplicate pairs (doc_i, doc_j, sim):", duplicates_rp)
print("Query similarity scores:", [f"{s:.2f}" for s in sims_rp])
print("Most relevant document index:", best_doc_rp)
