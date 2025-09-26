import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def pagerank(graph, damping=0.85, max_iterations=100, tolerance=1e-6):
    all_nodes = set(graph.keys())
    for links in graph.values():
        all_nodes.update(links)
    nodes = sorted(list(all_nodes))
    n = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}

    print(f"Nodes: {nodes}")
    print(f"Graph: {graph}")

    adj_matrix = np.zeros((n, n))
    for node, links in graph.items():
        node_idx = node_to_index[node]
        for link in links:
            if link in node_to_index:
                link_idx = node_to_index[link]
                adj_matrix[node_idx][link_idx] = 1

    print(f"Adjacency Matrix:\n{adj_matrix}")

    H = np.zeros((n, n))
    for i in range(n):
        row_sum = np.sum(adj_matrix[i])
        if row_sum == 0:
            H[i] = np.ones(n) / n  # Equal probability if no outlinks
        else:
            H[i] = adj_matrix[i] / row_sum

    print(f"Hyperlink Matrix H:\n{H}")

    teleportation_matrix = np.ones((n, n)) / n
    G = damping * H + (1 - damping) * teleportation_matrix

    print(f"Google Matrix G:\n{G}")

    eigenvals, eigenvecs = np.linalg.eig(G.T)
    idx = np.argmin(np.abs(eigenvals - 1.0))
    pr_vector = eigenvecs[:, idx].real
    if pr_vector.sum() < 0: pr_vector = -pr_vector
    pr_vector = pr_vector / pr_vector.sum()

    PR = np.ones(n) / n
    print(f"Initial PR: {PR}")

    for iteration in range(max_iterations):
        new_PR = PR @ G
        print(f"PR({iteration+1}): {new_PR}")

        if np.allclose(PR, new_PR, atol=tolerance):
            print(f"Converged after {iteration + 1} iterations")
            break
        PR = new_PR

    result = {nodes[i]: PR[i] for i in range(n)}
    print(f"Final PageRank: {result}")

    return result

def visualize(graph, scores):
    G = nx.DiGraph([(k, v) for k, links in graph.items() for v in links])
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=[scores.get(n, 0.1)*3000 for n in G.nodes()], arrows=True)
    plt.show()

graph = {'B': ['C'], 'C': ['A'], 'D': ['C']}
scores = pagerank(graph)
visualize(graph, scores)