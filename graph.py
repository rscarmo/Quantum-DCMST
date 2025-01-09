import networkx as nx
import random
import matplotlib.pyplot as plt
import itertools
import numpy as np

class Graph:
    def __init__(self, num_nodes, weight_range=(1, 100), seed=None):
        """
        Initialize a random complete graph.

        Parameters:
        - num_nodes: Number of nodes in the graph.
        - weight_range: Range of edge weights (inclusive).
        - seed: Random seed for reproducibility.
        """
        self.num_nodes = num_nodes
        self.weight_range = weight_range
        self.seed = seed
        self.G = None
        self.pos = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._generate_random_complete_graph()

    def _generate_random_complete_graph(self):
        """Generate a random complete graph."""
        self.G = nx.complete_graph(self.num_nodes)
        for (u, v) in self.G.edges():
            self.G.edges[u, v]['weight'] = random.randint(self.weight_range[0], self.weight_range[1])

    def draw(self, with_labels=True, node_color='lightblue', edge_color='gray', 
             node_size=500, font_size=12):
        """Draw the graph."""
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=42)
        
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(self.G, self.pos, edge_color=edge_color, width=1.5)

        if with_labels:
            nx.draw_networkx_labels(self.G, self.pos, font_size=font_size, font_color='black')

        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_color='red', font_size=10)

        plt.title("Random Complete Graph with Weighted Edges")
        plt.axis('off')
        plt.show()

    def plot_mst(self, mst_edges, node_color='lightblue', edge_color='gray', 
                 node_size=500, font_size=12, mst_color='green', mst_width=2, title="MST"):
        """Plot the Minimum Spanning Tree (MST)."""
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=42)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(self.G, self.pos, edge_color=edge_color, width=1.5)
        nx.draw_networkx_labels(self.G, self.pos, font_size=font_size, font_color='black')

        # Draw MST edges
        nx.draw_networkx_edges(self.G, self.pos, edgelist=mst_edges, edge_color=mst_color, width=mst_width)

        plt.title(title)
        plt.axis('off')
        plt.show()

    def find_mst_kruskal(self):
        """
        Find the Minimum Spanning Tree (MST) using Kruskal's algorithm.
        Returns:
        - mst_edges: List of edges in the MST.
        - total_weight: Total weight of the MST.
        """
        mst = nx.minimum_spanning_tree(self.G, algorithm='kruskal')
        mst_edges = list(mst.edges(data=True))
        total_weight = sum(data['weight'] for _, _, data in mst_edges)
        print("MST (Kruskal) Edges:", [(u, v, data['weight']) for u, v, data in mst_edges])
        print("MST (Kruskal) Total Weight:", total_weight)
        return mst_edges, total_weight

    @staticmethod
    def union_find_setup(n):
        """Set up Union-Find data structure."""
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            rootA = find(a)
            rootB = find(b)
            if rootA != rootB:
                if rank[rootA] < rank[rootB]:
                    parent[rootA] = rootB
                elif rank[rootA] > rank[rootB]:
                    parent[rootB] = rootA
                else:
                    parent[rootB] = rootA
                    rank[rootA] += 1

        return parent, rank, find, union

    @staticmethod
    def is_spanning_tree(edges_subset, n):
        """Check if the subset of edges forms a spanning tree."""
        if len(edges_subset) != n - 1:
            return False

        parent, rank, find, union = Graph.union_find_setup(n)

        for (u, v, _) in edges_subset:
            union(u, v)

        rep = find(0)
        for node in range(1, n):
            if find(node) != rep:
                return False

        return True

    @staticmethod
    def degree_constraint_satisfied(edges_subset, n, max_degree):
        """Check if all nodes in the subset satisfy the degree constraint."""
        degree_count = [0] * n
        for (u, v, _) in edges_subset:
            degree_count[u] += 1
            degree_count[v] += 1

        return all(d <= max_degree for d in degree_count)

    def find_dcmst_brute_force(self, max_degree):
        """
        Find the Degree-Constrained Minimum Spanning Tree (DC-MST) using brute force.
        Returns:
        - best_tree: List of edges in the DC-MST.
        - best_cost: Total weight of the DC-MST.
        """
        n = self.num_nodes
        all_edges = list(self.G.edges(data=True))

        best_tree = None
        best_cost = float('inf')

        for subset in itertools.combinations(all_edges, n - 1):
            if Graph.is_spanning_tree(subset, n):
                if Graph.degree_constraint_satisfied(subset, n, max_degree):
                    cost = sum(data['weight'] for (_, _, data) in subset)
                    if cost < best_cost:
                        best_cost = cost
                        best_tree = subset

        if best_tree is not None:
            print(f"\n** DC-MST Found (max_degree={max_degree}) **")
            print("DC-MST Edges:", [(u, v, data['weight']) for u, v, data in best_tree])
            print("DC-MST Total Weight:", best_cost)
        else:
            print(f"\nNo DC-MST found for max_degree={max_degree}.")

        return best_tree, best_cost

# Example usage
# if __name__ == "__main__":
#     N = 4
#     weight_range = (10, 100)
#     seed = 78
#     max_degree = 2

#     graph = Graph(N, weight_range, seed)
#     graph.draw()

#     mst_edges, total_weight = graph.find_mst_kruskal()
#     graph.plot_mst(mst_edges, title="Kruskal MST")

#     dcmst_edges, dcmst_cost = graph.find_dcmst_brute_force(max_degree)
#     if dcmst_edges is not None:
#         graph.plot_mst(dcmst_edges, mst_color='red', title=f"DC-MST (max_degree={max_degree})")
