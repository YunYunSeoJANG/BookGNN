import json
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse

def load_data(interactions_path, books_path):
    with open(interactions_path) as f:
        users = pd.DataFrame(json.loads(line) for line in f)
    with open(books_path) as f:
        items = pd.DataFrame(json.loads(line) for line in f)
    return users, items

def create_graph(users):
    G = nx.Graph()
    G.add_nodes_from(users['user_id'], type='user')
    G.add_nodes_from(users['book_id'], type='book')
    edges = [(row['user_id'], row['book_id']) for index, row in users.iterrows()]
    G.add_edges_from(edges)
    return G

def create_subgraphs(G, large_size, large_min_edges, small_size, small_max_edges):
    while True:
        rand_nodes_lg = random.sample(list(G.nodes()), large_size)
        sub_G_lg = G.subgraph(rand_nodes_lg)
        largest_cc_lg = max(nx.connected_components(sub_G_lg.to_undirected()), key=len)
        sub_G_lg = nx.Graph(sub_G_lg.subgraph(largest_cc_lg))
        
        if sub_G_lg.number_of_edges() < large_min_edges:
            continue
        
        rand_nodes_sm = random.sample(list(sub_G_lg.nodes()), small_size)
        sub_G_sm = sub_G_lg.subgraph(rand_nodes_sm)
        largest_cc_sm = max(nx.connected_components(sub_G_sm.to_undirected()), key=len)
        sub_G_sm = nx.Graph(sub_G_sm.subgraph(largest_cc_sm))
        
        if sub_G_sm.number_of_edges() > small_max_edges:
            break
    
    return sub_G_lg, sub_G_sm

def plot_graph(G, title, output_dir, bipartite=False):
    color_map = {"user": [1,0,0], "book": [0,0,1]}
    node_color = [color_map[attr["type"]] for (id, attr) in G.nodes(data=True)]
    
    plt.figure(figsize=(8,8))
    if bipartite:
        top = nx.bipartite.sets(G)[0]
        pos = nx.bipartite_layout(G, top)
    else:
        pos = nx.spring_layout(G)
    
    nx.draw(G, pos=pos, node_color=node_color, node_size=5, width=1, edge_color=(0,0,0,0.1))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

def visualize_graphs(interactions_path, books_path, large_size, large_min_edges, small_size, small_max_edges, output_dir):
    users, items = load_data(interactions_path, books_path)
    G = create_graph(users)
    
    print('Full graph:')
    print(f'Num nodes: {G.number_of_nodes()}, Num edges: {G.number_of_edges()}')
    
    sub_G_lg, sub_G_sm = create_subgraphs(G, large_size, large_min_edges, small_size, small_max_edges)
    
    print('Large subgraph:')
    print(f'Num nodes: {sub_G_lg.number_of_nodes()}, Num edges: {sub_G_lg.number_of_edges()}')
    
    print('Small subgraph:')
    print(f'Num nodes: {sub_G_sm.number_of_nodes()}, Num edges: {sub_G_sm.number_of_edges()}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_graph(sub_G_lg, "Large Subgraph", output_dir)
    plot_graph(sub_G_sm, "Small Subgraph", output_dir)
    plot_graph(sub_G_sm, "Small Subgraph (Bipartite)", output_dir, bipartite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize graphs from JSON files")
    parser.add_argument('--interactions_path', type=str, required=True, help="Path to the interactions JSON file")
    parser.add_argument('--books_path', type=str, required=True, help="Path to the books JSON file")
    parser.add_argument('--large_size', type=int, default=5000, help="Number of nodes in the large subgraph")
    parser.add_argument('--large_min_edges', type=int, default=150, help="Minimum number of edges in the large subgraph")
    parser.add_argument('--small_size', type=int, default=100, help="Number of nodes in the small subgraph")
    parser.add_argument('--small_max_edges', type=int, default=50, help="Maximum number of edges in the small subgraph")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save the output plots")
    args = parser.parse_args()

    visualize_graphs(args.interactions_path, args.books_path, args.large_size, args.large_min_edges, args.small_size, args.small_max_edges, args.output_dir)
