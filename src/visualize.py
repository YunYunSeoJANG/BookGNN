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

def plot_graph(G, title, output_dir, bipartite=False, items_df=None, figsize=(20, 20)):
    color_map = {"user": "red", "book": "blue"}
    node_color = [color_map[attr["type"]] for (id, attr) in G.nodes(data=True)]

    fig, ax = plt.subplots(figsize=figsize)
    if bipartite:
        top = nx.bipartite.sets(G)[0]
        pos = nx.bipartite_layout(G, top)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    node_size = [1000 if attr["type"] == "book" else 500 for (id, attr) in G.nodes(data=True)]

    nx.draw(G, pos=pos, node_color=node_color, node_size=node_size, 
            width=0.5, edge_color=(0, 0, 0, 0.1), with_labels=False, ax=ax)

    labels = {}
    for node, attr in G.nodes(data=True):
        if attr["type"] == "book":
            book_info = items_df[items_df['book_id'] == node].iloc[0]
            labels[node] = f"{book_info['title'][:20]}..."
        else:
            labels[node] = f"{node[:8]}..."
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    ax.set_title(title, fontsize=24)

    info_text = f"Total nodes: {G.number_of_nodes()}\n"
    info_text += f"Total edges: {G.number_of_edges()}\n"
    info_text += f"User nodes: {sum(1 for (_, attr) in G.nodes(data=True) if attr['type'] == 'user')}\n"
    info_text += f"Book nodes: {sum(1 for (_, attr) in G.nodes(data=True) if attr['type'] == 'book')}\n\n"

    info_text += "Top 5 Books by Connections:\n"
    book_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'book']
    top_books = sorted(book_nodes, key=lambda x: G.degree(x), reverse=True)[:5]
    for i, book_id in enumerate(top_books, 1):
        book_info = items_df[items_df['book_id'] == book_id].iloc[0]
        info_text += f"{i}. {book_info['title']} by {book_info['authors']} (Connections: {G.degree(book_id)})\n"

    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"), 
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    with open(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_info.txt"), 'w') as f:
        f.write(info_text + "\n")

    create_user_book_connections_json(G, items_df, output_dir, title)

def create_user_book_connections_json(G, items_df, output_dir, title):
    user_book_connections = {}
    for node, attr in G.nodes(data=True):
        if attr['type'] == 'user':
            connected_books = []
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['type'] == 'book':
                    book_info = items_df[items_df['book_id'] == neighbor].iloc[0]
                    connected_books.append(book_info['title'])
            user_book_connections[node] = connected_books
    
    with open(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_connections.json"), 'w') as f:
        json.dump(user_book_connections, f, indent=2)

def visualize_graphs(large_size, large_min_edges, small_size, small_max_edges):
    interactions_path = '../datasets/interactions_poetry.json'
    books_path = '../datasets/books_poetry.json'
    output_dir = 'output'
    
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
    
    plot_graph(sub_G_lg, "Large Subgraph", output_dir, items_df=items, figsize=(30, 30))
    plot_graph(sub_G_sm, "Small Subgraph", output_dir, items_df=items, figsize=(20, 20))
    plot_graph(sub_G_sm, "Small Subgraph (Bipartite)", output_dir, bipartite=True, items_df=items, figsize=(20, 20))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize graphs from JSON files")
    parser.add_argument('--large_size', type=int, default=5000, help="Number of nodes in the large subgraph")
    parser.add_argument('--large_min_edges', type=int, default=150, help="Minimum number of edges in the large subgraph")
    parser.add_argument('--small_size', type=int, default=100, help="Number of nodes in the small subgraph")
    parser.add_argument('--small_max_edges', type=int, default=50, help="Maximum number of edges in the small subgraph")
    args = parser.parse_args()

    visualize_graphs(args.large_size, args.large_min_edges, args.small_size, args.small_max_edges)