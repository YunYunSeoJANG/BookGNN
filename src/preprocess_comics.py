import json, pandas as pd
import os, pickle
import networkx as nx


def preprocess_interactions():
    with open('goodreads/goodreads_interactions_comics_graphic.json') as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    df = df.loc[:,['book_id', 'user_id', 'is_read', 'rating']]

    df.to_json('datasets/interactions_comics_graphic.json', orient='records', lines=True)


def load_graph():
    # Read interactions from preprocessed json file in data_preprocessing.ipynb
    with open('datasets/interactions_comics_graphic.json') as f:
      users = pd.DataFrame(json.loads(line) for line in f)

    # # is it necessary? it doesn't seem to be used
    # with open('../datasets/books_romance.json') as f:
    #   items = pd.DataFrame(json.loads(line) for line in f)

    # Make an empty graph
    G = nx.Graph()

    # Add user nodes to the Graph (377799 users)
    G.add_nodes_from(users['user_id'], type = 'user')
    # print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges()) # Num nodes: 377799 . Num edges: 0

    # Add item nodes to the graph (36514 books)
    G.add_nodes_from(users['book_id'], type = 'book')

    # Make a bipartite graph
    edges = [(row['user_id'], row['book_id']) for index, row in users.iterrows()]
    G.add_edges_from(edges)
    
    # print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges()) # Num nodes: 414313 . Num edges: 2734350

    kcore = 30
    G = nx.k_core(G, kcore)
    # print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges()) # Num nodes: 17738 . Num edges: 767616

    # print(list(G.nodes(data=True))[:5])


    os.makedirs('assets', exist_ok=True)
    with open('assets/graph_kcore_comics_graphic.gpickle', 'wb') as f:
        pickle.dump(G, f)

    return G

if __name__ == '__main__':
    print("preprocessing interactions")
    preprocess_interactions()
    print("loading graph")
    load_graph()