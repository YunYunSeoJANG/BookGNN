import torch
import pickle
import os
import json
from utils.preprocess import preprocess_graph

def load_embeddings_pt(embedding_path, device=None):
    embeddings = torch.load(embedding_path, map_location=device)
    return embeddings

def recommend_books(user_id, user_embeddings, item_embeddings, item_idx, idx_id):

    user_embedding = user_embeddings[user_id].unsqueeze(0)  # 1xD

    scores = torch.matmul(user_embedding, item_embeddings.t())  # 1xM

    top_30_scores, top_30_indices = torch.topk(scores, 30, dim=1)
    recommended_books = {idx_id[item_idx[idx]]: top_30_scores.squeeze().tolist()[i] 
                         for i, idx in enumerate(top_30_indices.squeeze().tolist())}

    return recommended_books

def top_5_keys_by_value(d):
    sorted_keys = sorted(d, key=d.get, reverse=True)
    return sorted_keys[:5]

if __name__ == "__main__":
    print("Loading graph from gpickle file.")
    with open(r'C:\Users\kjsid\Desktop\Prometheus\BookGNN\assets\graph_kcore.gpickle', 'rb') as f:
        G = pickle.load(f)

    G, user_idx, item_idx, n_user, n_item, idx_id = preprocess_graph(G)

    embedding_path = r"C:\Users\kjsid\Desktop\Prometheus\BookGNN_test\src\model_embeddings\LGCN_LGC_3_e64_nodes17738_\LGCN_LGC_3_e64_nodes17738__BPR_random_300.pt"
    embeddings = load_embeddings_pt(embedding_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    user_embeddings = embeddings[:n_user]
    item_embeddings = embeddings[n_user:n_user + n_item]

    user_id = 42  # 추천할 사용자 ID
    recommended_books = recommend_books(user_id, user_embeddings, item_embeddings, item_idx, idx_id)

    
    json_file_path = r"C:\Users\kjsid\Desktop\Prometheus\BookGNN_test\datasets\books_poetry.json"

    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_objects = []
        for line in file:
            json_objects.append(json.loads(line))

    # Function to get the average rating based on book_id
    def get_average_rating(book_id):
        for book in json_objects:
            if book["book_id"] == book_id:
                return float(book["average_rating"])
        print("Book ID not found")

    for book_id, score in recommended_books.items():
        recommended_books[book_id] = score + get_average_rating(book_id)


    print("추천된 책 리스트:", top_5_keys_by_value(recommended_books))