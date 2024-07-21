import json, pandas as pd
import os

def preprocess_books():
    with open('../goodreads/goodreads_books_poetry.json') as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    aut=df.loc[:,'authors']
    ddf = pd.DataFrame(line for line in aut)
    aut = ddf.loc[:,0]
    ddf = pd.DataFrame(line for line in aut)
    ddf = ddf.loc[:,'author_id']

    df.loc[:,'authors'] = ddf

    df = df.loc[:,['book_id', 'average_rating', 'ratings_count', 'authors', 'similar_books', 'title', 'url']]

    os.makedirs('../datasets', exist_ok=True)
    df.to_json('../datasets/books_poetry.json', orient='records', lines=True)

def preprocess_interactions():
    with open('../goodreads/goodreads_interactions_poetry.json') as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    df = df.loc[:,['book_id', 'user_id', 'is_read', 'rating']]

    df.to_json('../datasets/interactions_poetry.json', orient='records', lines=True)


if __name__ == '__main__':
    preprocess_books()
    preprocess_interactions()
