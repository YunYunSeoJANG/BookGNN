import json, pandas as pd
import os

def preprocess_books(path = '../goodreads/goodreads_books_mystery_thriller_crime.json', save = '../datasets/books_mystery_thriller_crime.json'):
    with open(path) as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    aut=df.loc[:,'authors']
    ddf = pd.DataFrame(line for line in aut)
    aut = ddf.loc[:,0]
    ddf = pd.DataFrame(line for line in aut)
    ddf = ddf.loc[:,'author_id']

    df.loc[:,'authors'] = ddf

    df = df.loc[:,['book_id', 'average_rating', 'ratings_count', 'authors', 'similar_books', 'title', 'url']]

    os.makedirs('../datasets', exist_ok=True)
    df.to_json(save, orient='records', lines=True)

def preprocess_interactions(path = '../goodreads/goodreads_interactions_mystery_thriller_crime.json', save = '../datasets/interactions_mystery_thriller_crime.json'):
    with open(path) as f:
        df = pd.DataFrame(json.loads(line) for line in f)

    df = df.loc[:,['book_id', 'user_id', 'is_read', 'rating']]

    df.to_json(save, orient='records', lines=True)


if __name__ == '__main__':
    preprocess_books()
    preprocess_interactions()
