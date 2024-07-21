# 0. DATASET OVERVIEW
From [Goodreads Book Graph Datasets](https://mengtingwan.github.io/data/goodreads.html), includes book/user/review ids, **Poetry (36,514 books, 2,734,350 interactions, 154,555 detailed reviews)**
cf. reviews는 interaction중 review text가 긴 것을 따로 모아 놓은 것임.

아래는 `"book_id": "402128"` 에 대한 예시

### 1. goodreads_books_poetry
36,514 books

```json
{"isbn": "0151686564", "text_reviews_count": "626", "series": [], "country_code": "US", "language_code": "en-US", "popular_shelves": [{"count": "15419", "name": "to-read"}, {"count": "1762", "name": "poetry"}, {"count": "302", "name": "classics"}, {"count": "187", "name": "favorites"}, {"count": "149", "name": "fiction"}, {"count": "136", "name": "cats"}, {"count": "107", "name": "animals"}, {"count": "104", "name": "owned"}, {"count": "100", "name": "books-i-own"}, {"count": "97", "name": "humor"}, {"count": "68", "name": "childrens"}, {"count": "63", "name": "currently-reading"}, {"count": "63", "name": "children"}, {"count": "55", "name": "classic"}, {"count": "46", "name": "literature"}, {"count": "45", "name": "children-s"}, {"count": "36", "name": "default"}, {"count": "31", "name": "humour"}, {"count": "31", "name": "children-s-books"}, {"count": "29", "name": "poems"}, {"count": "28", "name": "owned-books"}, {"count": "25", "name": "library"}, {"count": "25", "name": "favourites"}, {"count": "24", "name": "english"}, {"count": "23", "name": "british"}, {"count": "21", "name": "art"}, {"count": "20", "name": "childhood"}, {"count": "19", "name": "to-buy"}, {"count": "18", "name": "illustrated"}, {"count": "18", "name": "my-library"}, {"count": "18", "name": "kids"}, {"count": "18", "name": "american"}, {"count": "18", "name": "20th-century"}, {"count": "16", "name": "british-literature"}, {"count": "15", "name": "my-books"}, {"count": "14", "name": "children-s-literature"}, {"count": "12", "name": "picture-books"}, {"count": "12", "name": "home-library"}, {"count": "12", "name": "childrens-books"}, {"count": "12", "name": "childhood-favorites"}, {"count": "12", "name": "young-adult"}, {"count": "12", "name": "edward-gorey"}, {"count": "12", "name": "non-fiction"}, {"count": "11", "name": "read-in-2016"}, {"count": "11", "name": "verse"}, {"count": "11", "name": "poes\u00eda"}, {"count": "11", "name": "fantasy"}, {"count": "11", "name": "read-in-english"}, {"count": "11", "name": "own-it"}, {"count": "11", "name": "nobel"}, {"count": "10", "name": "read-in-2017"}, {"count": "10", "name": "poesia"}, {"count": "10", "name": "poetry-plays"}, {"count": "10", "name": "wish-list"}, {"count": "9", "name": "nobel-prize"}, {"count": "9", "name": "1930s"}, {"count": "9", "name": "re-read"}, {"count": "9", "name": "i-own"}, {"count": "8", "name": "2017-reading-challenge"}, {"count": "8", "name": "2016-reading-challenge"}, {"count": "8", "name": "childhood-books"}, {"count": "8", "name": "funny"}, {"count": "8", "name": "books-we-own"}, {"count": "8", "name": "read-aloud"}, {"count": "8", "name": "nonfiction"}, {"count": "7", "name": "read-in-2014"}, {"count": "7", "name": "to-read-poetry"}, {"count": "7", "name": "childrens-lit"}, {"count": "7", "name": "classic-literature"}, {"count": "7", "name": "other"}, {"count": "7", "name": "mine"}, {"count": "7", "name": "general-fiction"}, {"count": "6", "name": "read-2016"}, {"count": "6", "name": "modern-classics"}, {"count": "6", "name": "nobel-laureates"}, {"count": "6", "name": "cat"}, {"count": "6", "name": "poetry-and-plays"}, {"count": "6", "name": "kids-books"}, {"count": "6", "name": "read-alouds"}, {"count": "6", "name": "short-stories"}, {"count": "6", "name": "nobel-prize-winners"}, {"count": "6", "name": "lit"}, {"count": "6", "name": "want-to-own"}, {"count": "6", "name": "plays"}, {"count": "6", "name": "american-lit"}, {"count": "6", "name": "in-english"}, {"count": "5", "name": "read-2015"}, {"count": "5", "name": "musicals"}, {"count": "5", "name": "home"}, {"count": "5", "name": "talking-animals"}, {"count": "5", "name": "read-in-2012"}, {"count": "5", "name": "have"}, {"count": "5", "name": "adult"}, {"count": "5", "name": "modernism"}, {"count": "5", "name": "want"}, {"count": "5", "name": "read-in-2011"}, {"count": "5", "name": "t-s-eliot"}, {"count": "5", "name": "animal-fiction"}, {"count": "5", "name": "children-s-lit"}, {"count": "5", "name": "england"}], "asin": "", "is_ebook": "false", "average_rating": "4.09", "kindle_asin": "", "similar_books": ["884306", "234", "472443", "51244", "305154", "574889", "201711", "857597", "858497", "864051", "400723", "47564", "1391333", "133380", "285151"], "description": "T. S. Eliot's playful cat poems have delighted readers and cat lovers around the world ever since they were first published in 1939. They were originally composed for his godchildren, with Eliot posing as Old Possum himself, and later inspired the legendary musical Cats.", "format": "Hardcover", "link": "https://www.goodreads.com/book/show/402128.Old_Possum_s_Book_of_Practical_Cats", "authors": [{"author_id": "18540", "role": ""}, {"author_id": "21578", "role": "Illustrator"}], "publisher": "Harcourt Brace & Company", "num_pages": "56", "publication_day": "30", "isbn13": "9780151686568", "publication_month": "8", "edition_information": "Illustrated Edition", "publication_year": "1982", "url": "https://www.goodreads.com/book/show/402128.Old_Possum_s_Book_of_Practical_Cats", "image_url": "https://images.gr-assets.com/books/1327882662m/402128.jpg", "book_id": "402128", "ratings_count": "15716", "work_id": "372536", "title": "Old Possum's Book of Practical Cats", "title_without_series": "Old Possum's Book of Practical Cats"}
```
### 2. goodreads_interactions_poetry
2,734,350 interactions

```json
{"user_id": "80d52f5e70f023bd0098ab96599a3530", "book_id": "402128", "review_id": "fbd6a22a155c87a84fba7537f06cc94b", "is_read": true, "rating": 4, "review_text_incomplete": "", "date_added": "Fri Apr 19 08:15:15 -0700 2013", "date_updated": "Fri Apr 19 08:15:15 -0700 2013", "read_at": "", "started_at": ""}
```

### 3. goodreads_reviews_poetry
154,555 detail reviews

```json
{"user_id": "3ca7375dba942a760e53b726c472a7dd", "book_id": "402128", "review_id": "28423ff309bc896c071a8d9df4a10e8a", "rating": 5, "review_text": "I have three younger siblings and we grew up watching the musical Cats. We knew all the songs and attempted to do the dance moves too. I remember we used to get trouble for jumping off the sofa too. When I found out that Cats was based off of poems, I really wanted to read them. I asked for the book for Christmas one year and I read them all that day. The poems are beautifully written and actually tell stories, whereas some poems are just descriptions. I have no idea how T.S, Eliot came up with so creative and brilliant with something as familiar as the family cat. Eliot is a great writer and I would recommend this book to anyone who is looking for a break from all the intense, sophisticated poems/books they are usually reading. This book is fun and is guaranteed to brighten your day!", "date_added": "Tue Jun 12 08:59:04 -0700 2012", "date_updated": "Fri Jun 15 11:41:12 -0700 2012", "read_at": "", "started_at": "", "n_votes": 0, "n_comments": 0}

```

---
# For non-Colab Users (For Linux, MacOS, Windows users)

# 1. Dataset 📀
## 1.1. Download
Download *poetry* datasets from [Goodreads Book Graph Datasets](https://mengtingwan.github.io/data/goodreads.html) in `goodreads` folder.
```
mkdir goodreads
cd goodreads

# Download
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_poetry.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_interactions_poetry.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz # not used now

# unzip
gunzip *.gz
```

Then the datasets are stored as following:
```
cd goodreads
    .
    └── goodreads_books_poetry.json
    └── goodreads_interactions_poetry.json
    └── goodreads_reviews_poetry.json
```

# 2. Environment Setup ⚙️

For venv users (python==3.10.12 recommended)
```
python3.10 -m venv .bookgnn
source .bookgnn/bin/activate
pip3 install -r requirements.txt
```

For conda users (I haven't checked this yet)
```
conda create -n bookgnn python==3.10.12
conda activate bookgnn
pip3 install -r requirements.txt
```


# 3. Preprocess the Dataset 🔥
Preprocess the downloaded datasets. We first only use `goodreads_books_poetry.json` and `goodreads_interactions_poetry.json`. The preprocessed results will be saved in the `datasets` folder.

```
python3 src/preprocess.py

cd datasets
    .
    └── books_poetry.json
    └── interactions_poetry.json
```

# 4. Training 🚀

```
python3 src/training.py
```

# 5. Visualize  (WIP)

```
python3 src/visualize.py
```

# 6. References

[networkx](https://medium.com/@harshkjoya/connecting-the-dots-creating-network-graphs-from-pandas-dataframes-with-networkx-9c4fb60089cf), [Spotify RS](https://medium.com/stanford-cs224w/spotify-track-neural-recommender-system-51d266e31e16)
