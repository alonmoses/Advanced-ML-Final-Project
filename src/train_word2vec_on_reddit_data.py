import datetime as dt
from tqdm import tqdm
import praw
from psaw import PushshiftAPI
from gensim.models import Word2Vec

r = praw.Reddit(client_id="27WEcCXx6eUBFWTjd0AXBg",
                client_secret="NJUZ56HGxvyiTdsf0Xxu9L0e8Yz46w",
                password="^Iuktxzun2652",
                user_agent="USERAGENT",
                username="Disastrous_Egg6551 ")


api = PushshiftAPI(r)

start_epoch = int(dt.datetime(2017, 1, 1).timestamp())
end_epoch = int(dt.datetime(2018, 11, 19).timestamp())

subreddit_list = ['domesticviolence', 'anxiety', 'stress', 'almosthomeless', 'assistance', 'food_pantry', 'homeless', 'ptsd', 'relationships'] # NOTE: subreddit r/survivorsofabuse is private and not accesible 
texts = []
for subreddit in tqdm(subreddit_list):
    results = list(api.search_submissions(before=end_epoch, after=start_epoch,
                                          subreddit=subreddit,
                                          filter=['url','author', 'title', 'subreddit']))
    
    print(f"Done downloading reddit posts from subreddit: '{subreddit}'")
    for result in tqdm(results):
        texts.append(result.title.split(' '))

wv_model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=4)
wv_model.build_vocab(texts, progress_per=10000)
wv_model.train(texts, total_examples=wv_model.corpus_count, epochs=50, report_delay=1)
wv_model.save("word2vec.model")



