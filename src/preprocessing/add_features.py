import re
import preprocessor as twitter_preprocessor
import spacy
from spacy.symbols import ORTH
from tqdm import tqdm
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


def generate_embeddings(df):
    word2vec_model = Word2Vec.load('word2vec.model')

    embeddings_mean = pd.DataFrame()
    rows_to_drop = []
    for i, row in df.iterrows():
        sentence_relevant_embeddings = [word for word in row['text'].split() if word in word2vec_model.wv.key_to_index]
        if sentence_relevant_embeddings:
            embeddings_mean = embeddings_mean.append({'Embeddings': np.mean(word2vec_model.wv[sentence_relevant_embeddings], axis=0)}, ignore_index=True)
        else:
            rows_to_drop.append(i)
    df = df.drop(index=rows_to_drop, axis=0)
    df = pd.merge(df, embeddings_mean, left_index=True, right_index=True)
    return df


def add_boolean_features(df):
    added_features = []

    # Add boolean features
    for col_name, re_ in [
        ('hasqmark', '\?'), ('hasemark', '!'), ('hashashtag', '#'), ('hasurl', 'urlurlurl|http'), 
        ('haspic', 'picpicpic|pic\.twitter\.com|instagr\.am'), ('hasrumor', 'rumour|gossip')
    ]:
        df[col_name] = df['text'].str.contains(re_).astype(int)
        added_features.append(col_name)

    return df, added_features


def add_counter_features(df):
    added_features = []

    neg_words_seq = '|'.join(['not', 'no', 'nobody', 'nothing', 'none', 'never', 'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                                'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn', 'couldn', 'doesn'])
    swear_words_list = []
    with open('src/preprocessing/swearwords.txt', 'r') as f:
        for line in f:
            swear_words_list.append(line.strip().lower())
    swear_words_seq = '|'.join(swear_words_list)

    for col_name, re_ in [
        ('hasnegation', neg_words_seq), ('hasswearwords', swear_words_seq)
    ]:
        df[col_name] = df['text'].str.count(re_)
        added_features.append(col_name)
    
    return df, added_features


def add_ner_feature(df):
    added_features = []

    validNER = ["UNK",
            "PERSON",  # People, including fictional.
            "NORP",  # Nationalities or religious or political groups.
            "FAC",  # Buildings, airports, highways, bridges, etc.
            "ORG",  # Companies, agencies, institutions, etc.
            "GPE",  # Countries, cities, states.
            "LOC",  # Non-GPE locations, mountain ranges, bodies of water.
            "PRODUCT",  # Objects, vehicles, foods, etc. (Not services.)
            "EVENT",  # Named hurricanes, battles, wars, sports events, etc.
            "WORK_OF_ART",  # Titles of books, songs, etc.
            "LAW",  # Named documents made into laws.
            "LANGUAGE",  # Any named language.
            "DATE",  # Absolute or relative dates or periods.
            "TIME",  # Times smaller than a day.
            "PERCENT",  # Percentage, including "%".
            "MONEY",  # Monetary values, including unit.
            "QUANTITY",  # Measurements, as of weight or distance.
            "ORDINAL",  # "first", "second", etc.
            "CARDINAL",  # Numerals that do not fall under another type.
            ]

    ner_vec_list = []
    processed_vec_list = []

    for text in tqdm(df['text'].values):
        text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "$URL$",
                        text.strip())
        twitter_preprocessor.set_options('mentions')
        text = twitter_preprocessor.tokenize(text)

        nlpengine = spacy.load('en_core_web_sm')
        nlpengine.add_pipe('sentencizer')
        for x in ['URL', 'MENTION', 'HASHTAG', 'RESERVED', 'EMOJI', 'SMILEY', 'NUMBER', ]:
            nlpengine.tokenizer.add_special_case(f'${x}$', [{ORTH: f'${x}$'}])

        NERvec = []
        processed_chunk = ""
        doc = nlpengine(text)
        for sentence in doc.sents:
            for w in sentence:
                # Some phrases are automatically tokenized by Spacy
                # i.e. New York, in that case we want New_York in our dictionary
                word = "_".join(w.text.split())
                if word.isspace() or word == "":
                    continue

                output = word
                output = output.replace("n't", "not")
                processed_chunk += "%s " % (output)

                try:
                    NERvec.append(validNER.index(w.ent_type_))
                except ValueError:
                    NERvec.append(validNER.index('UNK'))

            processed_chunk += "[EOS]" + "\n"
            NERvec.append(0)

        processed_chunk = processed_chunk.strip()
        assert len(processed_chunk.split()) == len(NERvec)
        ner_vec_list.append(NERvec)
        processed_vec_list.append(processed_chunk)

    df['NERvec'] = ner_vec_list
    df['spacy_processed_text'] = processed_vec_list
    added_features.append('NERvec')

    return df, added_features
