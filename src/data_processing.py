import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from add_features import add_boolean_features, add_counter_features, add_ner_feature

class ProcessData:
    def __init__(self, model, word2vec_model, config):
        self.train_data = config[model]['train_data']
        self.test_data = config[model]['test_data'] 
        self.word2vec_model = Word2Vec.load(f'{word2vec_model}.model')
        self.config = config[model]

    def load_data_to_dataframe(self, use_saved=False):
        if use_saved:
            train_df = pd.read_csv('train_df.csv')
            test_df = pd.read_csv('test_df.csv')
        else:
            # keep_cols = ['id','subreddit','post_id','sentence_range','text','label','confidence','social_timestamp']
            keep_cols = None
            # load csv to dataframe
            train_df = pd.read_csv(self.train_data, usecols=keep_cols, nrows=100)
            test_df = pd.read_csv(self.test_data, usecols=keep_cols, nrows=100)

            # generate embeddings for each sentence
            train_df = self.generate_embeddings(train_df)
            test_df = self.generate_embeddings(test_df)

            # Add features
            train_df, bool_features_list = add_boolean_features(train_df)
            test_df, _ = add_boolean_features(test_df)
            train_df, count_features_list = add_counter_features(train_df)
            test_df, _ = add_counter_features(test_df)
            train_df, ner_features_list = add_ner_feature(train_df)
            test_df, _ = add_ner_feature(test_df)

            # drop columns based on filter_features() logic
            columns_to_keep = self.filter_features(train_df) + ['Embeddings'] + bool_features_list + count_features_list + ner_features_list
            train_df = train_df[train_df['confidence'] > 0.8]
            train_df = train_df[columns_to_keep]
            test_df = test_df[columns_to_keep]

            # Save data to csv
            train_df.to_csv('train_df.csv', index=False)
            test_df.to_csv('test_df.csv', index=False)

        return train_df, test_df

    def filter_features(self, df:pd.DataFrame, rho:float=0.4):
        corr_matrix = df.corr().abs()['label'] # correlation of each featur with the labels column
        features_to_keep = [column for column in corr_matrix.index if corr_matrix[column] > rho]
        return features_to_keep

    def generate_embeddings(self, df):
        embeddings_mean = pd.DataFrame()
        rows_to_drop = []
        for i, row in df.iterrows():
            sentence_relevant_embeddings = [word for word in row['text'].split() if word in self.word2vec_model.wv.key_to_index]
            if sentence_relevant_embeddings:
                embeddings_mean = embeddings_mean.append({'Embeddings': np.mean(self.word2vec_model.wv[sentence_relevant_embeddings], axis=0)}, ignore_index=True)
            else:
                rows_to_drop.append(i)
        df = df.drop(index=rows_to_drop, axis=0)
        df = pd.merge(df, embeddings_mean, left_index=True, right_index=True)
        return df

    def generate_train_features(self, df):
        features_arr = []
        for idx, row in df.iterrows():
            features = row[:-1].to_numpy()
            data = np.concatenate((features, row[-1]))
            features_arr.append(data.astype(np.float32))
        return torch.tensor(features_arr)

    def create_dataloaders(self, use_saved=False):
        train_df, test_df = self.load_data_to_dataframe(use_saved=use_saved)

        # create datasets
        train_features = self.generate_train_features(train_df.drop(columns=['label']))
        train_labels = torch.tensor(train_df['label'].values)
        train = TensorDataset(train_features, train_labels)
        self.train_dl = DataLoader(train, batch_size=self.config['hyperparameters']['batch_size'], shuffle=True)

        test_features = self.generate_train_features(test_df.drop(columns=['label']))
        test_labels = torch.tensor(test_df['label'].values)
        test = TensorDataset(test_features, test_labels)
        self.test_dl = DataLoader(test, batch_size=self.config['hyperparameters']['batch_size'], shuffle=True)  
