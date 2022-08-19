import numpy as np
import pandas as pd
from os import path
import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessing.add_features import generate_embeddings, add_ner_feature


def add_vec_features(df):
    # generate embeddings for each sentence
    df = generate_embeddings(df)

    # generate NER vectors
    df, _ = add_ner_feature(df)

    return df


def generate_train_features(df):
    features_arr = []
    non_embedding_cols = list(df.columns[~df.columns.isin(['Embeddings', 'text', 'NERvec', 'spacy_processed_text'])])
    for _, row in df.iterrows():
        features = row[non_embedding_cols].to_numpy()
        data = np.concatenate((features, row['Embeddings']))
        features_arr.append(data.astype(np.float32))
    return torch.tensor(features_arr)


def create_dataloaders(config, raw_data_path):
    # Load DataFrames
    train_df = pd.read_csv(path.join(raw_data_path, 'train_df.csv'))
    test_df = pd.read_csv(path.join(raw_data_path, 'test_df.csv'))

    # Add vec features
    train_df = add_vec_features(train_df)
    test_df = add_vec_features(test_df)

    # create datasets
    train_features = generate_train_features(train_df.drop(columns=['label']))
    train_labels = torch.tensor(train_df['label'].values)
    train = TensorDataset(train_features, train_labels)
    train_dl = DataLoader(train, batch_size=config['hyperparameters']['batch_size'], shuffle=True)

    test_features = generate_train_features(test_df.drop(columns=['label']))
    test_labels = torch.tensor(test_df['label'].values)
    test = TensorDataset(test_features, test_labels)
    test_dl = DataLoader(test, batch_size=config['hyperparameters']['batch_size'], shuffle=True)  

    return train_dl, test_dl


