import pandas as pd
import numpy as np
import json
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, TensorDataset

from gensim.models import Word2Vec

from models import LogisticRegressionExecute

class ProcessData:
    def __init__(self, model, word2vec_model, config):
        self.train_data = config[model]['train_data']
        self.test_data = config[model]['test_data'] 
        self.word2vec_model = Word2Vec.load(f'{word2vec_model}.model')
        self.config = config[model]

    def load_data_to_dataframe(self):
        # load csv to dataframe
        train_df = pd.read_csv(self.train_data)
        test_df = pd.read_csv(self.test_data)

        self.generate_embeddings(train_df)
        
        # drop columns based on filter_features() logic
        columns_to_keep = self.filter_features(train_df) + ['Embeddings']
        train_df = train_df[train_df['confidence'] > 0.8]
        train_df = train_df[columns_to_keep]
        test_df = test_df[columns_to_keep]

        return train_df, test_df

    def filter_features(self, df:pd.DataFrame, rho:float=0.4):
        corr_matrix = df.corr().abs()['label'] # correlation of each featur with the labels column
        features_to_keep = [column for column in corr_matrix.index if corr_matrix[column] > rho]
        return features_to_keep

    def generate_embeddings(self, df):
        df['Embeddings'] = ""
        for i, row in df.iterrows():
            sentence_relevant_embeddings = [word for word in row['text'].split() if word in self.word2vec_model.wv.key_to_index]
            embeddings_mean = np.mean(self.word2vec_model.wv[sentence_relevant_embeddings], axis=0)
            df.iloc[i]['Embeddings'] = embeddings_mean

    def create_dataloaders(self):
        train_df, test_df = self.load_data_to_dataframe()

        # create datasets
        train_features = torch.tensor(train_df.drop(columns=['label']).values)
        train_labels = torch.tensor(train_df['label'].values)
        train = TensorDataset(train_features, train_labels)
        self.train_dl = DataLoader(train, batch_size=self.config['hyperparameters']['batch_size'], shuffle=True)

        test_features = torch.tensor(test_df.drop(columns=['label']).values)
        test_labels = torch.tensor(test_df['label'].values)
        test = TensorDataset(test_features, test_labels)
        self.test_dl = DataLoader(test, batch_size=self.config['hyperparameters']['batch_size'], shuffle=True)  

def main(kwargs, config):
    model = kwargs['model']
    word2vec_model = kwargs['word2vec']
    dataset = ProcessData(model, word2vec_model, config)
    dataset.create_dataloaders()
    lr_executer = LogisticRegressionExecute(config[model])
    lr_executer.fit(dataset.train_dl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Choose the model to train', required=False, choices=['logistic_regression'], default='logistic_regression')
    parser.add_argument('-wm', '--word2vec', help='Choose the model to train', required=False, choices=['word2vec'], default='word2vec')
    args = parser.parse_args()
    kwargs = vars(args)

    with open('src/config.json', 'r') as f:
        config = json.load(f)

    main(kwargs, config)