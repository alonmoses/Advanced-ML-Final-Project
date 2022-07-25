import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class ProcessData:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def load_data_to_dataframe(self):
        # load csv to dataframe
        train_df = pd.read_csv(self.train_data)
        test_df = pd.read_csv(self.test_data)

        # drop columns based on remove_features() logic
        drop_columns = self.remove_features(train_df)
        train_df = train_df[train_df['confidence'] > 0.8]
        train_df.drop(drop_columns, axis=1, inplace=True)
        test_df.drop(drop_columns, axis=1, inplace=True)

        return train_df, test_df

    def remove_features(self, df, rho=0.4):
        corr_matrix = df.corr().abs()['label'] # correlation matrix
        features_to_drop = [column for column in corr_matrix.index if corr_matrix[column] < rho] # get relevant features- TODO: NEED TO FIX
        return features_to_drop

    def create_dataloaders(self):
        train_df, test_df = self.load_data_to_dataframe()

        # create datasets
        labels = pd.DataFrame(train_df['label'])
        print(len(train_df), len(labels))
        train = TensorDataset(train_df, labels)
        train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
        return train_loader


def main():
    dataset = ProcessData("/aml/data/dreaddit/dreaddit-train.csv", "/aml/data/dreaddit/dreaddit-test.csv")
    train_dl = dataset.create_dataloaders()
    print(train_dl)

if __name__ == '__main__':
    main()