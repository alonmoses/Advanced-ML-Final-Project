import pandas as pd
import numpy as np

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
        print(drop_columns)
        train_df.drop(drop_columns, axis=1, inplace=True)
        test_df.drop(drop_columns, axis=1, inplace=True)

        return train_df, test_df

    def remove_features(self, df, rho=0.4):
        corr_matrix = df.corr().abs() # correlation matrix
        upper_corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) # upper triangle of matrix
        features_to_drop = [column for column in upper_corr_matrix.columns if any(upper_corr_matrix[column] > rho)] # get relevant features- TODO: NEED TO FIX
        return features_to_drop


def main():
    dataset = ProcessData("/aml/data/dreaddit/dreaddit-train.csv", "/aml/data/dreaddit/dreaddit-test.csv")
    train_df, test_df = dataset.load_data_to_dataframe()
    print(train_df.head())

if __name__ == '__main__':
    main()