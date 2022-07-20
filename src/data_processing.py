import pandas as pd

class ProcessData:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def load_data_to_dataframe(self):
        chosen_columns = ['text', 'subreddit', 'social_timestamp', 'confidence', 'label'] ## Need to choose columns in more sophosticated way, as presented in the paper
        train_df = pd.read_csv(self.train_data, usecols=chosen_columns)
        test_df = pd.read_csv(self.test_data, usecols=chosen_columns)
        return train_df, test_df


def main():
    dataset = ProcessData("/aml/data/dreaddit/dreaddit-train.csv", "../data/dreaddit-test.csv")
    train_df, test_df = dataset.load_data_to_dataframe()


if __name__ == '__main__':
    main()