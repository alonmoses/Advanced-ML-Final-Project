from os import path
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.add_features import add_boolean_features, add_counter_features

NUM_EXAMPLES = 20 # None (use for full run)


def filter_features(df:pd.DataFrame, rho:float=0.4):
        corr_matrix = df.corr().abs()['label'] # correlation of each featur with the labels column
        features_to_keep = [column for column in corr_matrix.index if corr_matrix[column] > rho]
        return features_to_keep


def prep_dreaddit_pipeline():
    # Load dreaddit raw data
    raw_data_path = path.join('data', 'dreaddit')
    # load csv to dataframe
    train_df = pd.read_csv(path.join(raw_data_path, 'dreaddit-train.csv'), nrows=NUM_EXAMPLES)
    test_df = pd.read_csv(path.join(raw_data_path, 'dreaddit-test.csv'), nrows=NUM_EXAMPLES)

    # Add features
    train_df, bool_features_list = add_boolean_features(train_df)
    test_df, _ = add_boolean_features(test_df)
    train_df, count_features_list = add_counter_features(train_df)
    test_df, _ = add_counter_features(test_df)
    
    # drop columns based on filter_features() logic
    columns_to_keep = filter_features(train_df) + bool_features_list + count_features_list + ['text']
    train_df = train_df[train_df['confidence'] > 0.8]
    train_df = train_df[columns_to_keep]
    test_df = test_df[columns_to_keep]

    # Split train to train and dev
    final_train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # Save data to csv
    final_train_df.to_csv(path.join(raw_data_path, 'train_df.csv'), index=False)
    dev_df.to_csv(path.join(raw_data_path, 'dev_df.csv'), index=False)
    test_df.to_csv(path.join(raw_data_path, 'test_df.csv'), index=False)

            
if __name__ == '__main__':
    prep_dreaddit_pipeline()