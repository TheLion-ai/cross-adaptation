import os
import pandas as pd


def remove_age_from_df(old_path, new_path):
    df = pd.read_csv(old_path, index_col=0)
    if 'Age' in df.columns:
        df = df.drop(columns=['Age'], axis=1)
    df.to_csv(new_path, index=True)


if __name__=='__main__':
    train_old_path = 'experiments/data/data_with_age/train'
    train_new_path = 'experiments/data/processed/train'
    for f in os.listdir(train_old_path):
        remove_age_from_df(os.path.join(train_old_path, f), os.path.join(train_new_path, f))

    test_old_path = 'experiments/data/data_with_age/test'
    test_new_path = 'experiments/data/processed/test'
    for f in os.listdir(test_old_path):
        remove_age_from_df(os.path.join(test_old_path, f), os.path.join(test_new_path, f))
    