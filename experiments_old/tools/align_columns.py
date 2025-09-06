import pandas as pd
import os

def align_columns(ref_path, df_path):
    ref_df = pd.read_csv(ref_path, index_col=0)
    df = pd.read_csv(df_path, index_col=0)
    df = df[ref_df.columns]
    df.to_csv(df_path, index=True)

if __name__=='__main__':
    ref_path = 'experiments/data/data_with_age/train/Brazil-1.csv'

    align_columns(ref_path, 'experiments/data/data_with_age/train/Italy-4.csv')
    align_columns(ref_path, 'experiments/data/data_with_age/train/Poland.csv')
    align_columns(ref_path, 'experiments/data/data_with_age/test/Italy-4.csv')
    align_columns(ref_path, 'experiments/data/data_with_age/test/Poland.csv')

    ref_path = 'experiments/data/processed/train/Brazil-1.csv'
    align_columns(ref_path, 'experiments/data/data_with_age/train/Ethiopia.csv')
    align_columns(ref_path, 'experiments/data/data_with_age/test/Ethiopia.csv')