import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from configparser import ConfigParser
import os

# cfg = ConfigParser()
# RAW_DATA_PATH = os.path.join(cfg.root_path, "raw")
# PROCESSED_DATA_PATH = os.path.join(cfg.root_path, "processed")

COLUMNS = {
    # "Age": ["Age"] # Drop becaue ethiopia neg has no age
    "Sex": ["Sex"],
    "HCT": ["HCT"],
    "HGB": ["HGB"],
    "MCH": ["MCH"],
    "MCHC": ["MCHC"],
    "MCV": ["MCV"],
    "RBC": ["RBC"],
    "HCT": ["HCT"],
    "HGB": ["HGB"],
    "MCH": ["MCH"],
    "MCHC": ["MCHC"],
    "MCV": ["MCV"],
    "RBC": ["RBC"],
    "WBC": ["WBC"],
    "PLT": ["PLT", "PLT1"],
    "NE": ["NE"],
    "LY": ["LY"],
    "MO": ["MO"],
    "EO": ["EO"],
    "BA": ["BA"],
    "target": ["Target", "PCR "],
}

def group_datasets()->list[pd.DataFrame]:
    
    df_brazil_1 = pd.read_excel("experiments/data/tabular/raw/all_data_processed_v3.xlsx", "Lab1", header=1, index_col="id")
    df_brazil_2 = pd.read_excel("experiments/data/tabular/raw/all_data_processed_v3.xlsx", "Hosp1", header=1, index_col="id")
    df_brazil_3 = pd.read_excel("experiments/data/tabular/raw/all_data_processed_v3.xlsx", "Hosp2", header=1, index_col="id")

    df_italy_1 = pd.read_excel("experiments/data/tabular/raw/desio_cbc.xls", index_col="Unnamed: 0")
    df_italy_2 = pd.read_excel("experiments/data/tabular/raw/bergamo_cbc.xls", index_col="Unnamed: 0")
    df_italy_3 = pd.read_excel("experiments/data/tabular/raw/HSR_novembre.xlsx", index_col="id")

    df_spain_1 = pd.read_excel("experiments/data/tabular/raw/CBC Italy Octubre (sent 24-12-2020).xlsx", "PCR - n=42")
    df_spain_2 = pd.read_excel("experiments/data/tabular/raw/CBC Italy Octubre (sent 24-12-2020).xlsx", "PCR + asymptomatic n=18")
    df_spain_3 = pd.read_excel("experiments/data/tabular/raw/CBC Italy Octubre (sent 24-12-2020).xlsx", "PCR + n=60")
    df_spain = pd.concat([df_spain_1, df_spain_2, df_spain_3]).reset_index(drop=True)
    
    df_ethiopia_pos = pd.read_excel("experiments/data/tabular/raw/Etiopia 200 COVID +.xlsx", index_col="ID")
    df_ethiopia_neg = pd.read_excel("experiments/data/tabular/raw/cbc data_anna copy.xlsx", index_col="Patient ID")
    df_ethiopia_pos["target"] = 1
    df_ethiopia_neg["target"] = 0
    df_ethiopia = pd.concat([df_ethiopia_pos, df_ethiopia_neg]).reset_index(drop=True)
    
    return {
        "brazil_1": df_brazil_1,
        "brazil_2": df_brazil_2,
        "brazil_3": df_brazil_3,
        "italy_1": df_italy_1,
        "italy_2": df_italy_2,
        "italy_3": df_italy_3,
        "spain": df_spain,
        "ethiopia": df_ethiopia
    }
    
# italy 1, 2, 3 sex 0-1
# brazil M,F
# spain 1,2
# ethiopia M,F, 1,2

def rename_sex_values(data):
    """Rename values for specific datasets. We need to map va"""
    for name, df in data.items():
        if name in ["spain", "ethiopia"]:
            df['Sex'] = df['Sex']
            df['Sex'] = df['Sex'].map({
                1: 0,
                2: 1,
                "M": 0,
                "F": 1,
            })
    for name, df in data.items():
        if name in ["brazil_1", "brazil_2", "brazil_3"]:
            df['Sex'] = df['Sex'].map({
                "M": 0,
                "F": 1
            })
    return data
    
def rename_columns(data):

    for k, v in data.items():
        for official_col_name in COLUMNS.keys():
            if official_col_name not in v.columns:
                print(f"Missing {official_col_name} in {k}")
                for col_name in v.columns:
                    if col_name in COLUMNS[official_col_name]:
                        v = v.rename(columns={col_name: official_col_name})
                        print(f"Renamed {col_name} to {official_col_name} in {k}")
                        
        data[k] = v[COLUMNS.keys()].sample(frac=1)
        data[k] = data[k].reset_index(drop=True)
    
    data["brazil_1"]["WBC"] /= 1000
    data["brazil_2"]["WBC"] /= 1000
    data["brazil_3"]["WBC"] /= 1000
    
    return data

def ensure_numeric_format(data):
    for k, v in data.items():
        v = v.apply(pd.to_numeric, errors='coerce')
        data[k] = v
    return data

def remove_outliers_from_all_datasets(data, method='strict_iqr', threshold=10.0):
    """Remove significant outliers from all datasets
    
    Args:
        data: Dictionary containing datasets
        method: 'strict_iqr' (1.5*IQR), 'z_score' (Z-score), 'modified_z' (Modified Z-score), or 'percentile'
        threshold: Threshold value (3.0 for Z-score, 3.5 for modified Z-score, 10.0 for strict_iqr)
    """
    for dataset_name, df in data.items():
        df = df.copy()
        print(f"\n{dataset_name} dataset before outlier removal: {len(df)} rows")
        
        # Get numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            col_data = df[col].dropna()
            
            if method == 'strict_iqr':
                # Conservative: 1.5*IQR (standard IQR method)
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                100333.0
            elif method == 'z_score':
                # Z-score method (threshold typically 3.0)
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > threshold
                
            elif method == 'modified_z':
                # Modified Z-score using median (more robust)
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                if mad == 0:
                    col_outliers = pd.Series([False] * len(df), index=df.index)
                else:
                    modified_z_scores = 0.6745 * (df[col] - median) / mad
                    col_outliers = np.abs(modified_z_scores) > threshold
                
            elif method == 'percentile':
                # Remove only extreme percentiles (e.g., bottom 1% and top 1%)
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            outlier_mask = outlier_mask | col_outliers
            print(f"  Column {col}: {col_outliers.sum()} outliers detected")
        
        # Remove rows with outliers
        data[dataset_name] = df[~outlier_mask].reset_index(drop=True)
        print(f"{dataset_name} dataset after outlier removal: {len(data[dataset_name])} rows")
        print(f"Total outliers removed: {outlier_mask.sum()}")
    
    return data
        
        
def save_data(data):
    for k, v in data.items():
        v = v.dropna()
        train, test = train_test_split(v, test_size=0.2, random_state=42, stratify=v["target"])
        
        train.to_csv(f"experiments/data/tabular/processed/train/{k}.csv", index=False)
        test.to_csv(f"experiments/data/tabular/processed/test/{k}.csv", index=False)
        
def merge_italy_and_brazil_datasets():
    root_path = "experiments/data/tabular/processed/"
    # italy_files = ["italy_1.csv", "italy_2.csv", "italy_3.csv"]
    brazil_files = ["brazil_1.csv", "brazil_2.csv"]
    # italy_train_dfs =[pd.read_csv(os.path.join(root_path, "train", f)) for f in italy_files]
    # italy_test_dfs = [pd.read_csv(os.path.join(root_path, "test", f)) for f in italy_files]
    brazil_train_dfs = [pd.read_csv(os.path.join(root_path, "train", f)) for f in brazil_files]
    brazil_test_dfs = [pd.read_csv(os.path.join(root_path, "test", f)) for f in brazil_files]
    # italy_train = pd.concat(italy_train_dfs).sample(frac=1).reset_index(drop=True)
    # italy_test = pd.concat(italy_test_dfs).sample(frac=1).reset_index(drop=True)
    brazil_train = pd.concat(brazil_train_dfs).sample(frac=1).reset_index(drop=True)
    brazil_test = pd.concat(brazil_test_dfs).sample(frac=1).reset_index(drop=True)
    
    # italy_train.to_csv(f"experiments/data/tabular/processed/train/italy.csv", index=False)
    # italy_test.to_csv(f"experiments/data/tabular/processed/test/italy.csv", index=False)
    brazil_train.to_csv(f"experiments/data/tabular/processed/train/brazilv2.csv", index=False)
    brazil_test.to_csv(f"experiments/data/tabular/processed/test/brazilv2.csv", index=False)
    
    
    
def preprocess_data():
    data = group_datasets()
    data = rename_columns(data)
    data = rename_sex_values(data)
    data = ensure_numeric_format(data)
    # Remove outliers from all datasets using standard IQR method
    data = remove_outliers_from_all_datasets(data, method='strict_iqr')
    save_data(data)
    merge_italy_and_brazil_datasets()
    
if __name__ == "__main__":
    preprocess_data()