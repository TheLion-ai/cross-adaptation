import pandas as pd
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
    return data
        
        
def save_data(data):
    for k, v in data.items():
        v = v.dropna()
        train, test = train_test_split(v, test_size=0.2, random_state=42, stratify=v["target"])
        
        train.to_csv(f"experiments/data/tabular/processed/train/{k}.csv", index=False)
        test.to_csv(f"experiments/data/tabular/processed/test/{k}.csv", index=False)
        
def merge_italy_and_brazil_datasets():
    root_path = "experiments/data/tabular/processed/"
    italy_files = ["italy_1.csv", "italy_2.csv", "italy_3.csv"]
    brazil_files = ["brazil_1.csv", "brazil_2.csv", "brazil_3.csv"]
    italy_train_dfs =[pd.read_csv(os.path.join(root_path, "train", f)) for f in italy_files]
    italy_test_dfs = [pd.read_csv(os.path.join(root_path, "test", f)) for f in brazil_files]
    brazil_train_dfs = [pd.read_csv(os.path.join(root_path, "train", f)) for f in italy_files]
    brazil_test_dfs = [pd.read_csv(os.path.join(root_path, "test", f)) for f in brazil_files]
    italy_train = pd.concat(italy_train_dfs).sample(frac=1).reset_index(drop=True)
    italy_test = pd.concat(italy_test_dfs).sample(frac=1).reset_index(drop=True)
    brazil_train = pd.concat(brazil_train_dfs).sample(frac=1).reset_index(drop=True)
    brazil_test = pd.concat(brazil_test_dfs).sample(frac=1).reset_index(drop=True)
    
    italy_train.to_csv(f"experiments/data/tabular/processed/train/italy.csv", index=False)
    italy_test.to_csv(f"experiments/data/tabular/processed/test/italy.csv", index=False)
    brazil_train.to_csv(f"experiments/data/tabular/processed/train/brazil.csv", index=False)
    brazil_test.to_csv(f"experiments/data/tabular/processed/test/brazil.csv", index=False)
    
    
    
def preprocess_data():
    data = group_datasets()
    data = rename_columns(data)
    data = rename_sex_values(data)
    data = ensure_numeric_format(data)
    save_data(data)
    merge_italy_and_brazil_datasets()
    
if __name__ == "__main__":
    preprocess_data()