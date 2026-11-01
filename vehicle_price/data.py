import pandas as pd
from .config import DATA_PATH, TARGET

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    assert TARGET in df.columns
    return df

def validate(df):
    assert df.shape[0] > 0
    return True
