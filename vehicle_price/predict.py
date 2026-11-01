import joblib, pandas as pd
from .config import MODEL_PATH
from .features import add_features

def predict(df):
    model = joblib.load(MODEL_PATH)
    df = add_features(df)
    return model.predict(df)
