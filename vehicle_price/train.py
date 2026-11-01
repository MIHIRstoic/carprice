import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from .data import load_data, validate
from .features import add_features
from .config import TARGET, MODEL_PATH, RANDOM_SEED

def main():
    df = load_data()
    validate(df)
    df = add_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)])
    model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(random_state=RANDOM_SEED))])
    model.fit(Xtr, ytr)
    import os; os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("trained")

if __name__ == "__main__":
    main()
