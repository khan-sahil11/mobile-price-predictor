# Trains 4 models, picks the best, saves it as Trained Model/model.joblib
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = Path(__file__).parent.parent
df = pd.read_csv(BASE / "Dataset" / "clean_phones.csv").drop_duplicates()

NUM = ["RAM_GB", "Storage_GB", "Screen_inch", "Battery_mAh",
       "Main_Camera_MP", "Num_Cameras", "Is_5G"]      # number inputs
CAT = ["Brand", "OS"]                                 # text inputs

# Limit extreme values using IQR (outlier handling)
caps = {}
for c in ["Screen_inch", "Battery_mAh", "Main_Camera_MP"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    caps[c] = [float(lo), float(hi)]
    df[c] = df[c].clip(lo, hi)

X, y = df[NUM + CAT], df["Price_INR"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build(model):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    # Train on log(price), then convert predictions back with exp -> price can never be negative
    wrapped = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("prep", prep), ("model", wrapped)])


candidates = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10,
                                           min_samples_leaf=2, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

rows, fitted = [], {}
for name, model in candidates.items():
    pipe = build(model)
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=5,
                               scoring="neg_root_mean_squared_error").mean()
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    mse = mean_squared_error(y_test, p)
    rows.append({"Model": name, "CV_RMSE": cv_rmse, "MAE": mean_absolute_error(y_test, p),
                 "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_test, p),
                 "Train_R2": r2_score(y_train, pipe.predict(X_train))})
    fitted[name] = pipe

table = pd.DataFrame(rows).set_index("Model").sort_values("CV_RMSE")
print(table.round(3))
best_name = table.index[0]
print("Final model:", best_name)

out = BASE / "Trained Model"
out.mkdir(exist_ok=True)
joblib.dump(fitted[best_name], out / "model.joblib", compress=3)
print("Size MB:", round((out / "model.joblib").stat().st_size / 1e6, 2))

info = {"model_name": best_name, "rmse": float(table.loc[best_name, "RMSE"]),
        "r2": float(table.loc[best_name, "R2"]), "min_price": float(y.min()),
        "brands": sorted(df["Brand"].unique().tolist()), "caps": caps}
(out / "metrics.json").write_text(json.dumps(info, indent=2))
print("Saved model.joblib and metrics.json")