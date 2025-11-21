import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

data = pd.read_csv(r"tmdb-box-office-data/train_processed.csv")

y = data["revenue_log"]

X = data.drop(columns=["revenue_log", "id", "revenue"])
X = X.select_dtypes(include=["number", "bool"])

print("Number of features used:", X.shape[1])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print("XGBoost RMSLE:", rmsle)
