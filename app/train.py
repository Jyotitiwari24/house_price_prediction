import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load dataset
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["price"] = dataset.target

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save model
with open("model/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
