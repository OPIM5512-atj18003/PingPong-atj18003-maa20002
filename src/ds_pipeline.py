# Read in the california dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
print(df.shape)


# Train / Test Split (features + target)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)


# Scale features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled X_train:", X_train_scaled.shape)
print("Scaled X_test:", X_test_scaled.shape)


# Train MLP Regressor
mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(50,25),
                   max_iter=500,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) # important!

mlp.fit(X_train_scaled, y_train)

print("MLP training complete.")
print("Train score (R^2):", mlp.score(X_train_scaled, y_train))
print("Test score (R^2):", mlp.score(X_test_scaled, y_test))


# Box Plot
plt.figure(figsize=(6, 4))
df['MedHouseVal'].plot.box()
plt.title('Box plot of Median House Value')
plt.ylabel('Median House Value')

# Save into repo_root/figures
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.tight_layout()
plt.savefig(os.path.join(repo_root, "figures", "med_house_val_boxplot.png"))
plt.close()

# Training Predictions
y_pred_train = mlp.predict(X_train_scaled)

# Predicted vs Actual Scatter Plot - Training Set
plt.figure(figsize=(6,6))
plt.scatter(y_train, y_pred_train, alpha=0.3, s=10)
lo = min(np.min(y_train), np.min(y_pred_train))
hi = max(np.max(y_train), np.max(y_pred_train))
plt.plot([lo, hi], [lo, hi], linewidth=1, color='red')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs. Actual — Train")
plt.tight_layout()
plt.savefig(os.path.join(repo_root, "figures", "predicted_vs_actual_train.png"))
plt.show()
