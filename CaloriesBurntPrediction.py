import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("calories.csv")

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X = df.drop(columns=['User_ID', 'Calories'])
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}\n")

feature_importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("Feature Importance for Calories Burnt Prediction:")
print(importance_df.to_string(index=False))  # แสดงเป็นตาราง

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

error_metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}
sns.barplot(x=list(error_metrics.keys()), y=list(error_metrics.values()), ax=ax[0], hue=list(error_metrics.keys()), legend=False, palette="Blues_r")
ax[0].set_title("Model Error Metrics", fontsize=14)
ax[0].set_ylabel("Value")

sns.barplot(y=importance_df["Feature"], x=importance_df["Importance"], ax=ax[1], hue=importance_df["Feature"], legend=False, palette="coolwarm")
ax[1].set_title("Feature Importance for Calories Prediction", fontsize=14)
ax[1].set_xlabel("Importance")

plt.tight_layout()
plt.show()

joblib.dump(model, "calories_model.pkl")
joblib.dump(scaler, "scaler.pkl")

loaded_model = joblib.load("calories_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

X_sample = X_test[:5]  
y_sample_pred = loaded_model.predict(X_sample)

print("Examples of predicted values:", y_sample_pred)