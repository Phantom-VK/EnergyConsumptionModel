
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('combined_data.csv')
features = ['Temperature_C', 'Is_Weekend', 'Num_Devices', 'Avg_Power_Rating', 'Num_Events']
X = df[features]
y = df['Total_Energy_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation MSE Scores:", -scores)
print("Average Cross-Validation MSE:", -scores.mean())

# Display model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Energy Consumption (kWh)')
plt.ylabel('Predicted Energy Consumption (kWh)')
plt.title('Actual vs Predicted Energy Consumption')
plt.savefig('prediction_results.png')

# Visualize feature importance
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
plt.figure(figsize=(10, 6))
plt.barh(features, rf_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (RandomForest)')
plt.savefig('feature_importance_rf.png')

# Saving model here
import pickle
with open('energy_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved to 'energy_model.pkl'")