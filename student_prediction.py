import pandas as pd

data = pd.read_csv("student_data_expanded.csv")
data.columns=data.columns.str.strip()

print(data.head())

X = data[[
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score"
]]

y = data["final_marks"]

print("X:")
print(X)

print("\ny:")
print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:")
print(X_train)

print("\nTesting data:")
print(X_test)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed!")
predictions = model.predict(X_test)

print("Actual Marks:")
print(y_test.values)

print("Predicted Marks:")
print(predictions)
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)
new_student = pd.DataFrame(
    [[6, 85, 78, 80]],
    columns=[
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
)

result = model.predict(new_student)

print("Predicted Final Marks:", result[0])
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")

plt.show()
from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predict
rf_predictions = rf_model.predict(X_test)

# Calculate performance
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- Random Forest ---")
print("MAE:", rf_mae)
print("R2 Score:", rf_r2)
new_student = pd.DataFrame(
    [[6, 85, 78, 80]],
    columns=[
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
)

result = model.predict(new_student)

print("Predicted Final Marks:", result[0])
import joblib

joblib.dump(model, "student_performance_model.pkl")

print("Model saved successfully!")