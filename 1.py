import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("Students Performance .csv")

# Step 2: Clean column names
df.columns = [col.strip().replace(" ", "_") for col in df.columns]

# Step 3: Map Attendance to numeric %
attendance_map = {'Always': 100, 'Sometimes': 75, 'Rarely': 50, 'Never': 25}
df['Attendance'] = df['Attendance'].map(attendance_map)

# Step 4: Create derived features
df['Task_Completion_Rate'] = df['Project_work'].map({'Yes': 0.9, 'No': 0.4})
df['Feedback_Score'] = (
    df['Listening_in_Class'].map({'Yes': 8, 'No': 5}) +
    df['Reading'].map({'Yes': 8, 'No': 5})
) / 2

# Step 5: Map Grades to numeric performance scores
grade_mapping = {
    'AA': 95, 'BA': 85, 'BB': 75, 'CB': 65,
    'CC': 60, 'DC': 55, 'DD': 50, 'FD': 45, 'FF': 40
}
df['Final_Performance_Score'] = df['Grade'].map(grade_mapping)

# Step 6: Drop rows with missing data
df.dropna(subset=['Task_Completion_Rate', 'Feedback_Score', 'Attendance', 'Final_Performance_Score'], inplace=True)

# Step 7: Select features and target
X = df[['Task_Completion_Rate', 'Feedback_Score', 'Attendance']]
y = df['Final_Performance_Score']

# Step 8: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 11: Predict
y_pred = model.predict(X_test_scaled)

# Step 12: Evaluate model
print("\n📊 Model Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Step 13: Visualization
plt.figure(figsize=(8, 5))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
plt.xlabel("Actual Performance Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Intern Performance")
plt.grid(True)
plt.tight_layout()
plt.show()
