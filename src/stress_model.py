import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("data/stress_data.csv")

# Separate features and labels
X = data[["heart_rate", "sleep_hours", "workload_hours", "mood_score", "activity_minutes"]]
y = data["stress_level"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Example prediction
example = [[80, 6, 8, 5, 20]]  # Change values to test
print("\nPredicted stress level:", model.predict(example)[0])
