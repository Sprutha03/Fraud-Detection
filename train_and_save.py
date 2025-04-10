# train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
print("Loading data...")
data = pd.read_csv("creditcard.csv")

# Balance classes
print("Balancing data...")
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud])

# Train model
print("Training model...")
X = data.drop("Class", axis=1)
y = data["Class"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl!")