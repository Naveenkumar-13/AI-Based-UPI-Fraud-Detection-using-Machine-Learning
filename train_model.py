import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("upi_fraud_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Convert time to hour
df["hour"] = pd.to_datetime(
    df["time"],
    format="%I:%M %p"
).dt.hour

# Remove unwanted columns
df = df.drop(columns=[
    "transaction_id",
    "upi_id",
    "date",
    "time"
])

# Encode text columns
label_encoder = LabelEncoder()

for column in df.columns:

    df[column] = df[column].astype(str)

    df[column] = label_encoder.fit_transform(df[column])

# Features and target
X = df.drop("fraud_label", axis=1)

y = df["fraud_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy * 100)

# Save model
joblib.dump(model, "fraud_model.pkl")

print("Model trained successfully")