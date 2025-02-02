import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 1: Load the Dataset
print("Loading dataset...")
df = pd.read_csv("bank-additional-full.csv", sep=";")  # Using semicolon as separator

# Step 2: Preprocessing
print("Preprocessing data...")

# Identify categorical and numerical features
categorical_features = ["job", "marital", "education", "contact", "month", "poutcome"]
numerical_features = ["age", "balance", "campaign", "pdays", "previous"]

# Drop 'duration' since it's only available after the call
df = df.drop(columns=["duration"])

# Encode categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),  # Scale numerical values
        ("cat", OneHotEncoder(), categorical_features)  # Convert categorical values
    ]
)

# Encode target variable
le = LabelEncoder()
df["y"] = le.fit_transform(df["y"])  # Convert 'yes'/'no' to 1/0

# Split into train and test sets
X = df.drop(columns=["y"])
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(
    preprocessor.fit_transform(X), y, test_size=0.2, random_state=42
)

# Step 3: Define the Neural Network Model
print("Building the model...")
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.5),  # Prevents overfitting
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Outputs probability (0 to 1)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Step 4: Train the Model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 5: Make Predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probability to binary class

# Step 6: Evaluate Model Performance
print("Evaluating model performance...")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display Results
print("\n✅ Model Performance:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Save the model for future use
model.save("bank_marketing_model.h5")
print("\n✅ Model saved as 'bank_marketing_model.h5'")
