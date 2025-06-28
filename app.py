import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=None)
columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
           "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
           "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
           "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
data.columns = columns

# Label Encoding
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Train model
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit App
st.title("🍄 Mushroom Classification App")
st.markdown("Predict whether a mushroom is **edible** or **poisonous** based on its features.")
st.write(f"Model Accuracy: `{acc:.2f}`")

# Collect user inputs
user_input = {}
for feature in X.columns:
    options = label_encoders[feature].classes_
    selected = st.selectbox(f"Select {feature.replace('-', ' ').title()}", options)
    user_input[feature] = selected

# Predict button
if st.button("Classify Mushroom"):
    input_encoded = [label_encoders[feature].transform([user_input[feature]])[0] for feature in X.columns]
    prediction = model.predict([input_encoded])[0]
    result = "🍴 Edible" if prediction == 0 else "☠️ Poisonous"
    st.success(f"Prediction: **{result}**")
