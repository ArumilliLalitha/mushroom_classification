import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mushroom Classifier", layout="centered")

st.title("🍄 Mushroom Classification App")
st.markdown("Predict whether a mushroom is **Edible or Poisonous**.")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
        "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
        "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
        "spore-print-color", "population", "habitat"
    ]
    data = pd.read_csv(url, header=None, names=columns)
    label_encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data, label_encoders

data, label_encoders = load_data()

X = data.drop("class", axis=1)
y = data["class"]
model = RandomForestClassifier()
model.fit(X, y)
acc = accuracy_score(y, model.predict(X))

st.write(f"✅ Model Accuracy: **{acc:.2f}**")

st.subheader("🔍 Select Mushroom Features")

user_input = {}
for col in X.columns:
    options = list(label_encoders[col].classes_)
    user_input[col] = st.selectbox(col.replace("-", " ").title(), options)

if st.button("Predict"):
    input_encoded = [label_encoders[col].transform([user_input[col]])[0] for col in X.columns]
    prediction = model.predict([input_encoded])[0]
    result = "🍴 Edible" if prediction == 0 else "☠️ Poisonous"
    st.success(f"Prediction: **{result}**")

