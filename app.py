import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="🍄 Mushroom Classifier", layout="wide")

st.title("🍄 Mushroom Classification App")
st.markdown("### Enter mushroom characteristics below to predict if it's **Edible** or **Poisonous**.")

# Load data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
               "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
               "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    df = pd.read_csv(url, header=None, names=columns)
    return df

df = load_data()

# Encode features
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train model
X = df.drop("class", axis=1)
y = df["class"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Inverse mapping for dropdowns
inverse_options = {}
for col, le in label_encoders.items():
    inverse_options[col] = dict(zip(le.transform(le.classes_), le.classes_))

def get_user_input():
    user_data = {}
    st.sidebar.header("Choose mushroom features:")
    for col in X.columns:
        options = list(inverse_options[col].values())
        selected = st.sidebar.selectbox(f"{col}", options)
        encoded = label_encoders[col].transform([selected])[0]
        user_data[col] = encoded
    return pd.DataFrame([user_data])

user_input = get_user_input()

# Predict
prediction = model.predict(user_input)[0]
prediction_label = label_encoders['class'].inverse_transform([prediction])[0]
st.success(f"### 🌟 The mushroom is predicted to be: **{'Edible' if prediction_label == 'e' else 'Poisonous'}**")




