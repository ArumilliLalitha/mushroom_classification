import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="🍄 Mushroom Classifier", layout="wide")

st.title("🍄 Mushroom Classification App")
st.markdown("### Enter mushroom characteristics below to predict if it's **Edible** or **Poisonous**.")
st.markdown("#### 👩‍💻 Developed by **A.P.R.V. Lalitha** using streamlit app")

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

# Full form options
full_form_options = {
    "cap-shape": {"b": "Bell", "c": "Conical", "f": "Flat", "k": "Knobbed", "s": "Sunken", "x": "Convex"},
    "cap-surface": {"f": "Fibrous", "g": "Grooves", "s": "Smooth", "y": "Scaly"},
    "cap-color": {"b": "Buff", "c": "Cinnamon", "e": "Red", "g": "Gray", "n": "Brown", "p": "Pink", "r": "Purple", "u": "Blue", "w": "White", "y": "Yellow"},
    "bruises": {"f": "No", "t": "Yes"},
    "odor": {"a": "Almond", "c": "Creosote", "f": "Foul", "l": "Anise", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy", "y": "Fishy"},
    "gill-attachment": {"a": "Attached", "f": "Free"},
    "gill-spacing": {"c": "Close", "w": "Wide"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {"b": "Buff", "e": "Red", "g": "Gray", "h": "Chocolate", "k": "Black", "n": "Brown", "o": "Orange", "p": "Pink", "r": "Purple", "u": "Blue", "w": "White", "y": "Yellow"},
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {"b": "Bulbous", "c": "Club", "e": "Equal", "r": "Rooted", "?": "Unknown"},
    "stalk-surface-above-ring": {"f": "Fibrous", "k": "Silky", "s": "Smooth", "y": "Scaly"},
    "stalk-surface-below-ring": {"f": "Fibrous", "k": "Silky", "s": "Smooth", "y": "Scaly"},
    "stalk-color-above-ring": {"b": "Buff", "c": "Cinnamon", "e": "Red", "g": "Gray", "n": "Brown", "o": "Orange", "p": "Pink", "w": "White", "y": "Yellow"},
    "stalk-color-below-ring": {"b": "Buff", "c": "Cinnamon", "e": "Red", "g": "Gray", "n": "Brown", "o": "Orange", "p": "Pink", "w": "White", "y": "Yellow"},
    "veil-type": {"p": "Partial"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {"e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant"},
    "spore-print-color": {"b": "Buff", "h": "Chocolate", "k": "Black", "n": "Brown", "o": "Orange", "r": "Purple", "u": "Blue", "w": "White", "y": "Yellow"},
    "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"},
    "habitat": {"d": "Woods", "g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste"}
}

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

# Collect user input
def get_user_input():
    user_data = {}
    st.sidebar.header("Choose mushroom features:")
    for col in X.columns:
        if col in full_form_options:
            full_values = list(full_form_options[col].values())
            selected = st.sidebar.selectbox(f"{col}", full_values)
            # Get encoded short form (like 'f') from full form (like "Foul")
            short = [k for k, v in full_form_options[col].items() if v == selected][0]
            encoded = label_encoders[col].transform([short])[0]
            user_data[col] = encoded
        else:
            # fallback for features not in full_form_options
            options = list(label_encoders[col].classes_)
            selected = st.sidebar.selectbox(f"{col}", options)
            encoded = label_encoders[col].transform([selected])[0]
            user_data[col] = encoded
    return pd.DataFrame([user_data])

user_input = get_user_input()

# Predict
prediction = model.predict(user_input)[0]
prediction_label = label_encoders['class'].inverse_transform([prediction])[0]
st.success(f"### 🌟 The mushroom is predicted to be: **{'Edible' if prediction_label == 'e' else 'Poisonous'}**")

