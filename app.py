import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
                   header=None)

# Define column names
columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
           "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
           "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
           "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
data.columns = columns

# Full form options for UI
full_form_options = {
    "cap-shape": {"b": "b - Bell", "c": "c - Conical", "f": "f - Flat", "k": "k - Knobbed", "s": "s - Sunken", "x": "x - Convex"},
    "cap-surface": {"f": "f - Fibrous", "g": "g - Grooves", "s": "s - Smooth", "y": "y - Scaly"},
    "cap-color": {"b": "b - Buff", "c": "c - Cinnamon", "e": "e - Red", "g": "g - Gray", "n": "n - Brown", "p": "p - Pink", "r": "r - Purple", "u": "u - Blue", "w": "w - White", "y": "y - Yellow"},
    "bruises": {"f": "f - No", "t": "t - Yes"},
    "odor": {"a": "a - Almond", "c": "c - Creosote", "f": "f - Foul", "l": "l - Anise", "m": "m - Musty", "n": "n - None", "p": "p - Pungent", "s": "s - Spicy", "y": "y - Fishy"},
    "gill-attachment": {"a": "a - Attached", "f": "f - Free"},
    "gill-spacing": {"c": "c - Close", "w": "w - Wide"},
    "gill-size": {"b": "b - Broad", "n": "n - Narrow"},
    "gill-color": {"b": "b - Buff", "e": "e - Red", "g": "g - Gray", "h": "h - Chocolate", "k": "k - Black", "n": "n - Brown", "o": "o - Orange", "p": "p - Pink", "r": "r - Purple", "u": "u - Blue", "w": "w - White", "y": "y - Yellow"},
    "stalk-shape": {"e": "e - Enlarging", "t": "t - Tapering"},
    "stalk-root": {"b": "b - Bulbous", "c": "c - Club", "e": "e - Equal", "r": "r - Rooted", "?": "? - Unknown"},
    "stalk-surface-above-ring": {"f": "f - Fibrous", "k": "k - Silky", "s": "s - Smooth", "y": "y - Scaly"},
    "stalk-surface-below-ring": {"f": "f - Fibrous", "k": "k - Silky", "s": "s - Smooth", "y": "y - Scaly"},
    "stalk-color-above-ring": {"b": "b - Buff", "c": "c - Cinnamon", "e": "e - Red", "g": "g - Gray", "n": "n - Brown", "o": "o - Orange", "p": "p - Pink", "w": "w - White", "y": "y - Yellow"},
    "stalk-color-below-ring": {"b": "b - Buff", "c": "c - Cinnamon", "e": "e - Red", "g": "g - Gray", "n": "n - Brown", "o": "o - Orange", "p": "p - Pink", "w": "w - White", "y": "y - Yellow"},
    "veil-type": {"p": "p - Partial"},
    "veil-color": {"n": "n - Brown", "o": "o - Orange", "w": "w - White", "y": "y - Yellow"},
    "ring-number": {"n": "n - None", "o": "o - One", "t": "t - Two"},
    "ring-type": {"e": "e - Evanescent", "f": "f - Flaring", "l": "l - Large", "n": "n - None", "p": "p - Pendant"},
    "spore-print-color": {"b": "b - Buff", "h": "h - Chocolate", "k": "k - Black", "n": "n - Brown", "o": "o - Orange", "r": "r - Purple", "u": "u - Blue", "w": "w - White", "y": "y - Yellow"},
    "population": {"a": "a - Abundant", "c": "c - Clustered", "n": "n - Numerous", "s": "s - Scattered", "v": "v - Several", "y": "y - Solitary"},
    "habitat": {"d": "d - Woods", "g": "g - Grasses", "l": "l - Leaves", "m": "m - Meadows", "p": "p - Paths", "u": "u - Urban", "w": "w - Waste"}
}

# Encode all data
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Mushroom Classifier", layout="wide")
st.title("🍄 Mushroom Classification App")
st.markdown("Enter mushroom characteristics below to predict if it's **Edible or Poisonous**.")

# Collect user input via dropdowns
user_input = {}
for feature in X.columns:
    if feature in full_form_options:
        options = list(full_form_options[feature].values())
        selected_full = st.sidebar.selectbox(feature, options, key=feature)
        # get encoded letter
        encoded = [k for k, v in full_form_options[feature].items() if v == selected_full][0]
        user_input[feature] = encoded
    else:
        values = label_encoders[feature].classes_.tolist()
        selected = st.sidebar.selectbox(feature, values, key=feature)
        user_input[feature] = selected

# Predict button
if st.sidebar.button("Predict"):
    encoded_input = [label_encoders[col].transform([user_input[col]])[0] for col in X.columns]
    prediction = model.predict(np.array(encoded_input).reshape(1, -1))[0]
    result = label_encoders["class"].inverse_transform([prediction])[0]

    if result == 'e':
        st.success("✅ This mushroom is **Edible**.")
    else:
        st.error("⚠️ This mushroom is **Poisonous**!")




