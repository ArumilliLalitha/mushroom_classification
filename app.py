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

# Define full-form mappings
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


# Encode categorical features
label_encoders = {}
categorical_options = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    categorical_options[col] = [full_form_options[col][key] if col in full_form_options and key in full_form_options[col] else key for key in le.classes_]

# Split dataset
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Function to encode unknown input samples
def encode_unknown_samples(sample, label_encoders, feature_names):
    encoded_sample = []
    for i, feature in enumerate(sample):
        if feature in label_encoders[feature_names[i]].classes_:
            encoded_sample.append(label_encoders[feature_names[i]].transform([feature])[0])
        else:
            encoded_sample.append(-1)  # Assign unknown values
    return np.array(encoded_sample).reshape(1, -1)

# User input for unknown mushroom
print("Enter mushroom characteristics:")
user_input = []
feature_names = X.columns.tolist()
for feature in feature_names:
    options = categorical_options[feature]
    print(f"Options for {feature}: {options}")
    value = input(f"Enter {feature}: ")
    while value not in options:
        print("Invalid input. Please choose from the given options.")
        value = input(f"Enter {feature}: ")
    user_input.append(value)

# Encode and predict user input
encoded_input = encode_unknown_samples(user_input, label_encoders, feature_names)
prediction = model.predict(encoded_input)[0]
pred_result = "Edible" if prediction == 0 else "Poisonous"
print(f"Predicted: {pred_result}")
# STREAMLIT APP

st.title("🍄 Mushroom Classification App")
st.write("Enter mushroom details to predict if it is edible or poisonous.")

# Select box for user input
cap_shape = st.selectbox("Cap Shape", data["cap-shape"].unique())
cap_surface = st.selectbox("Cap Surface", data["cap-surface"].unique())
cap_color = st.selectbox("Cap Color", data["cap-color"].unique())
odor = st.selectbox("Odor", data["odor"].unique())

# Encode inputs
input_data = pd.DataFrame({
    "cap-shape": [cap_shape],
    "cap-surface": [cap_surface],
    "cap-color": [cap_color],
    "odor": [odor]
})

# Combine with rest of required columns with default values
for col in data.columns:
    if col not in input_data.columns and col != "class":
        input_data[col] = data[col].mode()[0]  # use mode as default

# Encode input and predict
encoded_input = pd.DataFrame(encoder.transform(input_data), columns=input_data.columns)
prediction = model.predict(encoded_input)

# Display result
result = "Edible 🍽️" if prediction[0] == "e" else "Poisonous ☠️"
st.subheader(f"The mushroom is likely: **{result}**")

