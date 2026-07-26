
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💼 Employee Salary Classification")
st.write("Predict whether a person's annual income is <=50K or >50K.")

# Category mappings
workclass_map = {
    "?": 0, "Federal-gov": 1, "Local-gov": 2,
    "Never-worked": 3, "Private": 4, "Self-emp-inc": 5,
    "Self-emp-not-inc": 6, "State-gov": 7, "Without-pay": 8
}

marital_map = {
    "Divorced": 0, "Married-AF-spouse": 1,
    "Married-civ-spouse": 2, "Married-spouse-absent": 3,
    "Never-married": 4, "Separated": 5, "Widowed": 6
}

occupation_map = {
    "?": 0, "Adm-clerical": 1, "Armed-Forces": 2,
    "Craft-repair": 3, "Exec-managerial": 4,
    "Farming-fishing": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Other-service": 8,
    "Priv-house-serv": 9, "Prof-specialty": 10,
    "Protective-serv": 11, "Sales": 12,
    "Tech-support": 13, "Transport-moving": 14
}

relationship_map = {
    "Husband": 0, "Not-in-family": 1, "Other-relative": 2,
    "Own-child": 3, "Unmarried": 4, "Wife": 5
}

race_map = {
    "Amer-Indian-Eskimo": 0, "Asian-Pac-Islander": 1,
    "Black": 2, "Other": 3, "White": 4
}

gender_map = {"Female": 0, "Male": 1}

country_map = {
    "?": 0, "Cambodia": 1, "Canada": 2, "China": 3,
    "Columbia": 4, "Cuba": 5, "Dominican-Republic": 6,
    "Ecuador": 7, "El-Salvador": 8, "England": 9,
    "France": 10, "Germany": 11, "Greece": 12,
    "Guatemala": 13, "Haiti": 14, "Holand-Netherlands": 15,
    "Honduras": 16, "Hong": 17, "Hungary": 18,
    "India": 19, "Iran": 20, "Ireland": 21,
    "Italy": 22, "Jamaica": 23, "Japan": 24,
    "Laos": 25, "Mexico": 26, "Nicaragua": 27,
    "Outlying-US(Guam-USVI-etc)": 28, "Peru": 29,
    "Philippines": 30, "Poland": 31, "Portugal": 32,
    "Puerto-Rico": 33, "Scotland": 34, "South": 35,
    "Taiwan": 36, "Thailand": 37,
    "Trinadad&Tobago": 38, "United-States": 39,
    "Vietnam": 40, "Yugoslavia": 41
}

# User inputs
age = st.number_input("Age", min_value=17, max_value=100, value=30)

workclass = st.selectbox("Workclass", list(workclass_map.keys()))

fnlwgt = st.number_input(
    "Final Weight (fnlwgt)",
    min_value=10000,
    max_value=1500000,
    value=200000
)

educational_num = st.slider("Educational Number", 1, 16, 10)

marital_status = st.selectbox(
    "Marital Status", list(marital_map.keys())
)

occupation = st.selectbox(
    "Occupation", list(occupation_map.keys())
)

relationship = st.selectbox(
    "Relationship", list(relationship_map.keys())
)

race = st.selectbox("Race", list(race_map.keys()))

gender = st.selectbox("Gender", list(gender_map.keys()))

capital_gain = st.number_input(
    "Capital Gain", min_value=0.0, value=0.0
)

capital_loss = st.number_input(
    "Capital Loss", min_value=0.0, value=0.0
)

hours = st.slider("Hours per Week", 1, 99, 40)

country = st.selectbox(
    "Native Country",
    list(country_map.keys()),
    index=list(country_map.keys()).index("United-States")
)

# Prediction
if st.button("Predict Salary Class"):

    input_data = pd.DataFrame([[
        age,
        workclass_map[workclass],
        fnlwgt,
        educational_num,
        marital_map[marital_status],
        occupation_map[occupation],
        relationship_map[relationship],
        race_map[race],
        gender_map[gender],
        capital_gain,
        capital_loss,
        hours,
        country_map[country]
    ]], columns=[
        "age",
        "workclass",
        "fnlwgt",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ])

    # Same scaling used during training
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == ">50K":
        st.success("💰 Predicted Income: >50K")
    else:
        st.info("💵 Predicted Income: <=50K")
