#  0   Survived  891 non-null    int64  
#  1   Pclass    891 non-null    int64  
#  2   Sex       891 non-null    object 
#  3   Age       891 non-null    float64
#  4   SibSp     891 non-null    int64  
#  5   Parch     891 non-null    int64  
#  6   Fare      891 non-null    float64
#  7   Embarked  891 non-null    object 
# dtypes: float64(2), int64(4), object(2


import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

st.title("Will you survive the Titanic?")

Pclass = st.selectbox("Passenger class", [1,2,3])
Sex = st.selectbox("Select your Gender",["male","female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Number of sibling/Spouse you're travelling with", 0,10,0)
Parch = st.number_input("Number of parents/childrens you're travelling with", 0,10,0)
Fare = st.number_input("Fare", 0.0, 600.0, 18.0)
Embarked = st.selectbox("Port of boarding (S = Southampton (England), C = Cherbourg (France), Q = Queenstown (now Cobh, Ireland)", ['S', 'C', 'Q    '])


data = {'Pclass': Pclass,
          'Sex' : Sex,
          'Age' : Age,
          'SibSp' : SibSp,
          'Parch' : Parch,
          'Fare' : Fare,
          'Embarked' : Embarked}

df = pd.DataFrame([data])

pred = model.predict(df)[0]
prob = model.predict_proba(df)[0,1]


if pred == 1:
    st.success(f"You survived! (Survival Probability: {prob:.2%})")
else:
    st.error(f"Did not survived. (Survival Probability: {prob:.2%})")

