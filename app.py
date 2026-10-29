import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
st.title("EELU AI Predictor - ID 2601089")
data = {'attendance':[90,60,85,40,95,70,55,88,92,30],'assignment_score':[85,50,90,30,88,65,45,80,95,25],'midterm':[80,45,92,35,90,60,50,85,94,20],'result':[1,0,1,0,1,1,0,1,1,0]}
df = pd.DataFrame(data)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df[['attendance','assignment_score','midterm']], df['result'])
att = st.slider("Attendance",0,100,75)
assign = st.slider("Assignment",0,100,70)
mid = st.slider("Midterm",0,100,65)
if st.button("Predict"):
    pred = model.predict([[att,assign,mid]])
    st.success("PASS" if pred[0]==1 else "AT RISK")
