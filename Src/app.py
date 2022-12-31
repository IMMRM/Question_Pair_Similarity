import pandas as pd
import streamlit as st
from PreProcess import PreProcess
import joblib

model=joblib.load(r'C:\Users\asus\Documents\Projects\Question_Similarity_Pairs\model\xgb_model.pkl')

st.header("Question Pair Similarity")

q1=st.text_input("Enter first question")
q2=st.text_input("Enter second question")

if st.button('Find'):
    query_df=pd.DataFrame([[q1,q2]],columns=['question1','question2'])
    transform=PreProcess()
    q_point=transform.transformation(query_df)
    res=model.predict(q_point)[0]
    if(res):
        st.header("Both are same questions")
    else:
        st.header("Both are different questions")
