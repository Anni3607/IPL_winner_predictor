
import streamlit as st
import pandas as pd
import joblib

# ✅ Load trained model
pipe = joblib.load('ipl_model.pkl')

# ✅ App Title
st.title("🏏 IPL Winner Predictor")

# ✅ User Inputs
team1 = st.selectbox("Select Team 1", sorted(["MI", "CSK", "RCB", "KKR", "RR", "SRH", "DC", "PBKS", "GT", "LSG"]))
team2 = st.selectbox("Select Team 2", sorted(["MI", "CSK", "RCB", "KKR", "RR", "SRH", "DC", "PBKS", "GT", "LSG"]))
venue = st.selectbox("Venue", ["Wankhede", "Chepauk", "Eden Gardens", "M. Chinnaswamy", "Narendra Modi Stadium"])
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.radio("Toss Decision", ["bat", "field"])

# ✅ Predict Button
if st.button("Predict Winner"):
    input_df = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'venue': [venue],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision]
    })

    result = pipe.predict(input_df)[0]
    st.success(f"🏆 Predicted Winner: {result}")
