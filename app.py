import streamlit as st
import joblib
import pandas as pd

# Load the trained model
pipe = joblib.load('ipl_model.pkl')

# Title
st.title("🏏 IPL Winner Predictor")

# Teams
teams = ['Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders',
         'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

# Cities (venues)
cities = ['Mumbai', 'Kolkata', 'Delhi', 'Chennai', 'Hyderabad', 'Jaipur', 'Bangalore', 'Ahmedabad']

# User input
batting_team = st.selectbox("Select Batting Team", sorted(teams))
bowling_team = st.selectbox("Select Bowling Team", sorted([team for team in teams if team != batting_team]))
city = st.selectbox("Select City", sorted(cities))

target = st.number_input("Target Score", min_value=1)
score = st.number_input("Current Score", min_value=0, max_value=target)
overs = st.number_input("Overs Completed", min_value=0.1, max_value=20.0, step=0.1)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

# Calculate derived features
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_left = 10 - wickets
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# Input DataFrame for model
input_df = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets_left': [wickets_left],
    'total_runs_x': [target],
    'crr': [crr],
    'rrr': [rrr]
})

# Prediction
if st.button("Predict Winner"):
    result = pipe.predict_proba(input_df)[0]
    win_prob = result[1] * 100  # Probability of winning
    st.subheader(f"🏆 {batting_team} has a {win_prob:.2f}% chance of winning")
