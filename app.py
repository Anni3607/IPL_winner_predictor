
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
pipe = joblib.load('ipl_model.pkl')

# Define team and city options
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
    'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
    'Bengaluru'
]

# Streamlit app layout
st.title('IPL Win Predictor')

# Input fields
batting_team = st.selectbox('Select Batting Team', teams)
bowling_team = st.selectbox('Select Bowling Team', [team for team in teams if team != batting_team])
city = st.selectbox('Select City', cities)

target = st.number_input('Target Score', min_value=1)
current_score = st.number_input('Current Score', min_value=0, max_value=target)
overs = st.number_input('Overs Completed (e.g., 5.3)', min_value=0.0, max_value=20.0, step=0.1)
wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)

if st.button('Predict Win Probability'):
    # Calculations
    runs_left = target - current_score
    balls_left = 120 - int(overs * 6)
    crr = current_score * 6 / (120 - balls_left) if balls_left != 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [10 - wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probability
    result = pipe.predict_proba(input_df)[0]
    loss = np.round(result[0] * 100, 2)
    win = np.round(result[1] * 100, 2)

    st.subheader('Prediction Results')
    st.success(f'Win Probability: {win}%')
    st.error(f'Loss Probability: {loss}%')
