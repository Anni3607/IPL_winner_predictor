import streamlit as st
import pandas as pd
import joblib

# Load the trained model
pipe = joblib.load('ipl_model.pkl')

st.title("IPL Winner Predictor")

# --- 1. Define Input Widgets for ALL Required Features ---

# Example: Assuming you have input widgets for these.
# You need to create similar widgets for all the missing columns.

# Example for 'city' and 'batting_team'
city = st.selectbox('City', ['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Jaipur', 'Ahmedabad', 'Pune', 'Durban', 'Port Elizabeth', 'Centurion', 'Durban', 'Johannesburg', 'East London', 'Kimberley', 'Bloemfontein', 'Cape Town'])
batting_team = st.selectbox('Batting Team', ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Rajasthan Royals'])
bowling_team = st.selectbox('Bowling Team', ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Rajasthan Royals'])

# Numeric inputs (ensure they are named exactly as expected by the model)
total_runs_x = st.number_input('Total Runs Scored (Target)', min_value=0, max_value=300, value=150) # Assuming this is the target score to chase
balls_left = st.number_input('Balls Left', min_value=0, max_value=120, value=60)
wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=5)

# --- 2. Feature Engineering for 'rrr' (Required Run Rate) if applicable ---
# If 'rrr' is not a direct input, it must be calculated.
# Assuming you have `crr` (current run rate) and `runs_needed`.
# This is a common derived feature in cricket prediction.
# If your model takes 'rrr' directly, then make a number_input for it.

# Let's assume 'runs_needed' is calculated from 'total_runs_x' and 'score_so_far'
# and 'overs_completed' is used with 'balls_left'.
# You need to align this with how you calculated 'rrr' during training.

# A common way to calculate 'rrr':
# runs_scored = total_runs_x - runs_needed # This might be total_runs_x if 'runs_needed' is how much more they need.
# overs_remaining = balls_left / 6
# If total_runs_x is the target to chase
runs_needed = total_runs_x + 1 # Target to win, assuming chasing a score of total_runs_x
# You need to define runs_scored by batting team and balls_played to get rrr correctly

# For simplicity, let's assume 'rrr' is based on the *current* situation for the *chasing team*
# You need to provide the actual logic used during training.
# Let's assume 'current_score' and 'overs_consumed' are also inputs or derived
# For this example, let's assume `rrr` is just a direct input for demonstration,
# but in a real scenario, you'd calculate it from current score, overs, and target.
rrr = st.number_input('Required Run Rate (RRR)', min_value=0.0, max_value=20.0, value=8.0)


# --- 3. Create a DataFrame with the EXACT Column Names ---
# This is where the error typically happens. The column names MUST match.

if st.button('Predict Winner'):
    # Create a dictionary with all the collected values
    input_data = {
        'city': [city],
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'total_runs_x': [total_runs_x],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'rrr': [rrr]
        # Add any other columns your model expects here
    }

    # Convert the dictionary to a pandas DataFrame
    # Ensure the order of columns or rely on the ColumnTransformer to handle it
    # but exact names are crucial.
    input_df = pd.DataFrame(input_data)

    try:
        # Make prediction
        prediction = pipe.predict(input_df)
        st.success(f"Predicted Winner: {prediction[0]}")
    except ValueError as e:
        st.error(f"Prediction error: {e}")
        st.error("Please ensure all required input fields are correctly filled and match the model's expectations.")
        st.write("Input DataFrame received by model:")
        st.dataframe(input_df) # This will help you debug what's being passed
