
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    pipe = joblib.load('ipl_model.pkl')
except FileNotFoundError:
    st.error("Error: 'ipl_model.pkl' not found. Make sure the model file is in the same directory as this script.")
    st.stop()

st.title("IPL Match Winner Prediction")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stSelectbox, .stNumberInput {
        border-radius: 8px;
        padding: 10px;
        box-shadow: inset 1px 1px 3px rgba(0,0,0,0.1);
    }
    .stSuccess {
        background-color: #e6ffe6;
        color: #006600;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #ffe6e6;
        color: #cc0000;
        border-left: 5px solid #ff0000;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- Input Widgets for ALL the columns your model expects ---
# Including raw inputs needed for derived features like runs_left and crr.

col1, col2 = st.columns(2)

with col1:
    batting_team_input = st.selectbox('Batting Team', [
        'Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings',
        'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
        'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Rajasthan Royals'
    ], key='bat_team')
    bowling_team_input = st.selectbox('Bowling Team', [
        'Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings',
        'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
        'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Rajasthan Royals'
    ], key='bowl_team')
    city_input = st.selectbox('City', [
        'Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad',
        'Jaipur', 'Ahmedabad', 'Pune', 'Durban', 'Port Elizabeth', 'Centurion',
        'Johannesburg', 'East London', 'Kimberley', 'Bloemfontein', 'Cape Town',
        'Abu Dhabi', 'Sharjah', 'Dubai', 'Ranchi', 'Rajkot', 'Indore',
        'Mohali', 'Vizag', 'Cuttack'
    ], key='city')

with col2:
    total_runs_x_input = st.number_input('Total Runs (1st Innings Score)', min_value=1, max_value=300, value=160, step=1, key='total_runs_x')
    current_score_input = st.number_input('Current Score (by Batting Team)', min_value=0, max_value=300, value=0, step=1, key='current_score')
    overs_completed_input = st.number_input('Overs Completed (by Batting Team)', min_value=0.0, max_value=20.0, value=0.0, step=0.1, key='overs_completed')
    wickets_input = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=0, step=1, key='wickets')

# --- Perform Feature Engineering (Crucial!) ---
# These calculations MUST EXACTLY match how you created these features in your training data.

# Calculate balls_left (from overs_completed)
num_overs = int(overs_completed_input)
num_balls_in_current_over = round((overs_completed_input - num_overs) * 10) # 0.1 -> 1, 0.5 -> 5 balls
total_balls_played = num_overs * 6 + num_balls_in_current_over
balls_left_calculated = 120 - total_balls_played
balls_left_calculated = max(0, balls_left_calculated) # Ensure it doesn't go negative

# Calculate runs_left
runs_left_calculated = total_runs_x_input - current_score_input

# Calculate rrr (Required Run Rate)
if balls_left_calculated > 0:
    rrr_calculated = (runs_left_calculated * 6) / balls_left_calculated
else:
    rrr_calculated = 0 # If no balls left, RRR is 0 (or infinity if runs still needed)

# Calculate crr (Current Run Rate)
if overs_completed_input > 0:
    crr_calculated = current_score_input / overs_completed_input
else:
    crr_calculated = 0 # Avoid division by zero if 0 overs completed


# --- Create the Input DataFrame with EXACT Column Names ---
# The keys in this dictionary MUST EXACTLY match the column names
# your model was trained on, including 'runs_left' and 'crr'.

input_data = {
    'batting_team': [batting_team_input],
    'bowling_team': [bowling_team_input],
    'city': [city_input],
    'total_runs_x': [total_runs_x_input],
    'balls_left': [balls_left_calculated], # Now calculated
    'wickets': [wickets_input],
    'rrr': [rrr_calculated], # Now calculated
    'runs_left': [runs_left_calculated], # NEW: Added for the error
    'crr': [crr_calculated] # NEW: Added for the error
}

# Convert the dictionary to a pandas DataFrame
input_df = pd.DataFrame(input_data)

# --- Make Prediction ---
if st.button('Predict Winner', key='predict_button'):
    try:
        # Display the DataFrame being sent to the model for debugging
        st.subheader("DataFrame sent to Model:")
        st.dataframe(input_df)

        prediction_proba = pipe.predict_proba(input_df)[0]

        try:
            team_names = pipe.classes_
            prediction_df = pd.DataFrame({'Team': team_names, 'Probability': prediction_proba})
            prediction_df = prediction_df.sort_values(by='Probability', ascending=False)

            st.subheader("Win Probabilities:")
            for index, row in prediction_df.iterrows():
                st.write(f"**{row['Team']}**: {row['Probability']:.2%}")

            st.success(f"Likely Winner: **{prediction_df.iloc[0]['Team']}** with {prediction_df.iloc[0]['Probability']:.2%} probability!")

        except AttributeError:
            st.warning("Could not determine team names from model. Displaying raw prediction probabilities.")
            st.json({f"Class {i}": prob for i, prob in enumerate(prediction_proba)})
            st.success(f"Raw Prediction (Index of winning class): {pipe.predict(input_df)[0]}")

    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        st.error("This often means there's a mismatch in column names or data types. Please check the 'DataFrame sent to Model' above and compare it with the expected columns from your training data.")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.write("Please check your input values and the model's compatibility.")
