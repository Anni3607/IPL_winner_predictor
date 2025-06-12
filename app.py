import streamlit as st
import pandas as pd
import joblib

# Load the trained model
pipe = joblib.load('ipl_model.pkl') # Make sure 'ipl_model.pkl' is in the same directory or provide full path

st.title("IPL Match Winner Prediction")

# --- Step 1: Collect Raw Inputs from User ---
# You need to define ALL the raw inputs needed to create both
# the direct features AND the derived features.

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Batting Team', ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Rajasthan Royals'])
    bowling_team = st.selectbox('Bowling Team', ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Rajasthan Royals'])
    city = st.selectbox('City', ['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Jaipur', 'Ahmedabad', 'Pune', 'Durban', 'Port Elizabeth', 'Centurion', 'Durban', 'Johannesburg', 'East London', 'Kimberley', 'Bloemfontein', 'Cape Town', 'Abu Dhabi', 'Sharjah', 'Dubai', 'Ranchi', 'Rajkot', 'Indore', 'Mohali', 'Vizag', 'Cuttack']) # Add all cities your model saw

with col2:
    target_runs = st.number_input('Target Runs (Total set by 1st Innings)', min_value=1, value=160)
    current_score = st.number_input('Current Score (by Batting Team)', min_value=0, value=0)
    overs_completed = st.number_input('Overs Completed (by Batting Team)', min_value=0.0, max_value=20.0, value=0.0, step=0.1) # Use float for overs
    wickets_fallen = st.number_input('Wickets Fallen (by Batting Team)', min_value=0, max_value=10, value=0)

# --- Step 2: Perform Feature Engineering (Crucial!) ---
# This is where you calculate 'rrr' and 'balls_left' based on other inputs.
# The calculations MUST EXACTLY match how you created these features during training.

# Calculate balls_left
balls_left = 120 - int(overs_completed * 6) # Total 120 balls in 20 overs
remaining_overs_decimal = 20 - overs_completed
balls_rem_in_current_over = int(overs_completed * 10) % 10 # This gets the balls played in current over e.g., for 5.3 overs, it's 3
balls_left = (20 - int(overs_completed)) * 6 - balls_rem_in_current_over


# Calculate runs_needed
runs_needed = target_runs - current_score

# Calculate rrr (Required Run Rate)
if balls_left > 0:
    rrr = (runs_needed * 6) / balls_left
else:
    rrr = 0 # Or a very large number if the chase is over/impossible

# Map wickets_fallen to 'wickets' if your model uses 'wickets' as the feature name
# If your model used 'wickets_fallen' then keep that name for the DataFrame.
# For simplicity, assuming 'wickets' was the name used in training.
wickets = wickets_fallen


# --- Step 3: Create the Input DataFrame for the Model ---
# The column names here MUST exactly match what your model expects.

input_data = {
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'total_runs_x': [target_runs], # Assuming 'total_runs_x' was the target score in your training
    'balls_left': [balls_left],
    'wickets': [wickets],
    'rrr': [rrr]
}

# Create the DataFrame
# Ensure column order if your ColumnTransformer was sensitive to it, though usually it uses names.
input_df = pd.DataFrame(input_data)

# --- Step 4: Make Prediction ---
if st.button('Predict Winner'):
    try:
        # Debugging: Show the DataFrame being passed to the model
        st.write("Input DataFrame for Prediction:")
        st.dataframe(input_df)

        prediction_proba = pipe.predict_proba(input_df)[0]
        # Assuming pipe.predict_proba returns probabilities for each team
        # You'll need to know the order of classes from your model's training
        # For example, if your classes were ['TeamA', 'TeamB', 'TeamC']
        # You might need to get the actual class names from your model (pipe.classes_ if it's a classifier)

        # Get class names (assuming it's a classifier that has .classes_)
        try:
            team_names = pipe.classes_
            prediction_df = pd.DataFrame({'Team': team_names, 'Probability': prediction_proba})
            prediction_df = prediction_df.sort_values(by='Probability', ascending=False)

            st.subheader("Win Probabilities:")
            for index, row in prediction_df.iterrows():
                st.write(f"{row['Team']}: {row['Probability']:.2%}")

            st.success(f"Likely Winner: **{prediction_df.iloc[0]['Team']}** with {prediction_df.iloc[0]['Probability']:.2%} probability")

        except AttributeError:
            # If your model doesn't have .classes_ (e.g., a regressor returning a single value)
            st.warning("Model does not expose class probabilities directly. Displaying raw prediction.")
            st.success(f"Raw Prediction: {prediction_proba}") # Or pipe.predict(input_df)[0]
            # You might need to interpret this raw prediction based on your model's output
            # (e.g., if it's a score difference, or a probability for one team)


    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        st.error("Please ensure all required input fields are correctly filled and match the model's expectations based on the dataframe shown above.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
