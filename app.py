
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Ensure 'ipl_model.pkl' is in the same directory as app.py or provide the full path.
try:
    pipe = joblib.load('ipl_model.pkl')
except FileNotFoundError:
    st.error("Error: 'ipl_model.pkl' not found. Make sure the model file is in the same directory as this script.")
    st.stop() # Stop the app if model isn't found

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
# Based on your error: 'wickets', 'rrr', 'total_runs_x', 'balls_left', 'city', 'bowling_team', 'batting_team'

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
    ], key='city') # Ensure this list is exhaustive with all cities your model saw in training!

with col2:
    total_runs_x_input = st.number_input('Total Runs (1st Innings Score)', min_value=1, max_value=300, value=160, step=1, key='total_runs_x')
    balls_left_input = st.number_input('Balls Left in Innings', min_value=0, max_value=120, value=60, step=1, key='balls_left')
    wickets_input = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=0, step=1, key='wickets')
    rrr_input = st.number_input('Required Run Rate (RRR)', min_value=0.0, max_value=25.0, value=8.0, step=0.1, key='rrr')


# --- Create the Input DataFrame with EXACT Column Names ---
# The keys in this dictionary MUST EXACTLY match the column names
# your model was trained on (from your Colab notebook inspection).

input_data = {
    'batting_team': [batting_team_input],
    'bowling_team': [bowling_team_input],
    'city': [city_input],
    'total_runs_x': [total_runs_x_input],
    'balls_left': [balls_left_input],
    'wickets': [wickets_input],
    'rrr': [rrr_input]
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

        # Attempt to get class names from the pipeline
        try:
            # This assumes your final estimator (e.g., LogisticRegression, RandomForestClassifier)
            # inside the pipeline has a .classes_ attribute.
            # You might need to adjust 'final_estimator' to the actual name of your model step in the pipeline.
            # For example, if your pipeline is `pipe = Pipeline([('preprocessor', preprocessor_ct), ('model', classifier)])`
            # then it would be `pipe.named_steps['model'].classes_`
            team_names = pipe.classes_ # This works if the final estimator directly exposes classes_
            # If not, you might have to list them manually in the order your model was trained on.
            # Example: team_names = ['TeamA', 'TeamB', 'TeamC']

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
