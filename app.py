import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
# 1. Load the trained model
try:
    pipe = joblib.load('ipl_model.pkl')
except FileNotFoundError:
    st.error("Error: 'ipl_model.pkl' not found. Make sure the model file is in the same directory as this script.")
    st.stop()

# 2. Define Team Colors and Logos
TEAM_INFO = {
    'Mumbai Indians': {'color': '#004B8D', 'logo': 'mi_logo.png'},
    'Chennai Super Kings': {'color': '#FDB913', 'logo': 'csk_logo.png'},
    'Royal Challengers Bangalore': {'color': '#652D8A', 'logo': 'rcb_logo.png'},
    'Kolkata Knight Riders': {'color': '#3B215E', 'logo': 'kkr_logo.png'},
    'Delhi Capitals': {'color': '#00008B', 'logo': 'dc_logo.png'},
    'Sunrisers Hyderabad': {'color': '#FF822C', 'logo': 'srh_logo.png'},
    'Punjab Kings': {'color': '#B31B1B', 'logo': 'pk_logo.png'},
    'Lucknow Super Giants': {'color': '#A6E503', 'logo': 'lsg_logo.png'},
    'Gujarat Titans': {'color': '#002E4E', 'logo': 'gt_logo.png'},
    'Rajasthan Royals': {'color': '#D21289', 'logo': 'rr_logo.png'},
    'Other Team': {'color': '#CCCCCC', 'logo': None}
}

# 3. Get Model Class Names (Team Names in the order your model predicts them)
#    !!!! IMPORTANT !!!!
#    REPLACE THIS LIST WITH THE EXACT ORDER YOU GOT FROM YOUR COLAB NOTEBOOK.
MODEL_CLASSES = np.array(['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
                          'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
                          'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans',
                          'Rajasthan Royals']) # Adjust this list EXACTLY to your model's output order!

# --- Helper Function to ensure scalar values ---
def get_scalar_value(value):
    """
    Extracts a scalar value from a potential list or numpy array.
    If it's already a scalar, returns it as is.
    """
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 1:
            return value[0]
        elif len(value) == 0:
            # Handle empty lists/arrays, return a default/NaN or raise error
            # For our case, likely indicates an issue if it's empty
            st.error(f"Encountered an empty list/array where a scalar was expected: {value}")
            return None # Or a suitable default like 0 or 0.0
        else:
            st.error(f"Encountered a list/array with more than one element: {value}")
            # This case indicates a larger logic error, but we'll take the first element defensively
            return value[0]
    return value


# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="IPL Winner Predictor")

st.title("IPL Match Winner Prediction")

# Custom CSS for styling (including potential background color based on winning team)
st.markdown(
    """
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
        /* Dynamic background color - will be injected later */
        body {
            transition: background-color 0.5s ease;
        }
        .reportview-container .main {
            transition: background-color 0.5s ease;
            background-color: var(--dynamic-bg-color, #0e1117); /* Default dark background */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Input Widgets ---
col1, col2 = st.columns(2)

with col1:
    batting_team_input = st.selectbox('Batting Team', list(TEAM_INFO.keys()), key='bat_team')
    bowling_team_input = st.selectbox('Bowling Team', list(TEAM_INFO.keys()), key='bowl_team')
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

# --- Feature Engineering ---
# Use get_scalar_value to ensure inputs are scalars before calculations
total_runs_x_input_s = get_scalar_value(total_runs_x_input)
current_score_input_s = get_scalar_value(current_score_input)
overs_completed_input_s = get_scalar_value(overs_completed_input)
wickets_input_s = get_scalar_value(wickets_input)

# Ensure all intermediate calculations result in scalar int/float values.
num_overs = int(overs_completed_input_s)
num_balls_in_current_over = int(round((overs_completed_input_s - num_overs) * 10))
total_balls_played = int(num_overs * 6 + num_balls_in_current_over)
balls_left_calculated = int(120 - total_balls_played)
balls_left_calculated = max(0, balls_left_calculated)

runs_left_calculated = int(total_runs_x_input_s - current_score_input_s)

if float(balls_left_calculated) > 0:
    rrr_calculated = float((runs_left_calculated * 6) / float(balls_left_calculated))
else:
    rrr_calculated = 0.0

if float(overs_completed_input_s) > 0:
    crr_calculated = float(current_score_input_s / float(overs_completed_input_s))
else:
    crr_calculated = 0.0


# --- Create Input DataFrame ---
# Apply get_scalar_value on all inputs just before putting into the final list
# and then cast to the expected type for the model.
input_data = {
    'batting_team': [str(get_scalar_value(batting_team_input))],
    'bowling_team': [str(get_scalar_value(bowling_team_input))],
    'city': [str(get_scalar_value(city_input))],
    'total_runs_x': [int(get_scalar_value(total_runs_x_input_s))],
    'balls_left': [int(get_scalar_value(balls_left_calculated))],
    'wickets': [int(get_scalar_value(wickets_input_s))],
    'rrr': [float(get_scalar_value(rrr_calculated))],
    'runs_left': [int(get_scalar_value(runs_left_calculated))],
    'crr': [float(get_scalar_value(crr_calculated))]
}
input_df = pd.DataFrame(input_data)

# --- Make Prediction ---
if st.button('Predict Winner', key='predict_button'):
    try:
        # Display the DataFrame being sent to the model for debugging
        st.subheader("DataFrame sent to Model:")
        st.dataframe(input_df)

        prediction_proba = pipe.predict_proba(input_df)[0]

        # Map probabilities to team names
        team_probabilities = pd.DataFrame({
            'Team': MODEL_CLASSES,
            'Probability': prediction_proba
        })
        team_probabilities = team_probabilities.sort_values(by='Probability', ascending=False)

        winning_team = team_probabilities.iloc[0]['Team']
        winning_probability = team_probabilities.iloc[0]['Probability']

        # --- Dynamic Background Color ---
        winning_team_color = TEAM_INFO.get(winning_team, {}).get('color', '#0e1117')
        st.markdown(
            f"""
            <style>
                .reportview-container .main {{
                    background-color: {winning_team_color};
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        # --- Display Winner and Probabilities ---
        st.subheader("Win Probabilities:")
        col_prob, col_logo = st.columns([0.7, 0.3])

        with col_prob:
            for index, row in team_probabilities.iterrows():
                st.write(f"**{row['Team']}**: {row['Probability']:.2%}")

        with col_logo:
            winning_team_logo_path = TEAM_INFO.get(winning_team, {}).get('logo')
            if winning_team_logo_path:
                try:
                    st.image(winning_team_logo_path, caption=f"{winning_team} Logo", width=150)
                except FileNotFoundError:
                    st.warning(f"Logo file '{winning_team_logo_path}' not found for {winning_team}. Make sure it's uploaded to your repository.")
            else:
                st.info(f"No logo available for {winning_team}.")


        st.success(f"Likely Winner: **{winning_team}** with {winning_probability:.2%} probability!")

    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        st.error("This often means there's a mismatch in column names or data types. Please check the 'DataFrame sent to Model' above and compare it with the expected columns from your training data.")
        st.write("Debug info from `input_data` values (before final list wrapping):")
        # This will now show the scalar value before it's wrapped in a list.
        # This is for internal debugging, you can remove this section later.
        st.write(f"- batting_team: {get_scalar_value(batting_team_input)} (type: {type(get_scalar_value(batting_team_input))})")
        st.write(f"- bowling_team: {get_scalar_value(bowling_team_input)} (type: {type(get_scalar_value(bowling_team_input))})")
        st.write(f"- city: {get_scalar_value(city_input)} (type: {type(get_scalar_value(city_input))})")
        st.write(f"- total_runs_x: {get_scalar_value(total_runs_x_input_s)} (type: {type(get_scalar_value(total_runs_x_input_s))})")
        st.write(f"- balls_left: {get_scalar_value(balls_left_calculated)} (type: {type(get_scalar_value(balls_left_calculated))})")
        st.write(f"- wickets: {get_scalar_value(wickets_input_s)} (type: {type(get_scalar_value(wickets_input_s))})")
        st.write(f"- rrr: {get_scalar_value(rrr_calculated)} (type: {type(get_scalar_value(rrr_calculated))})")
        st.write(f"- runs_left: {get_scalar_value(runs_left_calculated)} (type: {type(get_scalar_value(runs_left_calculated))})")
        st.write(f"- crr: {get_scalar_value(crr_calculated)} (type: {type(get_scalar_value(crr_calculated))})")

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.write("Please check your input values and the model's compatibility, and ensure your `MODEL_CLASSES` list in `app.py` is correct.")
