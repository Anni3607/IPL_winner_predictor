import streamlit as st
import pandas as pd
import joblib
import base64
from PIL import Image # Import PIL for image handling

# --- Load Model ---
try:
    pipe = joblib.load("ipl_model.pkl")
except FileNotFoundError:
    st.error("Error: Model file 'ipl_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# --- Define Teams, Cities, Taglines, Colors ---
teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", "Kolkata Knight Riders",
         "Delhi Capitals", "Sunrisers Hyderabad", "Punjab Kings", "Rajasthan Royals",
         "Gujarat Titans", "Lucknow Super Giants"]

cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
          'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Ahmedabad', 'Nagpur',
          'Dharamsala', 'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

team_taglines = {
    "Mumbai Indians": "Duniya Hila Denge 🔵",
    "Chennai Super Kings": "Whistle Podu 🦁",
    "Royal Challengers Bangalore": "Ee Sala Cup Namde 🔥",
    "Kolkata Knight Riders": "Korbo Lorbo Jeetbo 💜",
    "Delhi Capitals": "Roar Macha 🦅",
    "Sunrisers Hyderabad": "Orange Army 🧡",
    "Punjab Kings": "Sadda Punjab ❤",
    "Rajasthan Royals": "Halla Bol 💗",
    "Gujarat Titans": "Aava De! 💪",
    "Lucknow Super Giants": "Ab Apni Baari Hai 💥"
}

# Team colors for dynamic background
team_colors = {
    "Mumbai Indians": "#045093",
    "Chennai Super Kings": "#F2CB05",
    "Royal Challengers Bangalore": "#DA1818",
    "Kolkata Knight Riders": "#3D0066",
    "Delhi Capitals": "#17449B",
    "Sunrisers Hyderabad": "#EE5A24",
    "Punjab Kings": "#BA0C2F",
    "Rajasthan Royals": "#EA3D9C",
    "Gujarat Titans": "#0C2340",
    "Lucknow Super Giants": "#00B894"
}

# --- Streamlit Page Configuration ---
st.set_page_config(layout="centered", page_title="IPL Win Predictor", page_icon="🏆")

# --- Function to get base64 encoded image ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# --- Custom CSS (Refined and consolidated) ---
st.markdown("""
    <style>
    /* General App Styling - for consistency and smooth transitions */
    .stApp {
        background-color: var(--background-color, #F0F2F6); /* Use CSS variable for dynamic update */
        color: var(--text-color, #313131); /* Default text color from config.toml */
        font-family: 'sans-serif'; /* From config.toml */
        transition: background-color 0.5s ease; /* Smooth transition */
    }

    /* Streamlit's main block container for content padding */
    div.block-container {
        padding-top: 1rem; /* More top padding */
        padding-bottom: 3rem; /* More bottom padding for content */
        padding-left: 2rem; /* Adjusted left padding */
        padding-right: 2rem; /* Adjusted right padding */
    }

    /* Header (h1) specific styling */
    h1 {
        color: #e60023; /* Consistent IPL red for the main title */
        text-align: center;
        font-size: 2.5em; /* Larger, more prominent title */
        margin-bottom: 0.5em;
    }

    /* Subheader (h2, h3) specific styling */
    h2, h3 {
        color: #e60023; /* IPL red for subheaders */
        margin-top: 1.5em; /* Space above subheaders */
        margin-bottom: 0.8em; /* Space below subheaders */
    }

    /* General Paragraph Text */
    p {
        color: #313131; /* Dark grey for general text */
        font-size: 1.1em; /* Slightly larger text for readability */
        line-height: 1.6; /* Better line spacing */
    }

    /* Prediction Result Text (Your existing h3 style, modified slightly) */
    .stMarkdown h3 {
        color: #e0e0e0; /* Slightly off-white for distinction on dark backgrounds */
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent dark background for contrast */
        padding: 15px; /* More padding */
        border-radius: 8px; /* Softer corners */
        margin-top: 2em; /* More space above prediction results */
        text-align: center; /* Center prediction text */
        font-size: 1.5em; /* Make prediction results clearer */
    }

    /* Style the Prediction Button */
    .stButton > button {
        background-color: #e60023; /* IPL red/maroon color from theme primaryColor */
        color: white;
        font-weight: bold;
        padding: 12px 25px; /* More padding for a bigger button */
        border-radius: 8px; /* Softer corners */
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth hover effect */
        width: 100%; /* Make button span full width for better UX */
        margin-top: 2em; /* Space above the button */
    }
    .stButton > button:hover {
        background-color: #cc0020; /* Slightly darker on hover */
        transform: translateY(-2px); /* Slight lift effect */
    }

    /* Style for Selectbox, NumberInput */
    .stSelectbox, .stNumberInput {
        border-radius: 8px; /* Softer corners for inputs */
        border: 1px solid #ddd; /* Light grey border */
        padding: 5px; /* Internal padding */
        margin-bottom: 1em; /* Space below each input */
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--secondary-background-color, #FFFFFF); /* Use secondaryBackgroundColor from theme */
        color: var(--text-color, #313131); /* Text color for selectbox */
    }
    .stNumberInput div[data-baseweb="input"] input {
        background-color: var(--secondary-background-color, #FFFFFF); /* Use secondaryBackgroundColor from theme */
        color: var(--text-color, #313131); /* Text color for number input */
    }

    /* Hide the Streamlit dataframe elements (already present, kept) */
    .stDataFrame {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section (IPL Trophy Logo Removed) ---
# Use columns for better centering of the title
col_left, col_title, col_right = st.columns([1, 4, 1])

with col_title:
    st.markdown("<h1 style='text-align: center;'>IPL Win Predictor</h1>", unsafe_allow_html=True)

st.markdown("---") # Horizontal rule for separation

st.markdown("<p style='text-align: center; font-size: 0.9em; color: #888;'>Developed by <strong>Anirudha Pujari</strong> </p>", unsafe_allow_html=True)


# --- Input Section ---
st.subheader("Match Details:") # Using subheader for clearer section title

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams), key="batting_team_select") # Added keys for uniqueness
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams), key="bowling_team_select")

# Display a warning if teams are the same
if batting_team == bowling_team:
    st.error("Batting and Bowling teams cannot be the same. Please select different teams to predict.")
    predict_button_disabled = True
else:
    predict_button_disabled = False

# --- Venue and Score Details (arranged for better flow) ---
st.markdown("---") # Separator for match details
st.subheader("Match Progress:")

city = st.selectbox("Match City", sorted(cities), key="match_city_select")
target = st.number_input("Target Score", min_value=1, value=150, help="Total runs the batting team needs to chase.")
score = st.number_input("Current Score", min_value=0, value=0, help="Current runs scored by the batting team.")
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="Total overs bowled so far.")
wickets = st.number_input("Wickets Remaining", min_value=0, max_value=10, value=10, step=1, help="Wickets the batting team has left (starts from 10).")




# --- Predict Button ---
if st.button("Predict Winner", disabled=predict_button_disabled):
    try:
        # Basic validation for inputs (moved here for clarity)
        if overs == 0 and score > 0:
            st.error("Cannot have a score greater than 0 with 0 overs completed. Please adjust.")
            st.stop()
        if overs > 20:
            st.error("Overs completed cannot exceed 20.")
            st.stop()
        if wickets > 10:
            st.error("Wickets lost cannot exceed 10.")
            st.stop()
        if target <= score and overs < 20: # Edge case: target already reached before overs end
            st.success(f"Target of {target} runs already achieved by {batting_team}!")
            st.balloons()
            winner = batting_team # Assign winner immediately
            # Apply dynamic background for winner
            bg_color = team_colors.get(winner, "#1a1a1a")
            st.markdown(f"""
                <style>
                .stApp {{
                    background-color: {bg_color} !important;
                }}
                </style>
            """, unsafe_allow_html=True)

            # Display winner and logo
            st.markdown(f"### 🏆 **{batting_team} has already Won!** - *{team_taglines.get(winner, '')}*", unsafe_allow_html=True)
            logo_base_name = winner.lower().replace(' ', '_')
            # Adjust specific team logo names if needed
            if logo_base_name == "chennai_super_kings": logo_base_name = "csk"
            elif logo_base_name == "royal_challengers_bangalore": logo_base_name = "rcb"
            elif logo_base_name == "mumbai_indians": logo_base_name = "mumbai"

            logo_path = f"logos/{logo_base_name}.png"
            encoded_logo = get_base64_image(logo_path)
            if encoded_logo:
                st.markdown(f"<img src='data:image/png;base64,{encoded_logo}' width='150' style='display: block; margin: auto; margin-top: 20px;'>", unsafe_allow_html=True)
            else:
                st.info(f"Could not load logo for {winner}.")

            st.stop() # Stop further execution if target reached

        # Calculate remaining game state
        balls_bowled = int(overs * 6)
        balls_left = 120 - balls_bowled
        runs_left = target - score

        # Avoid division by zero
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else (runs_left * 6 if runs_left > 0 else 0)
        wickets_left = 10 - wickets

        # Prepare input for the model
        input_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets': wickets,
            'total_runs_x': target, # Renamed to total_runs_x to match model's expected feature name
            'crr': crr,
            'rrr': rrr
        }])

        # --- Make Prediction ---
        prediction = pipe.predict_proba(input_df)
        win_prob = prediction[0][1] # Probability of batting team winning
        loss_prob = prediction[0][0] # Probability of bowling team winning

        winner = batting_team if win_prob > loss_prob else bowling_team

        # --- Dynamic Background Update (Applied here for the predicted winner) ---
        bg_color = team_colors.get(winner, "#1a1a1a") # Fallback to dark if winner color not found
        st.markdown(f"""
            <style>
            .stApp {{
                background-color: {bg_color} !important; /* Use !important to ensure override */
                color: #FFFFFF; /* Change text color to white for better contrast on dark backgrounds */
            }}
            /* Specific text colors for results to ensure visibility on dynamic background */
            .stMarkdown h3 {{
                color: #FFFFFF; /* White for the prediction results */
                background-color: rgba(0, 0, 0, 0.4); /* Keep a subtle background */
            }}
            p, label, .stSelectbox label, .stNumberInput label {{ /* Adjust all general text and labels */
                color: #e0e0e0; /* Light grey for general text */
            }}
            /* Adjust input field text color when background is dark */
            .stSelectbox div[data-baseweb="select"] {{
                background-color: rgba(255, 255, 255, 0.1); /* Slightly transparent white for inputs */
                color: #FFFFFF;
            }}
            .stSelectbox div[data-baseweb="select"] > div {{
                color: #FFFFFF !important; /* Ensure selected text is white */
            }}
            .stSelectbox div[data-baseweb="select"] svg {{ /* Dropdown arrow color */
                fill: #FFFFFF !important;
            }}
            .stNumberInput div[data-baseweb="input"] input {{
                background-color: rgba(255, 255, 255, 0.1);
                color: #FFFFFF;
            }}
            </style>
        """, unsafe_allow_html=True)

        # --- Display Prediction Results ---
        st.markdown(f"### 🏏 **{batting_team} Win Chance:** <span style='color:#76F2F2;'>{win_prob*100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 **{bowling_team} Win Chance:** <span style='color:#F2F276;'>{loss_prob*100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"### 🏆 **Predicted Winner: {winner}** - *{team_taglines.get(winner, '')}*", unsafe_allow_html=True)

        # Show local team logo (Using base64 for better deployment reliability)
        logo_base_name = winner.lower().replace(' ', '_')
        # Adjust specific team logo names (Ensure these match your actual filenames in the 'logos' folder)
        if logo_base_name == "chennai_super_kings": logo_base_name = "csk"
        elif logo_base_name == "royal_challengers_bangalore": logo_base_name = "rcb"
        elif logo_base_name == "mumbai_indians": logo_base_name = "mumbai"
        elif logo_base_name == "sunrisers_hyderabad": logo_base_name = "srh"
        elif logo_base_name == "kolkata_knight_riders": logo_base_name = "kkr"
        elif logo_base_name == "delhi_capitals": logo_base_name = "dc"
        elif logo_base_name == "punjab_kings": logo_base_name = "pbks"
        elif logo_base_name == "rajasthan_royals": logo_base_name = "rr"
        elif logo_base_name == "gujarat_titans": logo_base_name = "gt"
        elif logo_base_name == "lucknow_super_giants": logo_base_name = "lsg"

        logo_path = f"logos/{logo_base_name}.png" # Ensure your 'logos' folder exists and contains these PNGs
        
        encoded_logo = get_base64_image(logo_path)
        if encoded_logo:
            st.markdown(f"<img src='data:image/png;base64,{encoded_logo}' width='180' style='display: block; margin: auto; margin-top: 20px;'>", unsafe_allow_html=True) # Increased width for clarity, centered
        else:
            st.info(f"Could not load logo for {winner}. Please ensure '{logo_path}' exists.")

    except Exception as e:
        st.error("⚠ An error occurred during prediction. Please check input values and try again.")
        st.exception(e) # Display full exception for debugging


# --- Footer ---
st.markdown("---") # Horizontal rule
st.markdown("<p style='text-align: center; font-size: 0.9em; color: #888;'>Developed by Your Name/Team | Data from IPL Seasons</p>", unsafe_allow_html=True)
