import streamlit as st
import pandas as pd
import joblib
import base64

# Load model
try:
    pipe = joblib.load("ipl_model.pkl")
except FileNotFoundError:
    st.error("Error: Model file 'ipl_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Define teams and cities
teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", "Kolkata Knight Riders",
         "Delhi Capitals", "Sunrisers Hyderabad", "Punjab Kings", "Rajasthan Royals",
         "Gujarat Titans", "Lucknow Super Giants"]

cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
          'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Ahmedabad', 'Nagpur',
          'Dharamsala', 'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Team taglines
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

# Team colors for background (used for dynamic background)
team_colors = {
    "Mumbai Indians": "#045093",
    "Chennai Super Kings": "#F2CB05", # Slightly adjusted for visibility
    "Royal Challengers Bangalore": "#DA1818",
    "Kolkata Knight Riders": "#3D0066",
    "Delhi Capitals": "#17449B",
    "Sunrisers Hyderabad": "#EE5A24",
    "Punjab Kings": "#BA0C2F",
    "Rajasthan Royals": "#EA3D9C",
    "Gujarat Titans": "#0C2340",
    "Lucknow Super Giants": "#00B894"
}

# --- Set Streamlit Page Configuration ---
st.set_page_config(layout="centered", page_title="IPL Win Predictor", page_icon="🏆")

# Function to get base64 encoded image
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Warning: Logo file not found at {image_path}. Displaying placeholder.")
        return None

# Inject custom CSS (remaining specific overrides)
st.markdown("""
    <style>
    /* Ensure the background color transition is smooth */
    .stApp {
        transition: background-color 0.5s ease; /* Smooth transition */
    }

    /* Make prediction text stand out */
    h3 {
        color: #e0e0e0; /* Slightly off-white for distinction */
        background-color: rgba(0, 0, 0, 0.4); /* Semi-transparent dark background */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* Style the prediction button */
    .stButton > button {
        background-color: #e60023; /* IPL red/maroon color from theme primaryColor */
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #cc0020;
    }

    /* Hide the Streamlit dataframe elements */
    .stDataFrame {
        display: none !important;
    }

    /* Adjust padding for better layout */
    div.block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏆 IPL Win Predictor")

st.subheader("Match Details:")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams))

city = st.selectbox("Match City", sorted(cities))
target = st.number_input("Target Score", min_value=1)
score = st.number_input("Current Score", min_value=0)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1)

# Ensure batting and bowling teams are different
if batting_team == bowling_team:
    st.warning("Batting and Bowling teams cannot be the same. Please select different teams.")
    predict_button_disabled = True
else:
    predict_button_disabled = False

if st.button("Predict Winner", disabled=predict_button_disabled):
    try:
        # Basic validation for overs and wickets
        if overs == 0 and score > 0:
            st.warning("Cannot have a score greater than 0 with 0 overs completed. Please adjust.")
            st.stop()
        if overs > 20:
            st.warning("Overs completed cannot exceed 20.")
            st.stop()
        if wickets > 10:
            st.warning("Wickets lost cannot exceed 10.")
            st.stop()

        balls_bowled = int(overs * 6)
        balls_left = 120 - balls_bowled
        runs_left = target - score

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else (runs_left * 6)

        wickets_left = 10 - wickets

        input_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets': wickets,
            'total_runs_x': target,
            'crr': crr,
            'rrr': rrr
        }])

        prediction = pipe.predict_proba(input_df)
        win_prob = prediction[0][1]
        loss_prob = prediction[0][0]

        winner = batting_team if win_prob > loss_prob else bowling_team

        # --- Dynamic background update (Apply only after prediction) ---
        # This CSS overrides the config.toml for the main background dynamically
        bg_color = team_colors.get(winner, "#1a1a1a") # Fallback to dark if winner color not found
        st.markdown(f"""
            <style>
            .stApp {{
                background-color: {bg_color} !important; /* Use !important to ensure override */
            }}
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"### 🏏 **{batting_team} Win Chance:** <span style='color:#76F2F2;'>{win_prob*100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 **{bowling_team} Win Chance:** <span style='color:#F2F276;'>{loss_prob*100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"### 🏆 **Predicted Winner: {winner}** - *{team_taglines[winner]}*", unsafe_allow_html=True)

        # Show local team logo (Using base64 for better deployment reliability)
        logo_base_name = winner.lower().replace(' ', '_')
        if logo_base_name == "chennai_super_kings":
            logo_base_name = "csk"
        elif logo_base_name == "royal_challengers_bangalore":
            logo_base_name = "rcb"
        elif logo_base_name == "mumbai_indians":
            logo_base_name = "mumbai"

        logo_path = f"logos/{logo_base_name}.png"
        
        encoded_logo = get_base64_image(logo_path)
        if encoded_logo:
            st.markdown(f"<img src='data:image/png;base64,{encoded_logo}' width='150'>", unsafe_allow_html=True)
        else:
            st.warning("Logo not found. Please ensure logo files are in the 'logos' directory and named correctly.")

    except Exception as e:
        st.error("⚠ Prediction failed. Please check input values and try again.")
        st.exception(e)
