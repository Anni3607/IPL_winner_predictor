import streamlit as st
import pandas as pd
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

# Taglines per team
team_taglines = {
    "Chennai Super Kings": "Whistle Podu! 🦁",
    "Mumbai Indians": "Duniya Hila Denge! 🔵",
    "Kolkata Knight Riders": "Korbo Lorbo Jeetbo! 🟣",
    "Delhi Capitals": "Yeh Hai Nayi Dilli! 🔴",
    "Royal Challengers Bangalore": "Ee Sala Cup Namde! 🔥",
    "Rajasthan Royals": "Halla Bol! 👑",
    "Punjab Kings": "Sher Squad! 🟥",
    "Sunrisers Hyderabad": "Orange Army Rising! 🟠"
}

# Set background color
def set_background():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f4f9f9;
            font-family: 'Arial', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

# Load image and convert to base64 for display
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Display team logos and IPL logo
def display_logo(image_path, width=100):
    b64 = get_base64_image(image_path)
    if b64:
        st.markdown(f"""
            <img src='data:image/png;base64,{b64}' width='{width}' style='margin-bottom:10px;'>
        """, unsafe_allow_html=True)

# Apply style
set_background()

# Top IPL logo
display_logo("ipl_logo.png", width=120)
st.title("🏏 IPL Match Win Predictor")

# Load model
pipe = joblib.load("ipl_model.joblib")

teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
    'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town',
    'Port Elizabeth', 'Durban', 'Centurion', 'East London',
    'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
    'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune',
    'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali'
]

# Sidebar input
st.sidebar.header("Match Inputs")
batting_team = st.sidebar.selectbox("Batting Team", sorted(teams))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted(teams))
city = st.sidebar.selectbox("Match City", sorted(cities))

target = st.sidebar.number_input("Target Score", min_value=1)
score = st.sidebar.number_input("Current Score", min_value=0, max_value=target)
overs = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10)

if st.sidebar.button("Predict"):

    # Feature Engineering
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Fix: wrap everything in lists so all columns are same length
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

    prediction = pipe.predict_proba(input_df)
    loss = prediction[0][0]
    win = prediction[0][1]

    # Display team logos
    st.markdown("### 🧢 Team Logos")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**{batting_team}**")
        display_logo(f"logos/{batting_team}.png", width=100)
    with cols[1]:
        st.markdown(f"**{bowling_team}**")
        display_logo(f"logos/{bowling_team}.png", width=100)

    # Results
    st.markdown("## 📊 Win Probability")
    st.success(f"🏆 {batting_team} Win Chance: **{win*100:.2f}%**")
    st.info(f"📉 {bowling_team} Win Chance: **{loss*100:.2f}%**")

    # Display tagline for higher predicted team
    winner = batting_team if win > loss else bowling_team
    tagline = team_taglines.get(winner, "Let the best team win!")
    st.markdown(f"### 🎉 **{winner} - {tagline}**")
