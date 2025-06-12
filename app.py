import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

# Taglines
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

# Team logos from URLs
team_logos = {
    "Chennai Super Kings": "https://upload.wikimedia.org/wikipedia/en/2/2d/Chennai_Super_Kings_Logo.png",
    "Mumbai Indians": "https://upload.wikimedia.org/wikipedia/en/2/25/Mumbai_Indians_Logo.svg",
    "Kolkata Knight Riders": "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg",
    "Delhi Capitals": "https://upload.wikimedia.org/wikipedia/en/d/dc/Delhi_Capitals.svg",
    "Royal Challengers Bangalore": "https://upload.wikimedia.org/wikipedia/en/0/09/Royal_Challengers_Bangalore_Logo.svg",
    "Rajasthan Royals": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg",
    "Punjab Kings": "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg",
    "Sunrisers Hyderabad": "https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad.svg"
}

# Logo display function
def display_logo_from_url(team_name, width=100):
    url = team_logos.get(team_name)
    if url:
        st.markdown(f"<img src='{url}' width='{width}' style='margin-bottom:10px;'>", unsafe_allow_html=True)

# Background color
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f9f9;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# IPL logo (optional)
st.markdown("""
    <div style="text-align:center;">
        <img src="https://upload.wikimedia.org/wikipedia/en/d/d7/Indian_Premier_League_Official_Logo.svg" width="120"/>
    </div>
""", unsafe_allow_html=True)

st.title("🏏 IPL Match Win Predictor")

# Load model
pipe = joblib.load("ipl_model.pkl")


# Inputs
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

# Sidebar
st.sidebar.header("Match Details")
batting_team = st.sidebar.selectbox("Batting Team", sorted(teams))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted(teams))
city = st.sidebar.selectbox("City", sorted(cities))
target = st.sidebar.number_input("Target Score", min_value=1)
score = st.sidebar.number_input("Current Score", min_value=0, max_value=target)
overs = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10)

if st.sidebar.button("Predict"):

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # ✅ FIXED COLUMN NAME: 'wickets_left'
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

    try:
        prediction = pipe.predict_proba(input_df)
        loss = prediction[0][0]
        win = prediction[0][1]

        # Logos
        st.markdown("### 🧢 Team Logos")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**{batting_team}**")
            display_logo_from_url(batting_team)
        with cols[1]:
            st.markdown(f"**{bowling_team}**")
            display_logo_from_url(bowling_team)

        # Win chance
        st.markdown("## 📊 Win Probability")
        st.success(f"🏆 {batting_team} Win Chance: **{win*100:.2f}%**")
        st.info(f"📉 {bowling_team} Win Chance: **{loss*100:.2f}%**")

        # Tagline
        winner = batting_team if win > loss else bowling_team
        st.markdown(f"### 🎉 **{winner} - {team_taglines.get(winner, 'Let the best team win!')}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.json(input_df.to_dict())  # show data if error for debugging
