import streamlit as st
import pandas as pd
import joblib

# Load model
pipe = joblib.load("ipl_model.pkl")

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
    "Punjab Kings": "Sadda Punjab ❤️",
    "Rajasthan Royals": "Halla Bol 💗",
    "Gujarat Titans": "Aava De! 💪",
    "Lucknow Super Giants": "Ab Apni Baari Hai 💥"
}

# Team colors for background
team_colors = {
    "Mumbai Indians": "#045093",
    "Chennai Super Kings": "#f2cb05",
    "Royal Challengers Bangalore": "#da1818",
    "Kolkata Knight Riders": "#3d0066",
    "Delhi Capitals": "#17449b",
    "Sunrisers Hyderabad": "#ee5a24",
    "Punjab Kings": "#ba0c2f",
    "Rajasthan Royals": "#ea3d9c",
    "Gujarat Titans": "#0c2340",
    "Lucknow Super Giants": "#00b894"
}

# Start with black background
st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏆 IPL Win Predictor")

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

if st.button("Predict Winner"):
    try:
        balls_bowled = overs * 6
        balls_left = 120 - balls_bowled
        runs_left = target - score
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0
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

        # Inject CSS to change background
        bg_color = team_colors.get(winner, "#000000")
        st.markdown(f"""
            <style>
                .stApp {{
                    background-color: {bg_color};
                    color: white;
                }}
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"### 🏏 **{batting_team} Win Chance:** `{win_prob*100:.2f}%`")
        st.markdown(f"### 🎯 **{bowling_team} Win Chance:** `{loss_prob*100:.2f}%`")
        st.markdown(f"### 🏆 **Winner: {winner} - {team_taglines[winner]}**")

        # Show local team logo
        logo_file = f"logos/{winner.lower().replace(' ', '_').replace('super_kings','csk').replace('royal_challengers_bangalore','rcb').replace('mumbai_indians','mumbai')}.png"
        st.image(logo_file, width=150)

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check input.")
        st.exception(e)
