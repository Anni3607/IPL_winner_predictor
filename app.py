import streamlit as st
import pandas as pd
import joblib

# Load model
pipe = joblib.load("ipl_model.pkl")

# PNG logos (as fallback if SVG doesn't work)
team_logos = {
    "Mumbai Indians": "https://upload.wikimedia.org/wikipedia/en/2/25/Mumbai_Indians_Logo.png",
    "Chennai Super Kings": "https://upload.wikimedia.org/wikipedia/en/2/2e/Chennai_Super_Kings_Logo.png",
    "Kolkata Knight Riders": "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.png",
    "Royal Challengers Bangalore": "https://upload.wikimedia.org/wikipedia/en/0/0a/Royal_Challengers_Bangalore_Logo.png",
    "Delhi Capitals": "https://upload.wikimedia.org/wikipedia/en/d/d4/Delhi_Capitals.png",
    "Sunrisers Hyderabad": "https://upload.wikimedia.org/wikipedia/en/e/e7/Sunrisers_Hyderabad.png",
    "Punjab Kings": "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.png",
    "Rajasthan Royals": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.png",
    "Gujarat Titans": "https://upload.wikimedia.org/wikipedia/en/0/09/Gujarat_Titans_Logo.png",
    "Lucknow Super Giants": "https://upload.wikimedia.org/wikipedia/en/5/5d/Lucknow_Super_Giants_Logo.png"
}

# Team taglines
team_taglines = {
    "Mumbai Indians": "Duniya Hila Denge 🔵",
    "Chennai Super Kings": "Whistle Podu 🦁",
    "Kolkata Knight Riders": "Korbo Lorbo Jeetbo 💜",
    "Royal Challengers Bangalore": "Ee Sala Cup Namde 🔥",
    "Delhi Capitals": "Roar Macha 🦅",
    "Sunrisers Hyderabad": "Orange Army 🧡",
    "Punjab Kings": "Sadda Punjab ❤️",
    "Rajasthan Royals": "Halla Bol 💗",
    "Gujarat Titans": "Aava De! 💪",
    "Lucknow Super Giants": "Ab Apni Baari Hai 💥"
}

# Background colors (safe darker colors for white text)
team_colors = {
    "Mumbai Indians": "#003366",
    "Chennai Super Kings": "#e6b800",
    "Kolkata Knight Riders": "#2e0854",
    "Royal Challengers Bangalore": "#7a0000",
    "Delhi Capitals": "#001f4d",
    "Sunrisers Hyderabad": "#b34700",
    "Punjab Kings": "#8b0000",
    "Rajasthan Royals": "#880e4f",
    "Gujarat Titans": "#001f33",
    "Lucknow Super Giants": "#003366"
}

# Inputs
teams = list(team_logos.keys())
cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
          'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
          'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg',
          'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
          'Bengaluru']

st.title("🏏 IPL Match Winner Predictor")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

city = st.selectbox('Match City', sorted(cities))
target = st.number_input('Target Score', min_value=1)
score = st.number_input('Current Score', min_value=0)
overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
wickets = st.number_input('Wickets Lost', min_value=0, max_value=10, step=1)

if st.button('Predict Winner'):
    balls_left = 120 - int(overs * 6)
    runs_left = target - score
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

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

    try:
        prediction = pipe.predict_proba(input_df)
        win_prob = prediction[0][1]
        loss_prob = prediction[0][0]
        winner = batting_team if win_prob > loss_prob else bowling_team

        # Change background and font color
        bg = team_colors[winner]
        st.markdown(f"""
            <style>
            .stApp {{
                background-color: {bg};
                color: white;
            }}
            </style>
        """, unsafe_allow_html=True)

        # Logo and output
        st.image(team_logos[winner], width=150)
        st.markdown(f"### 🏆 **{winner} - {team_taglines[winner]}**")
        st.success(f"{batting_team} win chance: {win_prob*100:.2f}%")
        st.info(f"{bowling_team} win chance: {loss_prob*100:.2f}%")

    except Exception as e:
        st.error("Prediction failed.")
        st.json(input_df.to_dict())
