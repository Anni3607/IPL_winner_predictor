import streamlit as st
import pandas as pd
import joblib

# Load model
pipe = joblib.load("ipl_model.pkl")

# Logos and taglines
team_logos = {
    "Mumbai Indians": "https://upload.wikimedia.org/wikipedia/en/2/25/Mumbai_Indians_Logo.svg",
    "Chennai Super Kings": "https://upload.wikimedia.org/wikipedia/en/2/2e/Chennai_Super_Kings_Logo.svg",
    "Kolkata Knight Riders": "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg",
    "Royal Challengers Bangalore": "https://upload.wikimedia.org/wikipedia/en/0/0a/Royal_Challengers_Bangalore_Logo.svg",
    "Delhi Capitals": "https://upload.wikimedia.org/wikipedia/en/d/d4/Delhi_Capitals.svg",
    "Sunrisers Hyderabad": "https://upload.wikimedia.org/wikipedia/en/e/e7/Sunrisers_Hyderabad.png",
    "Punjab Kings": "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg",
    "Rajasthan Royals": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg",
    "Gujarat Titans": "https://upload.wikimedia.org/wikipedia/en/0/09/Gujarat_Titans_Logo.svg",
    "Lucknow Super Giants": "https://upload.wikimedia.org/wikipedia/en/5/5d/Lucknow_Super_Giants_Logo.svg"
}

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

# Streamlit layout
st.markdown("<h1 style='text-align: center; color: #333;'>IPL Win Predictor 🏆</h1>", unsafe_allow_html=True)

teams = list(team_logos.keys())

cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
          'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
          'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg',
          'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
          'Bengaluru']

# Input form
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
    try:
        balls_bowled = overs * 6
        balls_left = int(120 - balls_bowled)
        runs_left = int(target - score)
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0
        wickets_left = 10 - wickets

        # ✅ Absolutely no lists inside
        input_dict = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'total_runs_x': target,
            'crr': crr,
            'rrr': rrr
        }

        # ✅ Convert scalars to DataFrame with one row
        input_df = pd.DataFrame([input_dict])

        # Debug - View DataFrame
        st.write("📊 Input DataFrame Sent to Model:")
        st.dataframe(input_df)

        # ✅ Predict
        prediction = pipe.predict_proba(input_df)
        win_prob = prediction[0][1]
        lose_prob = prediction[0][0]

        st.success(f"🏏 {batting_team} Win Chance: **{win_prob * 100:.2f}%**")
        st.info(f"🎯 {bowling_team} Win Chance: **{lose_prob * 100:.2f}%**")

        winner = batting_team if win_prob > lose_prob else bowling_team
        st.markdown(f"### 🏆 **{winner} - {team_taglines.get(winner, '')}**")
        st.image(team_logos[winner], width=150)

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.json(input_dict)
