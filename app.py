
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

# Global CSS Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    .stSelectbox label, .stNumberInput label {
        color: white !important;
    }
    .stButton > button {
        color: black !important;
        background-color: #f2cb05;
        border: none;
        font-weight: bold;
    }
    .stMarkdown {
        color: white;
    }
    img {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏆 IPL Win Predictor")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams))

if batting_team == bowling_team:
    st.warning("Batting and Bowling teams cannot be the same. Please select different teams.")
else:
    city = st.selectbox("Match City", sorted(cities))
    target = st.number_input("Target Score", min_value=1)
    score = st.number_input("Current Score", min_value=0)
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1)

    if st.button("Predict Winner"):
        try:
            if overs == 0 and score > 0:
                st.error("Overs completed cannot be 0 if current score is greater than 0.")
            elif overs > 20:
                st.error("Overs completed cannot exceed 20.")
            elif score > target:
                st.error("Current score cannot be greater than target score.")
            else:
                balls_bowled = int(overs * 6)
                balls_left = 120 - balls_bowled
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

                prediction = pipe.predict_proba(input_df)
                win_prob = prediction[0][1]
                loss_prob = prediction[0][0]

                winner = batting_team if win_prob > loss_prob else bowling_team
                bg_color = team_colors.get(winner, "#000000")

                # Change background to winner's theme
                st.markdown(f"""
                    <style>
                    .stApp {{
                        background-color: {bg_color};
                        color: white;
                    }}
                    </style>
                """, unsafe_allow_html=True)

                st.markdown(f"### 🏏 *{batting_team} Win Chance:* {win_prob*100:.2f}%`")
                st.markdown(f"### 🎯 *{bowling_team} Win Chance:* {loss_prob*100:.2f}%")
                st.markdown(f"### 🏆 *Winner: {winner} — {team_taglines[winner]}*")

                # Show winner logo
                logo_name_map = {
                    "Chennai Super Kings": "chennai_csk",
                    "Mumbai Indians": "mumbai_mumbai",
                    "Royal Challengers Bangalore": "royal_challengers_bangalore",
                    "Kolkata Knight Riders": "kolkata_knight_riders",
                    "Delhi Capitals": "delhi_capitals",
                    "Sunrisers Hyderabad": "sunrisers_hyderabad",
                    "Punjab Kings": "punjab_kings",
                    "Rajasthan Royals": "rajasthan_royals",
                    "Gujarat Titans": "gujarat_titans",
                    "Lucknow Super Giants": "lucknow_super_giants"
                }
                logo_base_name = logo_name_map.get(winner, winner.lower().replace(' ', '_'))
                logo_file = f"logos/{logo_base_name}.png"

                try:
                    st.image(logo_file, width=150)
                except:
                    st.warning(f"Logo not found for {winner}. Please check: {logo_file}")

        except Exception as e:
            st.error("⚠ Prediction failed. Please check input.")
            st.exception(e)

