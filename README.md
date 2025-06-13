**IPL Win Predictor**

Welcome to the IPL Win Predictor! This is a simple web application built with Streamlit that helps you predict the chances of an IPL team winning a match based on live match conditions.


What is this?

If you've ever found yourself wondering who's got the real edge during an IPL match, then this app is for you. It takes real-time match data (like current score, overs, wickets, and the target) and uses a machine learning model to estimate the win probabilities for both the batting and bowling teams. It's a fun way to get insights into match dynamics.


 **How to Use the App**

It's easy to get started!

1.  **Go to the App:**
    Just click this link: (https://iplwinnerpredictor-htcxhxorjtxkbc4gyou5bb.streamlit.app/)
  

2.  **Enter Match Details:**
    You'll see a few boxes to fill in:
    * Batting Team: Who's currently batting?
    * Bowling Team: Who's trying to stop them?
    * Match City: Where's the match happening?
    * Target Score: How many runs does the batting team need to chase?
    * Current Score: What's the score right now?
    * Overs Completed: How many overs have been bowled? (e.g., 5.3 for 5 overs and 3 balls).
    * Wickets Lost: How many players are back in the pavilion?

3.  **Click "Predict Winner":**
    The app will instantly display the win probabilities for both teams and highlight the predicted winner, along with their team logo and tagline.


 **The Model & The Problem**

This project tackles a binary classification problem. In simple terms, the goal is to predict one of two outcomes for any given IPL match scenario: either the Batting Team Wins or the Bowling Team Wins. It's not about predicting the exact score, but rather assessing who's more likely to emerge victorious.

The core of this predictor is a Machine Learning Pipeline. This pipeline combines several steps:

* Data Preprocessing: It prepares the raw match data (like converting team names or cities into a format the model understands).
* A Probabilistic Classifier: This is the actual algorithm that learns from historical IPL data. It doesn't just guess "win" or "lose," but calculates the probability of each team winning (e.g., Batting Team 75% chance, Bowling Team 25% chance). Common examples of such classifiers are Random Forests or Gradient Boosting models, which are very effective for this kind of prediction.

This entire pipeline is saved as the `ipl_model.pkl` file in this repository.

**Repository Structure**

For those curious about the files, here's what's in this GitHub repo:

* `app.py`: This is the main Python script for the Streamlit web app.
* `ipl_model.pkl`: The trained machine learning model used for predictions.
* `requirements.txt`: A list of all the Python libraries you'd need to run this locally.
* `.streamlit/`: A folder for Streamlit's configuration files.
* `logos/`: Contains all the team logos used in the application.


 **Contribution**

I'm always open to ideas! If you spot a bug, have a feature idea, or just want to tinker with the code, feel out to dive in, open an issue, or submit a pull request.


**Acknowledgements**

* Built with Streamlit
* Machine learning model built using Python libraries (e.g., pandas, scikit-learn, joblib).



Enjoy the game and happy predicting!
