from flask import Flask, render_template, request, redirect, flash
import pickle
import pandas as pd
import csv
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'

model = pickle.load(open('model.pkl', 'rb'))
win_model = pickle.load(open('win_model.pkl', 'rb'))

teams = ['CSK', 'DC', 'KKR', 'MI', 'PBKS', 'RCB', 'RR', 'SRH']

# Set up encoders to match the model's training setup
batting_encoder = LabelEncoder()
bowling_encoder = LabelEncoder()
batting_encoder.classes_ = teams
bowling_encoder.classes_ = teams

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    overs = float(request.form['overs'])
    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])

    input_data = {
        'over': overs,
        'runs_so_far': runs,
        'wickets': wickets
    }

    for team in teams:
        input_data['batting_team_' + team] = 1 if batting_team == team else 0
        input_data['bowling_team_' + team] = 1 if bowling_team == team else 0

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    final_score = round(prediction, 2)

    # let arr = [10, 25, 3, 99, 7, 56];
    # let max = arr[0]; // assume first value is max

    # for (let i = 1; i < arr.length; i++) {
    #     if (arr[i] > max) {
    #         max = arr[i]; // update max if current value is greater
    #     }
    # }


    with open('predictions.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), batting_team, bowling_team, overs, runs, wickets, final_score])

    return render_template('index.html', teams=teams, prediction=final_score, runs=runs)

@app.route('/history')
def history():
    try:
        df = pd.read_csv('predictions.csv', header=None)
        df.columns = ['Timestamp', 'Batting Team', 'Bowling Team', 'Overs', 'Runs', 'Wickets', 'Predicted Score']
        return render_template('history.html', tables=df.tail(20).to_html(classes='table table-striped', index=False))
    except FileNotFoundError:
        return "No predictions yet."

@app.route('/win_predictor')
def win_predictor():
    return render_template('win.html', teams=teams)

@app.route('/win_result', methods=['POST'])
def win_result():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    runs_left = int(request.form['runs_left'])
    balls_left = int(request.form['balls_left'])
    wickets = int(request.form['wickets'])
    target = int(request.form['target'])

    # Extra calculated fields
    run_rate = (target - runs_left) / ((120 - balls_left) / 6) if balls_left != 120 else 0
    required_run_rate = runs_left / (balls_left / 6) if balls_left != 0 else 0

    input_dict = {
        'batting_team': batting_encoder.transform([batting_team])[0],
        'bowling_team': bowling_encoder.transform([bowling_team])[0],
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets': wickets,
        'target': target,
        'run_rate': run_rate,
        'required_run_rate': required_run_rate
    }

    final_input = pd.DataFrame([input_dict])
    prediction = win_model.predict(final_input)[0]
    prob = win_model.predict_proba(final_input)[0][1]

    msg = "🟢 Win Likely ✅" if prediction == 1 else "🔴 Loss Likely ❌"

    return render_template('win.html', teams=teams, result=msg, prob=round(prob * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)
