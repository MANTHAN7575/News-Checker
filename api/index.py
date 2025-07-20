from flask import Flask, request, render_template, redirect, url_for, session, flash
import pickle
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'your-secret-key'  # Change this to a secure secret key

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    handlers=[RotatingFileHandler('logs/app.log', maxBytes=100000, backupCount=3)],
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

# Load the model
try:
    model_path = Path('models/model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        if model:
            try:
                prediction = model.predict([news])
                return render_template('news_details.html', prediction=prediction[0], news=news)
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                flash("An error occurred during prediction", "error")
                return redirect(url_for('home'))
        else:
            flash("Model not loaded", "error")
            return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
