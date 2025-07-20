#!/usr/bin/env python3
import sys
import os
import subprocess



# Now import all required packages
try:
    from flask import Flask, request, render_template, redirect, url_for, session, flash
    import pickle
    from pathlib import Path
    import logging
    from logging.handlers import RotatingFileHandler
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import requests
    from dotenv import load_dotenv
    from werkzeug.security import generate_password_hash, check_password_hash
    from flask_sqlalchemy import SQLAlchemy
    from urllib.parse import quote_plus
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Please ensure all packages are installed correctly.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# === Load environment variables ===
load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "7d3947dfa424c4f8ad6cb668c055aa81")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.getenv("POSTGRES_PASS", "manthan@2004")
POSTGRES_DB = os.getenv("POSTGRES_DB", "Newsdatabase")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
SECRET_KEY = os.getenv("SECRET_KEY", "my-secret")

# URL encode the password to handle special characters like @
POSTGRES_PASS_ENCODED = quote_plus(POSTGRES_PASS)

# === Project directories ===
current_dir = Path(__file__).resolve().parent
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')
data_dir = os.path.join(current_dir, 'data')
model_path = os.path.join(current_dir, 'models', 'model.pkl')

# === Flask App Setup ===
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS_ENCODED}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# === Logging ===
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# === Train and Save Model ===
def train_and_save_model():
    fake_df = pd.read_csv(os.path.join(data_dir, 'fake.csv'))
    real_df = pd.read_csv(os.path.join(data_dir, 'real.csv'))

    fake_df['label'] = 'FAKE'
    real_df['label'] = 'REAL'
    data = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=42)

    X = data['text']
    y = data['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


# === Load or Train Model ===
def load_or_retrain_model():
    """Load existing model or retrain if there are version compatibility issues."""
    should_retrain = False
    
    try:
        if os.path.exists(model_path):
            print("📁 Loading existing model...")
            # Capture warnings to detect version mismatches
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Check for version warnings
                if w and any("version" in str(warning.message).lower() for warning in w):
                    print("⚠️ Model version compatibility issues detected!")
                    should_retrain = True
                else:
                    print("✅ Model loaded successfully!")
                    return model
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        should_retrain = True
    
    if should_retrain or not os.path.exists(model_path):
        # Train new model if loading failed or model doesn't exist
        print("🔄 Retraining model with current scikit-learn version...")
        try:
            train_and_save_model()
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ New model trained and loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error training model: {e}")
            raise

model = load_or_retrain_model()

# === Helper: PostgreSQL Connection with Error Handling ===
def create_database_if_not_exists():
    """Create database if it doesn't exist using PostgreSQL connection."""
    try:
        # Connect to default postgres database first
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASS,
            database="postgres"  # Connect to default postgres database
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{POSTGRES_DB}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
            print(f"✅ Database '{POSTGRES_DB}' created successfully!")
        else:
            print(f"✅ Database '{POSTGRES_DB}' already exists!")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Error creating PostgreSQL database: {e}")
        print("\n🔧 PostgreSQL Setup Instructions:")
        print("="*60)
        print("1. Make sure PostgreSQL is installed and running")
        print("2. Open pgAdmin or psql command line")
        print("3. Connect with your credentials:")
        print(f"   Host: {POSTGRES_HOST}")
        print(f"   Port: {POSTGRES_PORT}")
        print(f"   Username: {POSTGRES_USER}")
        print(f"   Password: {POSTGRES_PASS}")
        print("4. Create database manually if needed:")
        print(f"   CREATE DATABASE \"{POSTGRES_DB}\";")
        print("="*60)
        return False

def get_db_connection():
    """Get PostgreSQL database connection with comprehensive error handling."""
    try:
        return psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASS,
            database=POSTGRES_DB
        )
    except psycopg2.Error as e:
        error_code = e.pgcode if hasattr(e, 'pgcode') else 'Unknown'
        if "could not connect to server" in str(e):
            print("❌ PostgreSQL Connection Error: PostgreSQL server is not running!")
            print("🔧 Solutions:")
            print("   1. Start PostgreSQL service")
            print("   2. Check if PostgreSQL is running on port 5432")
            print("   3. Verify server is installed and configured")
        elif "database" in str(e) and "does not exist" in str(e):
            print(f"❌ Database '{POSTGRES_DB}' does not exist!")
            print("🔧 Creating database automatically...")
            if create_database_if_not_exists():
                # Try connecting again after creating database
                return psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASS,
                    database=POSTGRES_DB
                )
        elif "authentication failed" in str(e):
            print("❌ Authentication failed! Check PostgreSQL username/password")
            print("🔧 Verify these credentials:")
            print(f"   Username: {POSTGRES_USER}")
            print(f"   Password: {POSTGRES_PASS}")
            print("🔧 Update your .env file with correct credentials:")
            print("   POSTGRES_USER=postgres")
            print("   POSTGRES_PASS=manthan@2004")
        else:
            print(f"❌ PostgreSQL Error ({error_code}): {e}")
        raise

def safe_db_operation(operation_func, *args, **kwargs):
    """Safely execute database operations with error handling."""
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        app.logger.error(f"DB Operation Error: {e}")
        return None

# === Helper: Insert Prediction ===
def insert_prediction(user_id, news_text, prediction):
    def _insert():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_predictions (user_id, news_text, prediction) VALUES (%s, %s, %s)",
                       (user_id, news_text, prediction))
        conn.commit()
        cursor.close()
        conn.close()
    
    return safe_db_operation(_insert)

# === Helper: NewsAPI Fetch ===
def fetch_news_articles(query):
    try:
        res = requests.get("https://gnews.io/api/v4/search", params={
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": GNEWS_API_KEY
        })
        return res.json().get("articles", []) if res.status_code == 200 else []
    except Exception as e:
        app.logger.error(f"NewsAPI Error: {e}")
        return []

# === Flask Routes ===
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash("Please fill in all fields.")
            return render_template("register.html")
        
        # Validate username length
        if len(username) < 3:
            flash("Username must be at least 3 characters long.")
            return render_template("register.html")
        
        # Validate password length
        if len(password) < 6:
            flash("Password must be at least 6 characters long.")
            return render_template("register.html")

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                flash("Username already exists. Please choose a different username.")
                cursor.close()
                conn.close()
                return render_template("register.html")
            
            # Check if email already exists
            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                flash("Email already exists. Please use a different email.")
                cursor.close()
                conn.close()
                return render_template("register.html")
            
            # Insert new user with hashed password
            hashed_password = generate_password_hash(password)
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                           (username, email, hashed_password))
            conn.commit()
            cursor.close()
            conn.close()
            
            flash("Registration successful! Please log in with your username and password.")
            return redirect(url_for('login'))
            
        except psycopg2.Error as e:
            flash("Database error occurred. Please try again.")
            app.logger.error(f"Registration error: {e}")
        except Exception as e:
            flash("An error occurred during registration. Please try again.")
            app.logger.error(f"Registration error: {e}")
    
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash("Please enter both username and password.")
            return render_template("login.html")

        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Find user by username
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user and check_password_hash(user['password'], password):
                # Successful login
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['email'] = user['email']
                
                flash(f"Welcome back, {user['username']}!")
                return redirect(url_for('index'))
            else:
                flash("Invalid username or password. Please try again.")
                
        except psycopg2.Error as e:
            flash("Database error occurred. Please try again.")
            app.logger.error(f"Login error: {e}")
        except Exception as e:
            flash("An error occurred during login. Please try again.")
            app.logger.error(f"Login error: {e}")
    
    return render_template("login.html")

@app.route('/logout')
def logout():
    username = session.get('username', 'User')
    session.clear()
    flash(f"Goodbye, {username}! You have been logged out successfully.")
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash("Please log in to make predictions.")
        return redirect(url_for('login'))

    news_text = request.form.get('news_text')
    if not news_text:
        return render_template("index.html", error="Please enter news text.")
    prediction = model.predict([news_text])[0]
    insert_prediction(session['user_id'], news_text, prediction)
    return render_template("index.html", prediction=prediction, input_text=news_text)

@app.route('/search', methods=['POST'])
def search_news():
    if 'user_id' not in session:
        flash("Please log in to search and predict.")
        return redirect(url_for('login'))

    query = request.form.get('query')
    if not query:
        return render_template("index.html", error="Enter a keyword to search.")

    articles = fetch_news_articles(query)
    results = []

    for article in articles:
        title = article.get("title", "")
        url = article.get("url", "")
        if title:
            prediction = model.predict([title])[0]
            insert_prediction(session['user_id'], title, prediction)
            results.append({"title": title, "url": url, "prediction": prediction})

    return render_template("index.html", search_results=results, query=query)

@app.route('/my_predictions')
def my_predictions():
    if 'user_id' not in session:
        flash("Login to view your predictions.")
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM user_predictions WHERE user_id = %s ORDER BY predicted_at DESC", (session['user_id'],))
    predictions = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("my_predictions.html", predictions=predictions)

@app.route('/news-details', methods=['POST'])
def get_news_details():
    news_text = request.form.get('news_text')
    if not news_text:
        return render_template("index.html", error="Please enter news text.")
    
    # Fetch relevant articles
    articles = fetch_news_articles(news_text)
    
    if not articles:
        return render_template("index.html", 
                             error="No additional information found for this news.",
                             input_text=news_text)
    
    # Get the most relevant article
    article = articles[0]
    
    # Format the date
    if article.get("publishedAt"):
        from datetime import datetime
        date = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
        article["publishedAt"] = date.strftime("%B %d, %Y")
    
    # Map GNews article fields to match our template
    formatted_article = {
        "title": article.get("title", ""),
        "author": article.get("source", {}).get("name", ""),
        "publishedAt": article.get("publishedAt", ""),
        "description": article.get("description", ""),
        "content": article.get("content", ""),
        "urlToImage": article.get("image", ""),
        "url": article.get("url", ""),
        "source": {
            "name": article.get("source", {}).get("name", "")
        }
    }
    
    return render_template("news_details.html", article=formatted_article)

# === Error Pages ===
@app.errorhandler(404)
def not_found(e):
    return f"<h1>404 - Page Not Found</h1><p>The page you are looking for does not exist.</p><a href='{url_for('index')}'>Go Home</a>", 404

@app.errorhandler(500)
def server_error(e):
    return f"<h1>500 - Internal Server Error</h1><p>Something went wrong on our end.</p><a href='{url_for('index')}'>Go Home</a>", 500

# === Database Tables Creation ===
def create_database_tables():
    """Create necessary database tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create users table with proper constraints
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP DEFAULT NULL
            )
        """)
        
        # Create user_predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_predictions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                news_text TEXT NOT NULL,
                prediction VARCHAR(10) NOT NULL,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON user_predictions(user_id);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ PostgreSQL database tables created/verified successfully!")
        
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        print("🔧 Manual database setup may be required.")

# Add helper function to update last login
def update_last_login(user_id):
    """Update user's last login timestamp."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s", (user_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        app.logger.error(f"Error updating last login: {e}")

# === Run App ===
try:
    print("🗄️ Setting up PostgreSQL database connection...")
    
    # First, ensure database exists
    create_database_if_not_exists()
    
    with app.app_context():
        print("🗄️ Setting up database tables...")
        create_database_tables()
        db.create_all()
        print("✅ Database setup completed!")
        
    if __name__ == '__main__':
        print("🚀 Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
        
except Exception as e:
    print(f"❌ Application startup error: {e}")
    print("\n🔧 Manual Steps for PostgreSQL:")
    print("1. Make sure PostgreSQL is installed and running")
    print("2. Open pgAdmin or connect via psql")
    print("3. Create database manually if needed:")
    print(f"   CREATE DATABASE \"{POSTGRES_DB}\";")
    print("4. Restart the application")
