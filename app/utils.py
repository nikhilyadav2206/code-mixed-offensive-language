"""import sqlite3
import os
import requests
from typing import Tuple, List, Dict
from hashlib import sha256
from config import (
    TWITTER_BEARER_TOKEN,
    INSTAGRAM_ACCESS_TOKEN,
)

# ✅ Import actual model-based prediction
from predict import smart_predict

# ---------- Safe Database Path Setup ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)
DATABASE_PATH = os.path.join(data_dir, 'database.db')

# ---------- Database Utilities -----------
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        text TEXT,
        context TEXT,
        language TEXT,
        prediction TEXT,
        confidence REAL,
        platform TEXT,
        post_id TEXT,
        flagged INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()

# -------- User Authentication -----------
def hash_password(password: str) -> str:
    return sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        conn.close()
        return False
    password_hash = hash_password(password)
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
    conn.commit()
    conn.close()
    return True

def authenticate_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return False
    return row["password_hash"] == hash_password(password)

# -------- Twitter API --------
def fetch_and_predict_twitter(username: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    # Step 1: Get User ID
    user_url = f"https://api.twitter.com/2/users/by/username/{username}"
    try:
        user_response = requests.get(user_url, headers=headers)
        user_response.raise_for_status()
        user_data = user_response.json()
        user_id = user_data.get("data", {}).get("id")
        if not user_id:
            print(f"User ID not found for username: {username}")
            return []
    except Exception as e:
        print(f"Error fetching user ID: {e}")
        return []

    # Step 2: Fetch Tweets
    tweets_url = (
        f"https://api.twitter.com/2/users/{user_id}/tweets"
        "?max_results=20&tweet.fields=id,text,created_at"
        "&exclude=retweets,replies"
    )
    try:
        tweets_response = requests.get(tweets_url, headers=headers)
        tweets_response.raise_for_status()
        tweets_data = tweets_response.json()
        tweets = tweets_data.get("data", [])
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

    results = []
    for tweet in tweets:
        text = tweet.get("text", "")
        label, confidence = smart_predict(text)
        results.append({
            "id": tweet.get("id"),
            "text": text,
            "label": label,
            "confidence": confidence,
        })
    return results

def delete_twitter_post(post_id: str) -> bool:
    return True  # Placeholder

# -------- Instagram API --------
def fetch_and_predict_instagram(user_id: str) -> List[Dict]:
    url = f"https://graph.instagram.com/me/media?fields=id,caption&access_token={INSTAGRAM_ACCESS_TOKEN}&limit=10"
    try:
        response = requests.get(url)
        response.raise_for_status()
        media = response.json().get("data", [])
    except Exception as e:
        print(f"Instagram API error: {e}")
        return []

    results = []
    for post in media:
        text = post.get("caption", "")
        label, confidence = smart_predict(text)
        results.append({
            "id": post.get("id"),
            "text": text,
            "label": label,
            "confidence": confidence,
        })
    return results

def delete_instagram_post(post_id: str) -> bool:
    url = f"https://graph.instagram.com/{post_id}?access_token={INSTAGRAM_ACCESS_TOKEN}"
    try:
        response = requests.delete(url)
        return response.status_code == 200
    except Exception as e:
        print(f"Instagram deletion error: {e}")
        return False"""
import sqlite3
import os
import requests
from typing import Tuple, List, Dict
from hashlib import sha256
from config import (
    TWITTER_BEARER_TOKEN,
)

# ✅ Import actual model-based prediction
from predict import smart_predict

# ---------- Safe Database Path Setup ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)
DATABASE_PATH = os.path.join(data_dir, 'database.db')

# ---------- Database Utilities -----------
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        text TEXT,
        context TEXT,
        language TEXT,
        prediction TEXT,
        confidence REAL,
        platform TEXT,
        post_id TEXT,
        flagged INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()

# -------- User Authentication -----------
def hash_password(password: str) -> str:
    return sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        conn.close()
        return False
    password_hash = hash_password(password)
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
    conn.commit()
    conn.close()
    return True

def authenticate_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return False
    return row["password_hash"] == hash_password(password)

# -------- Twitter API --------
def fetch_and_predict_twitter(username: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    # Step 1: Get User ID
    user_url = f"https://api.twitter.com/2/users/by/username/{username}"
    try:
        user_response = requests.get(user_url, headers=headers)
        user_response.raise_for_status()
        user_data = user_response.json()
        user_id = user_data.get("data", {}).get("id")
        if not user_id:
            print(f"User ID not found for username: {username}")
            return []
    except Exception as e:
        print(f"Error fetching user ID: {e}")
        return []

    # Step 2: Fetch Tweets
    tweets_url = (
        f"https://api.twitter.com/2/users/{user_id}/tweets"
        "?max_results=20&tweet.fields=id,text,created_at"
        "&exclude=retweets,replies"
    )
    try:
        tweets_response = requests.get(tweets_url, headers=headers)
        tweets_response.raise_for_status()
        tweets_data = tweets_response.json()
        tweets = tweets_data.get("data", [])
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

    results = []
    for tweet in tweets:
        text = tweet.get("text", "")
        label, confidence = smart_predict(text)
        results.append({
            "id": tweet.get("id"),
            "text": text,
            "label": label,
            "confidence": confidence,
        })
    return results

def delete_twitter_post(post_id: str) -> bool:
    return True  # Placeholder