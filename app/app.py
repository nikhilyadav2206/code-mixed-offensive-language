from flask import Flask
from utils import init_db  # Only import needed functions here

app = Flask(__name__)

# Initialize DB once
init_db()

@app.route("/")
def home():
    return "Welcome to the Multilingual Offensive & Hate Speech Detection System"

if __name__ == "__main__":
    # debug=True is fine now; use_reloader=False avoids signal issues on Windows
    app.run(debug=False, use_reloader=False)
