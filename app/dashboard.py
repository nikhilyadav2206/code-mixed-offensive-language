import streamlit as st
import pandas as pd
import plotly.express as px
from langdetect import detect, LangDetectException
from utils import (
    init_db, authenticate_user, register_user, smart_predict,
    fetch_and_predict_twitter,delete_twitter_post,
    get_db_connection
)

# Initialize database on start
init_db()

st.set_page_config(page_title="Code-Mixed Moderation Dashboard", layout="wide")

# --- Session Defaults ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'twitter_posts' not in st.session_state:
    st.session_state.twitter_posts = []
if 'instagram_posts' not in st.session_state:
    st.session_state.instagram_posts = []
if 'page_reload_trigger' not in st.session_state:
    st.session_state.page_reload_trigger = False  # Used to trigger UI updates after login/logout

# --- Helpers ---
def colored_label(label: str) -> str:
    color_map = {
        "non-offensive": "green",
        "offensive": "orange"
    }
    color = color_map.get(label.lower(), "gray")
    return f'<span style="color:{color}; font-weight:bold;">{label.upper()}</span>'

# --- Views ---
def login_view():
    st.title("🔐 Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.session_state.page_reload_trigger = not st.session_state.page_reload_trigger
        else:
            st.error("Invalid username or password.")

def register_view():
    st.title("📝 Register")
    username = st.text_input("Choose username", key="register_username")
    password = st.text_input("Choose password", type="password", key="register_password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("User registered successfully! Please login.")
        else:
            st.error("Username already exists. Try a different one.")

def predict_page():
    st.title("Code-Mixed Offensive Language Detection")
    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area("Enter text or post content", key="predict_text")
    with col2:
        context = st.text_input("Optional context", key="predict_context")

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        label, confidence = smart_predict(text, context)
        st.markdown(f"### 🧠 Prediction: {colored_label(label)}", unsafe_allow_html=True)
        st.markdown("#### 🔍 Confidence Scores:")
        st.json({
            "non-offensive": round(confidence[0], 3),
            "offensive": round(confidence[1], 3)
        })

def twitter_moderation_page():
    st.title("🐦 Moderate Twitter Posts")
    handle = st.text_input("Enter Twitter handle (without @)", key="twitter_handle")
    if st.button("Fetch Tweets"):
        if not handle.strip():
            st.warning("Please enter a Twitter handle.")
            return
        posts = fetch_and_predict_twitter(handle)
        st.session_state.twitter_posts = posts

        with get_db_connection() as conn:
            user_id = conn.execute("SELECT id FROM users WHERE username=?", (st.session_state.username,)).fetchone()["id"]
            for post in posts:
                text = post.get("text", "")
                language = "unknown"
                if text and text.strip():
                    try:
                        language = detect(text)
                    except LangDetectException:
                        language = "unknown"
                conn.execute('''
                    INSERT INTO posts (user_id, text, context, language, prediction, confidence, platform, post_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    text,
                    "",
                    language,
                    post["label"],
                    max(post["confidence"]),
                    "twitter",
                    post["id"]
                ))
            conn.commit()

    if st.session_state.twitter_posts:
        for i, post in enumerate(st.session_state.twitter_posts):
            st.write(f"**Tweet ID:** {post['id']}")
            st.write(post["text"])
            st.markdown(f"**Prediction:** {colored_label(post['label'])}", unsafe_allow_html=True)
            if st.button(f"Delete Tweet {post['id']}", key=f"del_tw_{post['id']}"):
                success = delete_twitter_post(post['id'])
                if success:
                    st.success(f"Tweet {post['id']} deleted!")
                    with get_db_connection() as conn:
                        conn.execute("UPDATE posts SET flagged=1 WHERE post_id=?", (post['id'],))
                        conn.commit()
                    st.session_state.twitter_posts.pop(i)
                    st.session_state.page_reload_trigger = not st.session_state.page_reload_trigger
                else:
                    st.error(f"Failed to delete Tweet {post['id']}.")



def analytics_page():
    st.title("📊 Offense Analytics Dashboard")
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM posts", conn)

    if df.empty:
        st.info("No posts data available for analytics.")
        return

    fig_pie = px.pie(df, names='prediction', title='Prediction Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_hist = px.histogram(df, x='language', color='prediction', barmode='group',
                            title='Offense Count by Language')
    st.plotly_chart(fig_hist, use_container_width=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.twitter_posts = []
    st.session_state.instagram_posts = []
    st.success("Logged out successfully!")
    st.session_state.page_reload_trigger = not st.session_state.page_reload_trigger

# --- Main App Router ---
def main():
    if not st.session_state.logged_in:
        st.sidebar.title("Account")
        mode = st.sidebar.radio("Choose:", ["Login", "Register"])
        if mode == "Login":
            login_view()
        else:
            register_view()
        st.stop()

    st.sidebar.title(f"Welcome, {st.session_state.username}")
    page = st.sidebar.radio("Navigate to:", ["Predict", "Twitter", "Analytics", "Logout"])

    if page == "Predict":
        predict_page()
    elif page == "Twitter":
        twitter_moderation_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Logout":
        logout()

if __name__ == "__main__":
    main()
