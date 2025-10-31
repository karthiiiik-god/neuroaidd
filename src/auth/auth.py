# src/auth/auth.py
import streamlit as st
from src.db.store import get_or_create_user

def login_block():
    """
    Handles user login with a single password field and optional username.
    Saves user info into st.session_state['user'] after successful login.
    """

    st.subheader("🔐 Login to NeuroAid")

    # Ask for username (optional but unique)
    username = st.text_input("Username", value=st.session_state.get("last_username", ""), key="username_input")
    password = st.text_input("Password", type="password", key="password_input")

    # Fake simple password (you can add real hash later)
    VALID_PASSWORD = "1234"

    if st.button("Login"):
        if not username.strip():
            st.warning("Please enter a username.")
        elif password != VALID_PASSWORD:
            st.error("❌ Incorrect password. Try again!")
        else:
            user = get_or_create_user(username.strip())
            st.session_state.user = user
            st.session_state.last_username = username.strip()
            st.success(f"✅ Welcome, {username.strip()}! Login successful.")
            st.toast("Login successful 🎉", icon="🎯")
            st.rerun()
