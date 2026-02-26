import streamlit as st

st.set_page_config(page_title="DPC Demo", layout="wide")

st.title("Driver Performance Contracts (DPC) Demo")

st.markdown("""
Welcome to the **Driver Performance Contracts (DPC)** live demonstration platform.

Use the sidebar to navigate between:

- **Admin Input** – Manage race state, odds, and publish price updates.
- **Stakeholder View** – Clean, live mark-to-market dashboard with price history.

This demo showcases how driver finishing-position contracts can be
financialized and marked in real time.
""")

st.info("Navigate using the left sidebar to begin.")
