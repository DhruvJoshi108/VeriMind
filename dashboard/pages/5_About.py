"""Dashboard project information without exposing credentials."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.config import display_api_origin, load_config
from dashboard.lib.session import connection_sidebar

connection_sidebar()
config = load_config()
st.title("About VERIMIND")
st.write("VERIMIND segments factual claims, retrieves evidence, and returns an evidence-bound verification report through its FastAPI service.")
st.subheader("Connection")
st.write(display_api_origin(config.api_url))
st.caption("The dashboard never displays or logs the API key. A key from the environment is copied only into session state.")
st.subheader("Version")
st.write("Dashboard/API integration version 0.2.0")
st.subheader("Links")
origin = display_api_origin(config.api_url)
if origin.startswith(("http://", "https://")):
    st.link_button("API health", f"{origin}/healthz")
    st.link_button("API metrics", f"{origin}/metrics")
else:
    st.caption("API links are unavailable until a valid HTTP(S) URL is configured.")
st.caption("See `README.md` for Windows setup, environment variables, deployment, and known limitations.")
