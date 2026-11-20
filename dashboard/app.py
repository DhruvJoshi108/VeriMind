"""VERIMIND dashboard home page; Streamlit discovers the pages directory."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.verify import render_verify_page

st.set_page_config(page_title="VERIMIND", page_icon="V", layout="wide", initial_sidebar_state="expanded")
render_verify_page()

