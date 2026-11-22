"""Shared session-only connection controls for each Streamlit page."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.api_client import ApiClient
from dashboard.lib.config import load_config
from dashboard.lib.ui import apply_dashboard_theme


def client() -> ApiClient:
    config = load_config()
    return ApiClient(
        st.session_state.get("dashboard_api_url") or config.api_url,
        st.session_state.get("dashboard_api_key", config.api_key),
        config.timeout_seconds,
    )


def _save_connection_inputs() -> None:
    st.session_state["dashboard_api_url"] = st.session_state["dashboard_api_url_input"]
    st.session_state["dashboard_api_key"] = st.session_state["dashboard_api_key_input"]
    st.session_state.pop("dashboard_connection", None)
    st.session_state.pop("dashboard_connection_url", None)


def connection_sidebar() -> None:
    apply_dashboard_theme()
    config = load_config()
    st.sidebar.markdown(
        """
        <div class="vm-brand">
            <div class="vm-brand-name">VERIMIND</div>
            <div class="vm-brand-caption">Evidence-grounded claim verification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.header("Connection")
    st.session_state.setdefault("dashboard_api_url", config.api_url)
    st.session_state.setdefault("dashboard_api_key", config.api_key)
    st.session_state.setdefault("dashboard_api_url_input", st.session_state["dashboard_api_url"])
    st.session_state.setdefault("dashboard_api_key_input", st.session_state["dashboard_api_key"])
    st.sidebar.text_input(
        "API URL",
        key="dashboard_api_url_input",
        on_change=_save_connection_inputs,
        help="Base URL only; do not append /verify.",
    )
    st.sidebar.text_input(
        "API key",
        type="password",
        key="dashboard_api_key_input",
        on_change=_save_connection_inputs,
        help="Used for verification requests; not sent to health or metrics endpoints.",
    )
    test_connection = st.sidebar.button("Test connection", use_container_width=True)
    api_url = st.session_state["dashboard_api_url"]
    result = st.session_state.get("dashboard_connection")
    if test_connection or result is None or st.session_state.get("dashboard_connection_url") != api_url:
        if test_connection:
            with st.sidebar:
                with st.spinner("Checking API connection..."):
                    result = client().health()
        else:
            result = client().health()
        st.session_state["dashboard_connection"] = result
        st.session_state["dashboard_connection_url"] = api_url
    if result.ok:
        st.sidebar.success("API connected")
    else:
        st.sidebar.error(f"API unavailable: {result.error.message}")
        st.sidebar.caption(f"Request ID: {result.request_id}")
