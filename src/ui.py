"""Streamlit layout helpers — avoid deprecated use_container_width."""

import streamlit as st

# Streamlit 1.58+: prefer width="stretch" over use_container_width=True
DATAFRAME_KW = {"width": "stretch"}
PLOTLY_KW = {"width": "stretch"}


def show_dataframe(df, *, hide_index: bool = False, **kwargs):
    opts = {**DATAFRAME_KW, **kwargs}
    if hide_index:
        opts["hide_index"] = True
    opts.pop("use_container_width", None)
    return st.dataframe(df, **opts)


def show_plotly(fig, **kwargs):
    opts = {**PLOTLY_KW, **kwargs}
    opts.pop("use_container_width", None)
    return st.plotly_chart(fig, **opts)
