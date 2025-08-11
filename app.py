""" Streamlit UI to connect to webservice and call API """

import pandas as pd
import streamlit as st
import requests

API_HOST_URL = "http://localhost:8007"

st.title("Headline Sentiment Predictor")

# input below from MVP, one large text area. Used for testing
# inputs = st.text_area("Enter headlines (one per line):")

if "headlines" not in st.session_state:
    st.session_state.headlines = [""]

def add_headline():
    """Helper function to dynamically add headlines"""
    st.session_state.headlines.append("")

def remove_headline(pos):
    """Helper function to dynamically remove headlines"""
    st.session_state.headlines.pop(pos)

for i, headline in enumerate(st.session_state.headlines):
    cols = st.columns([9, 1])
    with cols[0]:
        st.session_state.headlines[i] = st.text_input(f"Headline {i+1}",
                                                      value=headline, key=f"headline_{i}")
    with cols[1]:
        if st.button("   Delete   ", key=f"del_{i}"):
            remove_headline(i)
            st.rerun()

st.button("Add Headline", on_click=add_headline)

if st.button("Get Sentiment"):

    headlines_list = [h.strip() for h in st.session_state.headlines if h.strip()]

    if not headlines_list:
        st.error("Please enter at least one headline.")
    else:
        try:
            response = requests.post(f"{API_HOST_URL}/score_headlines",
                json={"headlines": headlines_list},
                timeout=30
            )
            if response.status_code == 200:
                st.success("Headline Sentiment:")
                df = pd.DataFrame({
                        "Headline": headlines_list,
                        "Prediction": response.json().get("predictions")
                    })
                st.table(df)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API. Please check connection")
