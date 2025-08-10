""" Streamlit UI to connect to webservice and call API """

import pandas as pd
import streamlit as st
import requests

API_HOST_URL = "http://localhost:8007"

st.title("Headline Sentiment Predictor")

inputs = st.text_area("Enter headlines (one per line):")

if st.button("Enter"):

    headlines_list = [h.strip() for h in inputs.split("\n") if h.strip()]

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
