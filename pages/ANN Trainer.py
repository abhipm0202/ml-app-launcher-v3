import ann_gui
import streamlit as st

st.sidebar.title("📥 Sample Data")

# Download button for ANN_trainX.csv
with open("sample_data/ANN_trainX.csv", "rb") as f:
    st.sidebar.download_button("Download ANN Training Input (X)", f, file_name="ANN_trainX.csv")

# Download button for ANN_trainY.csv
with open("sample_data/ANN_trainY.csv", "rb") as f:
    st.sidebar.download_button("Download ANN Training Output (Y)", f, file_name="ANN_trainY.csv")
