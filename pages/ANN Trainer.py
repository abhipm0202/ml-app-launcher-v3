import streamlit as st
import ann_gui

# Sidebar sample data section
st.sidebar.title("📥 Sample Data")
# Download button for ANN_trainX.csv
with open("sample_data/ann_sample.csv", "rb") as f:
    st.sidebar.download_button("Download ANN Training Input (X)", f, file_name="ANN_trainX.csv")

# Download button for ANN_trainY.csv
with open("sample_data/ann_sample_y.csv", "rb") as f:
    st.sidebar.download_button("Download ANN Training Output (Y)", f, file_name="ANN_trainY.csv")

# Run the actual ANN GUI
ann_gui.run_ann_gui()
