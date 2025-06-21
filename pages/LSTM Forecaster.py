import streamlit as st
import lstm_gui

# Sidebar sample data section
st.sidebar.title("📥 Sample Data")
st.sidebar.markdown("Download a sample LSTM time series file:")
with open("sample_data/Timeseries_train.csv", "rb") as f:
    st.sidebar.download_button("Download LSTM Sample", f, file_name="Timeseries_train.csv")
# Launch the actual GUI
lstm_gui.run_lstm_gui()
