import streamlit as st
import lstm_gui

# Sidebar sample data section
st.sidebar.title("📥 Sample Data")
st.sidebar.markdown("Download a sample LSTM time series file:")
st.sidebar.markdown(
    "- [🔗 lstm_sample.csv](https://github.com/abhipm0202/ml-app-launcher-clean/blob/main/sample_data/lstm_sample.csv)"
)

# Launch the actual GUI
lstm_gui.run_lstm_gui()
