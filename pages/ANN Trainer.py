import streamlit as st
import ann_gui

# Sidebar sample data section
st.sidebar.title("📥 Sample Data")
st.sidebar.markdown(
    "Download sample training files:"
)
st.sidebar.markdown(
    "- [🔗 ANN_trainX.csv](https://github.com/abhipm0202/ml-app-launcher-clean/blob/main/sample_data/ANN_trainX.csv)"
)
st.sidebar.markdown(
    "- [🔗 ANN_trainY.csv](https://github.com/abhipm0202/ml-app-launcher-clean/blob/main/sample_data/ANN_trainY.csv)"
)

# Run the actual ANN GUI
ann_gui.run_ann_gui()
