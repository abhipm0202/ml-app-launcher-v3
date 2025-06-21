import cnn_gui
import streamlit as st

st.sidebar.title("📥 Sample Data")

st.sidebar.markdown(
    """
    🔗 [Click here to access the CNN sample dataset](https://github.com/Charmve/Surface-Defect-Detection/tree/master/Magnetic-Tile-Defect)

    Dataset Source: Charmve / Surface-Defect-Detection
    """,
    unsafe_allow_html=True
)

# Main app logic
cnn_gui.run_cnn_gui()