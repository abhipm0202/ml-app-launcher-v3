import streamlit as st
from PIL import Image
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="ML Toolkit", layout="wide")

# Load logos
nmis_logo = Image.open("assets/nmis_logo.png")
colab_logo = Image.open("assets/d3mcolab_logo.png")

# Header layout
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image(nmis_logo, use_container_width=True)
with col2:
    st.markdown("<h1 style='text-align: center;'>ML Toolkit</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Developed by D3M Colab</h4>", unsafe_allow_html=True)
with col3:
    st.image(colab_logo, use_container_width=True)

st.markdown("---")

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_xpfdvvvu.json"
lottie_json = load_lottieurl(lottie_url)

# Main layout with animation and welcome message
left, right = st.columns(2)
with left:
    st_lottie(lottie_json, height=300, speed=1)

with right:
    st.markdown("""
    ### 👋 Welcome!!

    This toolkit offers a suite of easy-to-use machine learning trainers:

    - 🔢 **ANN Trainer** – For numerical prediction using feedforward neural networks.
    - 🧠 **CNN Trainer** – For image classification using convolutional neural networks.
    - 📈 **LSTM Forecaster** – For multivariate, multi-step time series forecasting.

    Use the sidebar to navigate between tools. You can also download a sample dataset on each page to get started quickly.
    """)
