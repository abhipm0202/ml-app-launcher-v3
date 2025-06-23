import streamlit as st
from PIL import Image
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="ML Toolkit", layout="wide")

# --- Load logos ---
nmis_logo = Image.open("assets/nmis_logo.png")
d3m_logo = Image.open("assets/d3mcolab_logo.png")

# --- Load Lottie animation ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"⚠️ Failed to load animation. HTTP {r.status_code}")
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading animation: {e}")
        return None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_zrqthn6o.json"
lottie_json = load_lottieurl(lottie_url)

# --- Layout ---
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.image(nmis_logo, use_container_width=True)

with col2:
    st.markdown("<h1 style='text-align: center;'>ML Toolkit</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Developed by D3M Colab</h4>", unsafe_allow_html=True)

with col3:
    st.image(d3m_logo, use_container_width=True)

st.markdown("---")

col_left, col_anim = st.columns([2, 1])

with col_left:
    st.markdown("### 👋 Welcome!!")
    st.markdown("""
This toolkit offers a suite of easy-to-use machine learning trainers:

- 🔢 **ANN Trainer** – For numerical prediction using feedforward neural networks.  
- 🧠 **CNN Trainer** – For image classification using convolutional neural networks.  
- 📈 **LSTM Forecaster** – For multivariate, multi-step time series forecasting.

Use the sidebar to navigate between tools.  
Each page offers a sample dataset you can use to get started quickly.
""")

with col_anim:
    if lottie_json:
        st_lottie(lottie_json, height=300, speed=1)
    else:
        st.info("Lottie animation could not be loaded.")

