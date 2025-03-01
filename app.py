import streamlit as st
from fnews import manual_testing, train_model

# Cache the model to avoid reloading it each time
@st.cache_resource
def load_model():
    model, vectorizer, accuracy = train_model()
    return model, vectorizer, accuracy

# Enhanced CSS styling for a sleek UI with custom font for title
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap'); /* Custom font */

    body {
        background: linear-gradient(to right, #ece9e6, #ffffff);
    }
    .main {
        background-color: #fff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        width: 85%;
        margin: 0 auto;
    }
    h1 {
        color: #3498db; /* Custom color */
        font-family: 'Arial Rounded MT Bold', cursive; /* Custom font */
        font-weight: 600;
        text-align: center;
        margin-bottom: 40px;
        font-size: 42px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-size: 20px;
        padding: 12px 18px;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .result-box {
        border-radius: 50px;
        padding: 20px;
        font-size: 18px;
        background-color: white;
        width: 120px;
        height: 120px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 30px auto;
        font-weight: bold;
    }
    .fake-result-box {
        border: 3px solid #e74c3c;
        color: #e74c3c;
        background-color: rgba(231, 76, 60, 0.1);
    }
    .real-result-box {
        border: 3px solid #2ecc71;
        color: #2ecc71;
        background-color: rgba(46, 204, 113, 0.1);
    }
    footer {
        text-align: center;
        margin-top: 50px;
        color: #7f8c8d;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title with stylish formatting and custom font
st.markdown("<h1> NewsNexus 📰</h1>", unsafe_allow_html=True)

# Load the cached model and vectorizer
model, vectorizer, accuracy = load_model()

# Display model accuracy with formatted text
st.markdown(f"<h4>🔍 Model Accuracy: <span style='color: #16a085;'>{accuracy * 100:.2f}%</span></h4>", unsafe_allow_html=True)

# Input text box for entering the news
news_article = st.text_area("📝 Enter the news or headline to verify:")

# Add a stylish button for checking news
if st.button("🔍 Check"):
    if news_article:
        with st.spinner("Analyzing the article..."):
            result = manual_testing(news_article, model, vectorizer)
            if result == 0:
                st.markdown("<div class='result-box fake-result-box'>Fake ☹️</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box real-result-box'>Real 😃</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a news article or headline.")

# Footer
st.markdown("""
    <footer>
    <hr>
    <p>Developed with ❤️ for Fake News Detection</p>
    </footer>
""", unsafe_allow_html=True)
