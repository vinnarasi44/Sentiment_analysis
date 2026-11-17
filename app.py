import streamlit as st
import joblib
import re

model = joblib.load("model/sentiment_model.pkl")

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clear_text():
    st.session_state.review_text = ""

st.title("😊 AI-Based Sentiment Analysis Tool")

st.write("Analyze your review using Machine Learning.")

st.text_area(
    "Enter Your Review",
    key="review_text",
    height=150,
    placeholder="Type your review here..."
)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.button(
        "🗑️ Clear",
        on_click=clear_text,
        use_container_width=True
    )

if analyze:

    user_text = st.session_state.review_text

    if user_text.strip():

        cleaned_text = clean_text(user_text)

        prediction = model.predict([cleaned_text])

        probabilities = model.predict_proba([cleaned_text])[0]

        confidence = max(probabilities) * 100

        st.subheader("Sentiment Result")

        if prediction[0] == "Positive":
            st.success("😊 Positive Sentiment")

        elif prediction[0] == "Negative":
            st.error("😠 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")


st.markdown(
    """
    <div style="
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 18px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
"


<div style="
    background-color: #2563EB;
    color: #FFFFFF;
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;"
</div>
""", unsafe_allow_html=True)
