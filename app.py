import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_objects():
    model = load_model('best_lstm_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tfidf, label_encoder

model, tfidf, label_encoder = load_objects()

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .neutral {
        background-color: #e2e3e5;
        border: 2px solid #d6d8db;
        color: #383d41;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48c78e);
        height: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 15px;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎭 Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover the emotional tone behind any text using advanced LSTM neural networks</p>', unsafe_allow_html=True)

st.markdown("### 💡 Try these examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("I love this product!", use_container_width=True):
        st.session_state.user_input = "I love this product! It's absolutely amazing and works perfectly."
with col2:
    if st.button("This is terrible", use_container_width=True):
        st.session_state.user_input = "This is terrible. I'm very disappointed with the poor quality and bad service."
with col3:
    if st.button("It's okay", use_container_width=True):
        st.session_state.user_input = "The product is okay. Nothing special but it gets the job done."

st.markdown("### ✍️ Enter your text below:")
user_input = st.text_area(
    "",
    value=st.session_state.get('user_input', ''),
    placeholder="Type your text here to analyze its sentiment...",
    height=150,
    label_visibility="collapsed"
)

if st.button("🔮 Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("🔍 Analyzing sentiment..."):
            x_input = tfidf.transform([user_input]).toarray()
            x_input = np.expand_dims(x_input, axis=1)
            y_pred_probs = model.predict(x_input)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            predicted_label = label_encoder.inverse_transform(y_pred_classes)[0]
            confidence = np.max(y_pred_probs) * 100

        sentiment_class = "positive" if "positive" in predicted_label.lower() else "negative" if "negative" in predicted_label.lower() else "neutral"
        
        st.markdown(f'<div class="prediction-box {sentiment_class}">', unsafe_allow_html=True)
        st.markdown(f"### {predicted_label.upper()}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.metric("Confidence Level", f"{confidence:.2f}%")
        
        st.markdown('<div class="confidence-bar"></div>', unsafe_allow_html=True)
        
        progress_color = "#48c78e" if confidence > 70 else "#feca57" if confidence > 50 else "#ff6b6b"
        st.markdown(f"""
        <div style="background: #f0f0f0; border-radius: 10px; padding: 5px; margin: 10px 0;">
            <div style="background: {progress_color}; width: {confidence}%; border-radius: 8px; padding: 8px; text-align: center; color: white; font-weight: bold;">
                {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        if sentiment_class == "positive":
            st.balloons()

st.markdown("---")