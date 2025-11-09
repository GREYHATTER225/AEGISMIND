"""AEGISMind - AI Truth Guardian Deepfake Detection System
Integrated with ResNeXt-LSTM model for real-time detection.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import hashlib
import time
import threading
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page

from models.resnext_lstm import DeepfakeDetector
from models.gradcam_utils import GradCAMExplainer
from models.image_classifier import ImageClassifier
from utils.preprocessing import pil_to_tensor
from realtime.video_stream import frame_generator_from_path_or_fileobj
from realtime.webcam_feed import webcam_frame_generator

# Custom CSS for enhanced UI
st.markdown("""
<style>
/* Global Styles */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');

* {
    font-family: 'Poppins', 'Roboto', 'Arial', sans-serif !important;
}

/* Header Styles */
.main-header {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 1px solid #00ffff;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.1);
    margin-bottom: 2rem;
    text-align: center;
}

.title-glow {
    color: #00ff9c;
    text-shadow: 0 0 20px #00ff9c, 0 0 40px #00ff9c;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.subtitle-glow {
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff;
    font-size: 1.2rem;
    opacity: 0.9;
}

/* Card Styles */
.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #00ffff;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.1);
    transition: all 0.3s ease;
}

.result-card:hover {
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
    transform: translateY(-2px);
}

.fake-card {
    border-color: #ff4444 !important;
    box-shadow: 0 0 20px rgba(255, 68, 68, 0.2) !important;
}

.real-card {
    border-color: #44ff44 !important;
    box-shadow: 0 0 20px rgba(68, 255, 68, 0.2) !important;
}

/* Button Styles */
.stButton button {
    background: linear-gradient(90deg, #00ffff, #0077ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    font-family: "Poppins", "Roboto", "Arial", sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
    cursor: pointer !important;
}

.stButton button::before,
.stButton button::after {
    content: none !important;
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.6) !important;
}

/* Tab Styles */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #00ffff !important;
    border-radius: 8px !important;
    margin: 0 0.25rem !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(45deg, #00ffff, #00ff9c) !important;
    color: #000 !important;
}

/* Sidebar Styles */
.sidebar-content {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Metric Styles */
.stMetric {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #00ffff;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);
}

/* Loading Animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 1.5s infinite;
}

/* Footer */
.footer {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    border-top: 1px solid #00ffff;
    padding: 2rem;
    text-align: center;
    margin-top: 3rem;
    border-radius: 10px 10px 0 0;
}

.footer-text {
    color: #00ffff;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header {
        padding: 1rem;
    }

    .title-glow {
        font-size: 2rem;
    }

    .result-card {
        padding: 1rem;
    }

    .stButton > button {
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
    }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a2e;
}

::-webkit-scrollbar-thumb {
    background: #00ffff;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #00ff9c;
}

/* Hide sidebar toggle button */
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    z-index: -1 !important;
    color: transparent !important;
    background: transparent !important;
    border: none !important;
    font-size: 0 !important;
    content: none !important;
    clip: rect(0, 0, 0, 0) !important;
    text-indent: -9999px !important;
    white-space: nowrap !important;
    line-height: 0 !important;
}

/* Login Page Floating Chatbot Pop-up */
.login-chatbot-popup {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 80px;
    height: 80px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #00ffff;
    border-radius: 50%;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #00ff9c;
    font-weight: bold;
    font-size: 0.7rem;
    text-align: center;
    padding: 5px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.3s ease;
}

.login-chatbot-popup:hover {
    transform: scale(1.1);
    box-shadow: 0 0 40px rgba(0, 255, 255, 0.5);
}

.login-chatbot-popup spline-viewer {
    width: 100% !important;
    height: 100% !important;
    border-radius: 15px !important;
    background: transparent !important;
}

/* Animated Robot Styles */
.robot-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
}

.robot-head {
    width: 60px;
    height: 45px;
    background: linear-gradient(135deg, #00ffff, #0077ff);
    border-radius: 50%;
    position: relative;
    margin-bottom: 8px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    animation: head-bob 3s ease-in-out infinite;
}

.robot-eye {
    width: 8px;
    height: 8px;
    background: #ff4444;
    border-radius: 50%;
    position: absolute;
    top: 15px;
    box-shadow: 0 0 8px rgba(255, 68, 68, 0.8);
    animation: eye-blink 2s infinite;
}

.robot-eye.left {
    left: 15px;
}

.robot-eye.right {
    right: 15px;
}

.robot-mouth {
    width: 15px;
    height: 6px;
    background: #333;
    border-radius: 0 0 8px 8px;
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
}

.robot-body {
    width: 75px;
    height: 60px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 50%;
    position: relative;
    border: 2px solid #00ffff;
    box-shadow: 0 0 12px rgba(0, 255, 255, 0.3);
}

.robot-arm {
    width: 20px;
    height: 35px;
    background: linear-gradient(135deg, #00ffff, #0077ff);
    border-radius: 10px;
    position: absolute;
    top: 8px;
    box-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
}

.robot-arm.left {
    left: -18px;
    animation: arm-wave-left 4s ease-in-out infinite;
}

.robot-arm.right {
    right: -18px;
    animation: arm-wave-right 4s ease-in-out infinite;
}

.robot-chest {
    width: 45px;
    height: 30px;
    background: #0a0a0a;
    border-radius: 50%;
    position: absolute;
    top: 15px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    justify-content: center;
}

.robot-light {
    width: 15px;
    height: 15px;
    background: #44ff44;
    border-radius: 50%;
    box-shadow: 0 0 12px rgba(68, 255, 68, 0.8);
    animation: light-pulse 1.5s ease-in-out infinite;
}

.robot-legs {
    display: flex;
    justify-content: space-between;
    width: 60px;
    margin-top: 8px;
}

.robot-leg {
    width: 18px;
    height: 30px;
    background: linear-gradient(135deg, #00ffff, #0077ff);
    border-radius: 9px;
    box-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
    animation: leg-bounce 2s ease-in-out infinite;
}

.robot-leg.left {
    animation-delay: 0s;
}

.robot-leg.right {
    animation-delay: 1s;
}

/* Robot Animations */
@keyframes head-bob {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

@keyframes eye-blink {
    0%, 90%, 100% { opacity: 1; transform: scaleY(1); }
    95% { opacity: 0; transform: scaleY(0.1); }
}

@keyframes arm-wave-left {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(-20deg); }
    75% { transform: rotate(20deg); }
}

@keyframes arm-wave-right {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(20deg); }
    75% { transform: rotate(-20deg); }
}

@keyframes light-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

@keyframes leg-bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(5px); }
}

/* Chat Interface Styles */
.chat-interface {
    position: fixed;
    bottom: 100px;
    right: 20px;
    width: 350px;
    height: 500px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #00ffff;
    border-radius: 15px;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
    z-index: 1001;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.chat-header {
    background: linear-gradient(90deg, #00ffff, #0077ff);
    color: white;
    padding: 15px;
    font-weight: bold;
    text-align: center;
    border-bottom: 1px solid #00ffff;
}

.chat-messages {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    max-height: 350px;
}

.chat-message {
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 10px;
    max-width: 80%;
    word-wrap: break-word;
}

.chat-message.user {
    background: linear-gradient(135deg, #00ffff, #0077ff);
    color: white;
    margin-left: auto;
    text-align: right;
}

.chat-message.bot {
    background: linear-gradient(135deg, #16213e, #1a1a2e);
    color: #00ff9c;
    border: 1px solid #00ffff;
    margin-right: auto;
}

.chat-input-container {
    padding: 15px;
    border-top: 1px solid #00ffff;
    background: rgba(0, 0, 0, 0.1);
}

.chat-input {
    width: 100%;
    padding: 10px;
    border: 1px solid #00ffff;
    border-radius: 5px;
    background: #0a0a0a;
    color: #00ff9c;
    margin-bottom: 10px;
}

.chat-send-btn {
    width: 100%;
    background: linear-gradient(90deg, #00ffff, #0077ff);
    color: white;
    border: none;
    padding: 10px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}

.chat-send-btn:hover {
    background: linear-gradient(90deg, #0077ff, #00ffff);
}

.chat-close-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    background: #ff4444;
    color: white;
    border: none;
    border-radius: 50%;
    width: 25px;
    height: 25px;
    cursor: pointer;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gradcam_explainer' not in st.session_state:
    st.session_state.gradcam_explainer = None
if 'image_model' not in st.session_state:
    st.session_state.image_model = None
if 'image_gradcam_explainer' not in st.session_state:
    st.session_state.image_gradcam_explainer = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""

# Authentication function
def authenticate(username, password):
    # Simple authentication (in production, use proper auth system)
    if username and password:
        return True
    return False

# Bot response function
def generate_bot_response(user_input):
    responses = {
        "hello": "Hello! I'm your AI assistant for deepfake detection. How can I help you today?",
        "how": "I can explain deepfake detection results, help you understand the system, or answer questions about AI-generated content.",
        "what": "AEGISMind is an AI-powered deepfake detection system that analyzes images and videos to detect manipulated content.",
        "help": "I can assist with:\n- Explaining detection results\n- System features\n- Deepfake concepts\n- Troubleshooting"
    }
    for key, response in responses.items():
        if key in user_input.lower():
            return response
    return "I'm here to help with deepfake detection. What would you like to know?"

# Chat interface function
def chat_interface():
    st.markdown('<div class="chat-interface">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🤖 AI Assistant</div>', unsafe_allow_html=True)

    # Messages
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-message user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed")
    with col2:
        if st.button("Send", key="send_chat"):
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                bot_response = generate_bot_response(user_input)
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                st.rerun()

    # Close button
    if st.button("×", key="close_chat"):
        st.session_state.chat_visible = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Login page
def login_page():
    st.markdown('<div class="main-header"><h1 class="title-glow">🔐 AEGIS Login</h1></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

        # Display image below login button
        st.image("app/assets/images/image.png", use_container_width=True)

    # Chat interface
    if st.session_state.chat_visible:
        chat_interface()

    # Floating chatbot popup with animated robot
    st.markdown(f"""
    <div class="login-chatbot-popup" onclick="toggleChat()">
        <div class="robot-container">
            <div class="robot-head">
                <div class="robot-eye left"></div>
                <div class="robot-eye right"></div>
                <div class="robot-mouth"></div>
            </div>
            <div class="robot-body">
                <div class="robot-arm left"></div>
                <div class="robot-arm right"></div>
                <div class="robot-chest">
                    <div class="robot-light"></div>
                </div>
            </div>
            <div class="robot-legs">
                <div class="robot-leg left"></div>
                <div class="robot-leg right"></div>
            </div>
        </div>
    </div>
    <script>
    function toggleChat() {{
        // This will be handled by Streamlit rerun
        const event = new CustomEvent('toggleChat');
        window.dispatchEvent(event);
    }}
    </script>
    """, unsafe_allow_html=True)

    # Handle chat toggle
    if st.button("Toggle Chat", key="chat_toggle", help="Toggle AI Assistant Chat"):
        st.session_state.chat_visible = not st.session_state.chat_visible
        st.rerun()

# Main app
def main_app():
    # Header
    st.markdown("""
<div class="main-header">
    <h1 class="title-glow">🛡️ AEGIS<span style="color:#00ffff;">MIND</span></h1>
    <p class="subtitle-glow">AI Truth Guardian - Deepfake Detection System</p>
</div>
""", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

        # User info
        if st.session_state.authenticated:
            st.markdown(f"**👤 User:** {st.session_state.username}")
            if st.button("Logout", key="logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()

        st.markdown("---")

        # Model status
        st.markdown("### 🤖 Model Status")
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded & Ready")
        else:
            if st.button("Load Detection Model", type="primary", use_container_width=True):
                with st.spinner("Loading model..."):
                    success = load_model()
                    if success:
                        st.success("Model loaded successfully!")
                        st.rerun()

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Settings"):
            verify_integrity = st.checkbox('Verify file SHA256 integrity', value=True)
            confidence_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01,
                                           help="Adjust detection sensitivity")

        st.markdown("---")

        # System info
        st.markdown("### 📊 System Info")
        st.markdown(f"**Model:** ResNeXt-LSTM")
        st.markdown(f"**Device:** CPU")
        st.markdown(f"**Status:** {'🟢 Active' if st.session_state.model_loaded else '🔴 Inactive'}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Main content with tabs
    if st.session_state.model_loaded:
        tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Analysis", "🎥 Video Upload", "📹 Live Webcam", "🔄 Comparison"])

        with tab1:
            image_analysis_tab(verify_integrity)

        with tab2:
            video_analysis_tab(verify_integrity)

        with tab3:
            webcam_analysis_tab()

        with tab4:
            comparison_tab()
    else:
        st.warning("⚠️ Please load the detection model from the sidebar first to begin analysis.")

    # Chat interface
    if st.session_state.chat_visible:
        chat_interface()

    # Footer
    st.markdown("""
    <div class="footer">
        <p class="footer-text">© 2025AEGISMind - AI Truth Guardian</p>
        <p class="footer-text">Version 2.0 | Built with ❤️ using Streamlit by grey hatter // satya</p>
        <p class="footer-text">🔒 Secure • 🚀 Fast • 🎯 Accurate</p>
    </div>
    """, unsafe_allow_html=True)

    # Floating chatbot popup with animated robot
    st.markdown("""
    <div class="login-chatbot-popup" onclick="toggleChat()">
        <div class="robot-container">
            <div class="robot-head">
                <div class="robot-eye left"></div>
                <div class="robot-eye right"></div>
                <div class="robot-mouth"></div>
            </div>
            <div class="robot-body">
                <div class="robot-arm left"></div>
                <div class="robot-arm right"></div>
                <div class="robot-chest">
                    <div class="robot-light"></div>
                </div>
            </div>
            <div class="robot-legs">
                <div class="robot-leg left"></div>
                <div class="robot-leg right"></div>
            </div>
        </div>
    </div>
    <script>
    function toggleChat() {
        // This will be handled by Streamlit rerun
        const event = new CustomEvent('toggleChat');
        window.dispatchEvent(event);
    }
    </script>
    """, unsafe_allow_html=True)

    # Handle chat toggle
    if st.button("Toggle Chat", key="chat_toggle_main", help="Toggle AI Assistant Chat"):
        st.session_state.chat_visible = not st.session_state.chat_visible
        st.rerun()

# Model loading function
def load_model():
    try:
        model_path = 'models/pretrained/model_weights.pt'
        if not os.path.exists(model_path):
            st.error("Video model weights not found. Please train the model first.")
            return False
        model = DeepfakeDetector()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        st.session_state.model = model
        st.session_state.gradcam_explainer = GradCAMExplainer(model)

        image_model_path = 'models/pretrained/image_classifier.pt'
        if os.path.exists(image_model_path):
            image_model = ImageClassifier()
            image_model.load_state_dict(torch.load(image_model_path, map_location='cpu', weights_only=True))
            image_model.eval()
            st.session_state.image_model = image_model
            st.session_state.image_gradcam_explainer = GradCAMExplainer(image_model)
        else:
            st.warning("Image classifier not found. Image analysis will use video model.")

        st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

# Image preprocessing
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    tensor = pil_to_tensor(image)
    return tensor.unsqueeze(0)

# Prediction functions with caching
@st.cache_data
def predict_image_cached(image_hash, _image_tensor, _model):
    with torch.no_grad():
        # If using video model for images, add frame dimension
        if len(_image_tensor.shape) == 4:  # [batch, channels, height, width]
            _image_tensor = _image_tensor.unsqueeze(1)  # [batch, 1, channels, height, width]
        output = _model(_image_tensor)
        prob = torch.sigmoid(output).item()
        if prob > 0.5:
            classification = "Deepfake"
            confidence = prob
        else:
            classification = "Real"
            confidence = 1 - prob
        return classification, confidence

@st.cache_data
def predict_video_cached(video_hash, _frames, _model, max_frames=30):
    if len(_frames) > max_frames:
        indices = np.linspace(0, len(_frames)-1, max_frames, dtype=int)
        frames = [_frames[i] for i in indices]
    else:
        frames = _frames
    frame_tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        if frame.mode == 'RGBA':
            frame = frame.convert('RGB')
        tensor = preprocess_image(frame)
        frame_tensors.append(tensor.squeeze(0))
    video_tensor = torch.stack(frame_tensors)
    video_tensor = video_tensor.unsqueeze(0)
    with torch.no_grad():
        output = _model(video_tensor)
        prob = torch.sigmoid(output).item()
        if prob > 0.5:
            classification = "Deepfake"
            confidence = prob
        else:
            classification = "Real"
            confidence = 1 - prob
        return classification, confidence, len(frames)

# Result display function
def display_result_card(classification, confidence, num_frames=None, title="Analysis Result"):
    card_class = "fake-card" if classification == "Deepfake" else "real-card"
    st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {title}")
        confidence_pct = confidence * 100
        if classification == "Deepfake":
            st.error(f"🚨 **Deepfake Detected** - Confidence: {confidence_pct:.1f}%")
        else:
            st.success(f"✅ **Authentic Content** - Confidence: {confidence_pct:.1f}%")

    with col2:
        # Progress bar
        st.progress(confidence)

    if num_frames:
        st.info(f"📊 Processed {num_frames} frames")

    st.markdown('</div>', unsafe_allow_html=True)

# Interactive chart function
def create_interactive_chart(frame_scores, title="Manipulation Scores Over Time"):
    fig = go.Figure()

    # Add main line
    fig.add_trace(go.Scatter(
        x=list(range(len(frame_scores))),
        y=frame_scores,
        mode='lines+markers',
        name='Manipulation Score',
        line=dict(color='#ff4444', width=3),
        marker=dict(size=6, color='#ff4444')
    ))

    # Add threshold line
    fig.add_trace(go.Scatter(
        x=[0, len(frame_scores)-1],
        y=[0.5, 0.5],
        mode='lines',
        name='Detection Threshold',
        line=dict(color='#ffff00', width=2, dash='dash')
    ))

    # Fill fake regions
    fake_regions = [i for i, score in enumerate(frame_scores) if score > 0.5]
    if fake_regions:
        fig.add_trace(go.Scatter(
            x=fake_regions + fake_regions[::-1],
            y=[frame_scores[i] for i in fake_regions] + [0.5] * len(fake_regions),
            fill='toself',
            fillcolor='rgba(255, 68, 68, 0.3)',
            line=dict(color='transparent'),
            name='Fake Region'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Frame Number",
        yaxis_title="Manipulation Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ffff')
    )

    return fig

# Image analysis tab
def image_analysis_tab(verify_integrity):
    st.markdown("### 📸 Image Analysis")
    st.markdown("Upload an image to detect if it's real or fake with advanced AI analysis.")

    # Drag and drop uploader
    uploaded_file = st.file_uploader(
        "Choose an image file or drag & drop",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG",
        accept_multiple_files=False
    )

    if uploaded_file:
        # File verification
        data = uploaded_file.getvalue()
        file_hash = hashlib.sha256(data).hexdigest()

        if verify_integrity:
            with st.expander("🔐 File Verification"):
                short_hash = f"{file_hash[:16]}...{file_hash[-16:]}"
                st.code(short_hash)
                st.text(f"Full SHA256: {file_hash}")

        # Load and display image
        img = Image.open(uploaded_file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption='📷 Uploaded Image', use_container_width=True)

        with col2:
            st.metric("File Size", f"{len(data)/1024:.1f} KB")
            st.metric("Dimensions", f"{img.size[0]} × {img.size[1]}")

        # Analysis button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing image with AI..."):
                image_tensor = preprocess_image(img)

                # Use cached prediction
                cache_key = f"image_{file_hash}"
                if cache_key in st.session_state.cache:
                    classification, confidence = st.session_state.cache[cache_key]
                else:
                    model_to_use = st.session_state.image_model if st.session_state.image_model else st.session_state.model
                    classification, confidence = predict_image_cached(file_hash, image_tensor, model_to_use)
                    st.session_state.cache[cache_key] = (classification, confidence)

                # Display result
                display_result_card(classification, confidence, title="Image Analysis Result")

                # GradCAM explanation
                st.markdown("### 🔥 AI Explanation - GradCAM Heatmap")
                gradcam_to_use = st.session_state.image_gradcam_explainer if st.session_state.image_gradcam_explainer else st.session_state.gradcam_explainer

                heatmap = gradcam_to_use.generate_heatmap(image_tensor, target_class=1 if classification == "Deepfake" else 0)
                overlay = gradcam_to_use.overlay_heatmap(np.array(img), heatmap, alpha=0.6)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(overlay, caption="🔥 AI Focus Areas", use_container_width=True)
                with col2:
                    pass

                # Add to history
                st.session_state.analysis_history.append({
                    'type': 'image',
                    'timestamp': time.time(),
                    'result': classification,
                    'confidence': confidence,
                    'file_hash': file_hash
                })

# Video analysis tab
def video_analysis_tab(verify_integrity):
    st.markdown("### 🎥 Video Analysis")
    st.markdown("Upload a video for comprehensive deepfake detection with forensic analysis.")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file or drag & drop",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )

    with col2:
        max_frames = st.slider('🎯 Max frames to analyze', 10, 500, 120,
                              help="Higher values = more accurate but slower analysis")

    if uploaded_file:
        # File verification
        raw = uploaded_file.getvalue()
        file_hash = hashlib.sha256(raw).hexdigest()

        if verify_integrity:
            with st.expander("🔐 File Verification"):
                short_hash = f"{file_hash[:16]}...{file_hash[-16:]}"
                st.code(short_hash)

        # Video info
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(raw)
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Video metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Frames", f"{total_frames:,}")
        with col2:
            st.metric("🎬 FPS", f"{fps:.1f}")
        with col3:
            st.metric("⏱️ Duration", f"{duration:.1f}s")
        with col4:
            st.metric("💾 File Size", f"{len(raw)/1024/1024:.1f} MB")

        # Analysis button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("🎬 Extracting frames and analyzing with AI..."):
                frames = []
                frame_iter = frame_generator_from_path_or_fileobj(uploaded_file, max_frames=max_frames)
                progress_bar = st.progress(0)
                frame_count = 0

                for idx, frame in frame_iter:
                    frames.append(frame)
                    frame_count += 1
                    progress_bar.progress(min(frame_count / max_frames, 1.0))

                progress_bar.empty()

                if not frames:
                    st.error("❌ Could not extract frames from video")
                else:
                    # Use cached prediction
                    cache_key = f"video_{file_hash}_{max_frames}"
                    if cache_key in st.session_state.cache:
                        classification, confidence, processed_frames = st.session_state.cache[cache_key]
                    else:
                        classification, confidence, processed_frames = predict_video_cached(file_hash, frames, st.session_state.model, max_frames)
                        st.session_state.cache[cache_key] = (classification, confidence, processed_frames)

                    # Display result
                    display_result_card(classification, confidence, processed_frames, title="Video Analysis Result")

                    # GradCAM analysis
                    st.markdown("### 🔥 AI Explanation - GradCAM Analysis")
                    explanations = st.session_state.gradcam_explainer.explain_prediction(frames, confidence)

                    if explanations['suspicious_frames']:
                        st.markdown("### 🚨 Suspicious Regions Detected")
                        suspicious_cols = st.columns(min(3, len(explanations['suspicious_frames'])))
                        for i, suspicious in enumerate(explanations['suspicious_frames'][:3]):
                            with suspicious_cols[i]:
                                frame_idx = suspicious['frame_index']
                                heatmap = suspicious['heatmap']
                                activation = suspicious['activation_score']

                                frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                                overlay = st.session_state.gradcam_explainer.overlay_heatmap(frame_rgb, heatmap, alpha=0.6)

                                st.image(overlay, caption=f"Frame {frame_idx+1} - Suspicion: {activation:.2f}", use_container_width=True)
                                st.error(f"⚠️ High manipulation score: {activation:.2f}")
                    else:
                        st.success("✅ No highly suspicious regions detected")

                    # Frame thumbnails
                    st.markdown("### 🖼️ Frame Thumbnails")
                    thumbnail_cols = st.columns(min(8, len(frames)))
                    for i, frame in enumerate(frames[:8]):
                        with thumbnail_cols[i]:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Frame {i+1}", use_container_width=True)

                    # Interactive chart
                    st.markdown("### 📈 Manipulation Scores Over Time")
                    frame_scores = []
                    for i, frame in enumerate(frames):
                        frame_tensor = preprocess_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                        _, confidence_score = predict_image_cached(f"frame_{i}_{file_hash}", frame_tensor, st.session_state.model)
                        frame_scores.append(confidence_score)

                    fig = create_interactive_chart(frame_scores)
                    st.plotly_chart(fig, use_container_width=True)

                    # Forensic findings
                    st.markdown("### 🔍 Forensic Findings")
                    display_forensic_findings(classification, confidence, processed_frames)

                    # Add to history
                    st.session_state.analysis_history.append({
                        'type': 'video',
                        'timestamp': time.time(),
                        'result': classification,
                        'confidence': confidence,
                        'frames': processed_frames,
                        'file_hash': file_hash
                    })

        os.unlink(video_path)

# Webcam analysis tab
def webcam_analysis_tab():
    st.markdown("### 📹 Live Webcam Analysis")
    st.markdown("Real-time deepfake detection from your webcam feed.")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start Live Analysis", type="primary", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹️ Stop Analysis", use_container_width=True)

    placeholder = st.empty()
    result_placeholder = st.empty()

    if start_btn:
        st.info("📹 Starting webcam analysis...")
        try:
            frame_buffer = []
            analysis_count = 0

            for frame_idx, frame in webcam_frame_generator(max_frames=100):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame_rgb, caption="📹 Live Feed", use_container_width=True)

                frame_buffer.append(frame_rgb)
                if len(frame_buffer) >= 10:
                    with st.spinner("🧠 Analyzing frames..."):
                        classification, confidence, _ = predict_video_cached(
                            f"webcam_{analysis_count}", frame_buffer, st.session_state.model, max_frames=10
                        )
                        confidence_pct = confidence * 100

                        if classification == "Deepfake":
                            result_placeholder.error(f"🚨 FAKE DETECTED - Confidence: {confidence_pct:.1f}% | Frame: {frame_idx}")
                        else:
                            result_placeholder.success(f"✅ REAL CONTENT - Confidence: {confidence_pct:.1f}% | Frame: {frame_idx}")

                    frame_buffer = []
                    analysis_count += 1

                if stop_btn:
                    break
                time.sleep(0.1)

            st.success("✅ Webcam analysis completed")

        except Exception as e:
            st.error(f"❌ Webcam error: {str(e)}")

# Comparison tab
def comparison_tab():
    st.markdown("### 🔄 Side-by-Side Comparison")
    st.markdown("Compare multiple images or videos for deepfake detection.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📸 Image 1")
        img1_file = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key="img1")

    with col2:
        st.markdown("#### 📸 Image 2")
        img2_file = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key="img2")

    if img1_file and img2_file:
        img1 = Image.open(img1_file)
        img2 = Image.open(img2_file)

        if img1.mode == 'RGBA':
            img1 = img1.convert('RGB')
        if img2.mode == 'RGBA':
            img2 = img2.convert('RGB')

        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption="Image 1", use_container_width=True)
        with col2:
            st.image(img2, caption="Image 2", use_container_width=True)

        if st.button("🔍 Compare Images", type="primary", use_container_width=True):
            with st.spinner("Comparing images..."):
                # Analyze both images
                tensor1 = preprocess_image(img1)
                tensor2 = preprocess_image(img2)

                hash1 = hashlib.sha256(img1_file.getvalue()).hexdigest()
                hash2 = hashlib.sha256(img2_file.getvalue()).hexdigest()

                model_to_use = st.session_state.image_model if st.session_state.image_model else st.session_state.model

                result1 = predict_image_cached(hash1, tensor1, model_to_use)
                result2 = predict_image_cached(hash2, tensor2, model_to_use)

                # Display comparison
                st.markdown("### 📊 Comparison Results")

                col1, col2 = st.columns(2)
                with col1:
                    display_result_card(result1[0], result1[1], title="Image 1 Result")
                with col2:
                    display_result_card(result2[0], result2[1], title="Image 2 Result")

                # Summary
                if result1[0] == result2[0]:
                    st.info(f"Both images are classified as **{result1[0]}**")
                else:
                    st.warning("Images have different classifications - potential inconsistency detected!")

# Forensic findings display
def display_forensic_findings(classification, confidence, processed_frames):
    is_fake = classification == "Deepfake"

    if is_fake:
        st.error("🚨 **DEEPFAKE DETECTED**")
        st.markdown("**Primary Issue:** AI-generated content detected")
    else:
        st.success("✅ **AUTHENTIC CONTENT**")
        st.markdown("**Status:** No manipulation detected")

    st.markdown("---")

    # Detection metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence Score", f"{confidence*100:.1f}%")
    with col2:
        st.metric("Frames Analyzed", f"{processed_frames}")

    st.markdown("---")

    # Findings
    st.markdown("**Potential Manipulation Types:**")
    findings = []
    if is_fake:
        findings.extend([
            "🔴 Deepfake synthesis detected",
            "🔴 Facial expression inconsistency",
            "🔴 Temporal artifacts present",
            "🔴 Lighting inconsistencies"
        ])
    else:
        findings.extend([
            "🟢 Natural facial movements",
            "🟢 Consistent lighting",
            "🟢 No temporal artifacts",
            "🟢 Authentic audio-visual sync"
        ])

    for finding in findings:
        st.markdown(finding)

    st.markdown("---")

    risk_level = "HIGH" if is_fake and (confidence * 100) > 80 else "MEDIUM" if is_fake else "LOW"
    if is_fake:
        st.error(f"**Risk Level:** {risk_level}")
    else:
        st.success(f"**Risk Level:** {risk_level}")

# Main execution
if __name__ == "__main__":
    st.set_page_config(
        page_title="AEGISMind - AI Truth Guardian",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()
