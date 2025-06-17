import streamlit as st
import base64
import os

# Set page config
st.set_page_config(
    page_title="Gradient Descent Blog",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Encode video as base64 with error handling
video_path = r"C:\Users\mjhus\Downloads\3129977-uhd_3840_2160_30fps.mp4"
video_base64 = ""
try:
    if os.path.exists(video_path):
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode()
        st.write("...")
    else:
        st.error(f"Video file {video_path} not found. Using fallback background.")
except Exception as e:
    st.error(f"Error loading video: {str(e)}. Using fallback background.")

# Inject CSS with increased spacing between paragraphs
st.markdown(
    f"""
    <style>
    .stApp {{
        position: relative;
        overflow: hidden;
        min-height: 100vh;
        background: url('https://via.placeholder.com/1920x1080?text=Fallback+Background') no-repeat center center fixed;
        background-size: cover;
    }}
    #bgvid {{
        position: fixed;
        top: 50%;
        left: 50%;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        transform: translate(-50%, -50%);
        z-index: -2;
        display: {'' if video_base64 else 'none'};
    }}
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.3);
        z-index: -1;
        pointer-events: none;
    }}
    .content {{
        position: relative;
        z-index: 10;
        color: #ffffff;
        text-align: center;
        padding-top: 10vh;
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }}
    .content * {{
        background: none !important;
        border: none !important;
        font-family: Arial, sans-serif !important;
    }}
    .content p {{
        margin: 0.5em 0 1.5em 0; /* Increased bottom margin for more space between paragraphs */
        display: block;
    }}
    .content h3 {{
        margin: 3em 0 0.5em 0; /* Adjusted to ensure consistent spacing */
        display: block;
    }}
    h1 {{
        font-size: 3em;
        margin-bottom: 0.3em;
    }}
    pre, code {{
        background: none !important;
        border: none !important;
        color: #ffffff !important;
        padding: 0 !important;
    }}
    </style>

    <video autoplay muted loop id="bgvid">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)

# Content with proper HTML rendering
st.markdown(
    """<div class="content">
<h1>📉 Welcome to the Ultimate Gradient Descent Guide</h1>
<p style="font-size: 1.3em;">Where optimization meets clarity — visualize, simulate, and master.<br>Let the gradients lead the way. 🧠✨</p>
<h3>📦 What You'll Learn</h3>
<p>✔️ What is Gradient Descent?<br>✔️ How learning rate affects convergence<br>✔️ Exploring cost functions and surfaces<br>✔️ Optimization variants like SGD, Momentum, Adam</p>
<h3>🎥 Visuals and Animations</h3>
<p>Step-by-step plots and simulations that help you "see" how learning unfolds.</p>
<h3>🧠 Ready to Begin?</h3>
<p>Explore the notes, tweak parameters, and deepen your understanding interactively!</p>
</div>""",
    unsafe_allow_html=True
)

# Footer
st.markdown(
    """<div style="text-align: center; margin-top: 4rem; color: #cccccc; position: relative; z-index: 10;">
Created with ❤️ by Mohammed Junaid | <a href="https://www.linkedin.com/in/mohammedjunaidd" style="color: #cccccc;" target="_blank">LinkedIn</a>
</div>""",
    unsafe_allow_html=True
)