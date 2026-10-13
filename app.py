import os

import tempfile



import cv2

import numpy as np

import streamlit as st

import torch

import torch.nn as nn



from PIL import Image

from torchvision import models, transforms



from auth import authenticate_user, register_user





# ============================================================

# PAGE CONFIG

# ============================================================



st.set_page_config(

    page_title="DeepGuard | AI Media Security",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="collapsed",

)





# ============================================================

# DEVICE / SESSION

# ============================================================



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if "logged_in" not in st.session_state:

    st.session_state.logged_in = False



if "username" not in st.session_state:

    st.session_state.username = ""



if "page" not in st.session_state:

    st.session_state.page = "Dashboard"





# ============================================================

# GLOBAL STYLE

# ============================================================



st.html("""

  <style>

    /* Google Fonts */

    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');



    /* Global Dark Theme */

    .stApp {

        background-color: #070b12;

        background-image:&#x20;

            radial-gradient(circle at 15% 15%, rgba(0, 229, 255, 0.05) 0%, transparent 40%),

            radial-gradient(circle at 85% 85%, rgba(121, 40, 202, 0.06) 0%, transparent 40%);

        color: #e2e8f0;

        font-family: 'Inter', sans-serif;

    }



    /* Remove excess Streamlit margins while keeping controls accessible */

    .block-container {

        padding-top: 2rem !important;

        padding-bottom: 2rem !important;

        max-width: 1400px !important;

    }



    /* Glowing Glassmorphism Card */

    .glass-card {

        background: rgba(13, 21, 38, 0.75);

        backdrop-filter: blur(14px);

        -webkit-backdrop-filter: blur(14px);

        border: 1px solid rgba(0, 229, 255, 0.18);

        border-radius: 14px;

        padding: 24px;

        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);

        transition: all 0.3s ease;

    }

    .glass-card:hover {

        border-color: rgba(0, 229, 255, 0.35);

        box-shadow: 0 10px 36px 0 rgba(0, 229, 255, 0.12);

    }



    /* Hero Brand Text */

    .brand-title {

        font-family: 'Orbitron', sans-serif;

        font-size: 2.2rem;

        font-weight: 900;

        letter-spacing: 2px;

        background: linear-gradient(135deg, #00e5ff 0%, #7928ca 100%);

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        margin-bottom: 8px;

    }

    .tagline {

        font-family: 'Rajdhani', sans-serif;

        font-size: 1.5rem;

        font-weight: 700;

        letter-spacing: 3px;

        color: #f8fafc;

        margin-bottom: 12px;

    }

    .subtagline {

        color: #94a3b8;

        font-size: 0.95rem;

        line-height: 1.6;

    }



    /* Glowing Status Badges */

    .badge-fake {

        display: inline-block;

        padding: 8px 24px;

        font-family: 'Orbitron', sans-serif;

        font-size: 1.3rem;

        font-weight: 800;

        color: #ff3366;

        background: rgba(255, 51, 102, 0.12);

        border: 2px solid #ff3366;

        border-radius: 8px;

        box-shadow: 0 0 18px rgba(255, 51, 102, 0.4);

        letter-spacing: 2px;

    }

    .badge-real {

        display: inline-block;

        padding: 8px 24px;

        font-family: 'Orbitron', sans-serif;

        font-size: 1.3rem;

        font-weight: 800;

        color: #00ff88;

        background: rgba(0, 255, 136, 0.12);

        border: 2px solid #00ff88;

        border-radius: 8px;

        box-shadow: 0 0 18px rgba(0, 255, 136, 0.4);

        letter-spacing: 2px;

    }

    .badge-uncertain {

        display: inline-block;

        padding: 8px 24px;

        font-family: 'Orbitron', sans-serif;

        font-size: 1.3rem;

        font-weight: 800;

        color: #ffb800;

        background: rgba(255, 184, 0, 0.12);

        border: 2px solid #ffb800;

        border-radius: 8px;

        box-shadow: 0 0 18px rgba(255, 184, 0, 0.4);

        letter-spacing: 2px;

    }



    /* Metric Cards */

    .metric-box {

        background: rgba(15, 23, 42, 0.6);

        border: 1px solid rgba(255, 255, 255, 0.08);

        border-radius: 10px;

        padding: 14px;

        text-align: center;

        margin-bottom: 10px;

    }

    .metric-value {

        font-family: 'Orbitron', sans-serif;

        font-size: 1.4rem;

        font-weight: 700;

        color: #00e5ff;

    }

    .metric-label {

        font-size: 0.78rem;

        text-transform: uppercase;

        letter-spacing: 1.2px;

        color: #94a3b8;

    }



    /* Shared dashboard/card blocks */

    .dark-card {

        background: rgba(12, 17, 29, 0.78);

        border: 1px solid rgba(148, 163, 184, 0.12);

        border-radius: 16px;

        padding: 22px 20px;

        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.22);

        backdrop-filter: blur(10px);

        -webkit-backdrop-filter: blur(10px);

    }

    .card-icon {

        font-size: 2rem;

        color: #00e5ff;

        margin-bottom: 10px;

        display: inline-block;

    }

    .card-title {

        font-family: 'Orbitron', sans-serif;

        font-size: 1rem;

        font-weight: 700;

        letter-spacing: 0.12em;

        text-transform: uppercase;

        color: #f8fafc;

        margin-bottom: 8px;

    }

    .card-description {

        color: #a9b5c9;

        font-size: 0.95rem;

        line-height: 1.65;

    }



    .eyebrow {

        font-family: 'Orbitron', sans-serif;

        font-size: 0.7rem;

        font-weight: 700;

        letter-spacing: 0.22em;

        text-transform: uppercase;

        color: #7dd3fc;

        margin-bottom: 10px;

    }

    .page-title {

        font-family: 'Orbitron', sans-serif;

        font-size: 2.5rem;

        line-height: 1.06;

        font-weight: 800;

        letter-spacing: 0.04em;

        color: #f8fafc;

        margin-bottom: 12px;

    }

    .page-title span {

        color: #00e5ff;

    }

    .page-description {

        color: #a8b0c0;

        font-size: 1rem;

        line-height: 1.7;

        max-width: 720px;

    }



    /* Navbar / header */

    .navbar {

        display: flex;

        justify-content: space-between;

        align-items: center;

        padding: 12px 24px;

        background: rgba(13, 21, 38, 0.85);

        backdrop-filter: blur(12px);

        border: 1px solid rgba(0, 229, 255, 0.2);

        border-radius: 12px;

        margin-bottom: 24px;

    }

    .brand {

        display: flex;

        align-items: center;

        gap: 12px;

    }

    .brand-icon {

        width: 38px;

        height: 38px;

        border-radius: 12px;

        display: grid;

        place-items: center;

        font-size: 1.25rem;

        background: linear-gradient(135deg, rgba(0, 229, 255, 0.22), rgba(121, 40, 202, 0.22));

        border: 1px solid rgba(0, 229, 255, 0.28);

    }

    .brand-name {

        font-family: 'Orbitron', sans-serif;

        font-size: 1.05rem;

        font-weight: 800;

        letter-spacing: 0.12em;

        text-transform: uppercase;

        color: #f8fafc;

    }

    .brand-sub {

        font-size: 0.7rem;

        letter-spacing: 0.14em;

        text-transform: uppercase;

        color: #93a3be;

    }



    /* Result cards */

    .result {

        background: rgba(12, 18, 30, 0.8);

        border: 1px solid rgba(255, 255, 255, 0.08);

        border-radius: 14px;

        padding: 18px 20px;

        min-height: 140px;

        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);

    }

    .result-icon {

        font-size: 2.2rem;

        margin-bottom: 10px;

    }

    .result-label {

        font-family: 'Orbitron', sans-serif;

        font-size: 1.5rem;

        font-weight: 800;

        letter-spacing: 0.08em;

        text-transform: uppercase;

        color: #f8fafc;

        margin-bottom: 6px;

    }

    .result-confidence {

        color: #9fb2d1;

        font-size: 0.8rem;

        letter-spacing: 0.08em;

        text-transform: uppercase;

    }



    /* Background animation */

    .morph-container {

        position: fixed;

        inset: 0;

        pointer-events: none;

        overflow: hidden;

        z-index: 0;

    }

    .morph {

        position: absolute;

        border-radius: 50%;

        filter: blur(28px);

        opacity: 0.2;

        animation: floatPulse 18s ease-in-out infinite;

    }

    .morph.one {

        width: 220px;

        height: 220px;

        background: rgba(0, 229, 255, 0.32);

        top: 12%;

        left: 8%;

    }

    .morph.two {

        width: 260px;

        height: 260px;

        background: rgba(121, 40, 202, 0.28);

        bottom: 16%;

        right: 12%;

        animation-delay: 4s;

    }

    .morph.three {

        width: 200px;

        height: 200px;

        background: rgba(0, 255, 136, 0.16);

        top: 40%;

        right: 28%;

        animation-delay: 7s;

    }

    @keyframes floatPulse {

        0%, 100% { transform: translate3d(0, 0, 0) scale(1); }

        50% { transform: translate3d(12px, -18px, 0) scale(1.08); }

    }

    @keyframes pageIn {

        from { opacity: 0; transform: translateY(5px); }

        to { opacity: 1; transform: translateY(0); }

    }



    /* Auth layout */

    .auth-brand-card {

        position: relative;

        overflow: hidden;

        min-height: 560px;

        background: linear-gradient(135deg, rgba(12, 17, 29, 0.9), rgba(18, 25, 42, 0.8));

        border: 1px solid rgba(0, 229, 255, 0.15);

        border-radius: 18px;

        padding: 28px;

    }

    .auth-glow {

        position: absolute;

        inset: 0;

        background: radial-gradient(circle at 20% 20%, rgba(0, 229, 255, 0.12), transparent 35%);

    }

    .auth-label {

        position: relative;

        z-index: 1;

        font-family: 'Orbitron', sans-serif;

        font-size: 0.7rem;

        letter-spacing: 0.24em;

        color: #7dd3fc;

        margin-bottom: 24px;

    }

    .auth-brand-center {

        position: relative;

        z-index: 1;

        display: flex;

        flex-direction: column;

        justify-content: center;

        min-height: 360px;

    }

    .auth-title {

        font-family: 'Orbitron', sans-serif;

        font-size: clamp(2.4rem, 4vw, 4.2rem);

        font-weight: 800;

        letter-spacing: 0.08em;

        line-height: 0.95;

        color: #f8fafc;

        margin-bottom: 18px;

    }

    .auth-title span {

        color: #00e5ff;

    }

    .auth-description {

        max-width: 420px;

        color: #c5d0e4;

        font-size: 1rem;

        line-height: 1.7;

    }

    .auth-footer {

        position: relative;

        z-index: 1;

        display: flex;

        align-items: center;

        gap: 10px;

        margin-top: 24px;

        color: #8eb2d8;

        font-family: 'Orbitron', sans-serif;

        font-size: 0.7rem;

        letter-spacing: 0.18em;

        text-transform: uppercase;

    }

    .auth-footer i {

        width: 6px;

        height: 6px;

        border-radius: 50%;

        background: rgba(0, 229, 255, 0.8);

        display: inline-block;

    }

    .auth-form-head {

        margin-bottom: 18px;

    }

    .auth-form-kicker {

        font-family: 'Orbitron', sans-serif;

        font-size: 0.68rem;

        letter-spacing: 0.2em;

        color: #7dd3fc;

        margin-bottom: 8px;

    }

    .auth-form-title {

        font-family: 'Orbitron', sans-serif;

        font-size: 1.8rem;

        font-weight: 700;

        color: #f8fafc;

        margin-bottom: 8px;

    }

    .auth-form-subtitle {

        color: #a9b8cd;

        font-size: 0.94rem;

        line-height: 1.6;

    }

    .auth-hint {

        font-size: 0.72rem;

        letter-spacing: 0.08em;

        text-transform: uppercase;

        color: #8ca2be;

        margin-top: 1rem;

        padding-top: 0.8rem;

        border-top: 1px solid rgba(148, 163, 184, 0.12);

    }

    .auth-bottom-note {

        margin-top: 18px;

        text-align: center;

        font-size: 0.7rem;

        letter-spacing: 0.22em;

        text-transform: uppercase;

        color: #5f6879;

    }



    /* Button overrides */

    div.stButton > button:first-child {

        background: linear-gradient(135deg, rgba(0, 229, 255, 0.2) 0%, rgba(121, 40, 202, 0.25) 100%);

        color: #00e5ff;

        border: 1px solid #00e5ff;

        border-radius: 8px;

        padding: 8px 24px;

        font-family: 'Rajdhani', sans-serif;

        font-weight: 700;

        font-size: 1.05rem;

        letter-spacing: 1.5px;

        transition: all 0.25s ease;

        box-shadow: 0 0 12px rgba(0, 229, 255, 0.15);

    }

    div.stButton > button:first-child:hover {

        background: linear-gradient(135deg, #00e5ff 0%, #7928ca 100%);

        color: #ffffff;

        border-color: #00e5ff;

        box-shadow: 0 0 20px rgba(0, 229, 255, 0.45);

        transform: translateY(-1px);

    }



    /* Streamlit Input Styling */

    div[data-baseweb="input"] {

        background-color: rgba(10, 16, 29, 0.8) !important;

        border-radius: 8px !important;

        border: 1px solid rgba(0, 229, 255, 0.25) !important;

        color: #f1f5f9 !important;

    }

    div[data-baseweb="input"]:focus-within {

        border-color: #00e5ff !important;

        box-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;

    }



    /* Progress bar customization */

    .stProgress > div > div > div > div {

        background-image: linear-gradient(to right, #00e5ff, #7928ca);

    }

  </style>

""")



# Background

st.html("""

<div class="morph-container">

    <div class="morph one"></div>

    <div class="morph two"></div>

    <div class="morph three"></div>

</div>

""")





# ============================================================

# MODEL

# ============================================================



@st.cache_resource

def load_model():

    model = models.resnet18(weights=None)

    features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Dropout(0.4),

        nn.Linear(features, 2),

    )



    model.load_state_dict(

        torch.load(

            "models/best_resnet18_deepfake.pth",

            map_location=DEVICE,

        )

    )

    model.to(DEVICE)

    model.eval()

    return model





model = load_model()



transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    ),

])



cascade_path = os.path.join(

    ".venv",

    "Lib",

    "site-packages",

    "cv2",

    "data",

    "haarcascade_frontalface_default.xml",

)



face_detector = cv2.CascadeClassifier(cascade_path)





# ============================================================

# HELPERS

# ============================================================



def show_navbar():

    st.html(f"""

    <div class="navbar">

        <div class="brand">

            <div class="brand-icon">🛡️</div>

            <div>

                <div class="brand-name">DeepGuard</div>

                <div class="brand-sub">AI Media Security</div>

            </div>

        </div>

        <div style="color:#858696;font-size:10px;">

            USER · {st.session_state.username}

        </div>

    </div>

    """)





def prediction_values(fake_probability):

    real_probability = 1.0 - fake_probability



    if fake_probability >= 0.65:

        return "FAKE", fake_probability, "fake", "⚠️"

    if fake_probability <= 0.35:

        return "REAL", real_probability, "real", "✓"

    return "UNCERTAIN", max(fake_probability, real_probability), "uncertain", "◐"





def result_card(prediction, confidence, css_class, icon):

    st.html(f"""

    <div class="result {css_class}">

        <div class="result-icon">{icon}</div>

        <div class="result-label">{prediction}</div>

        <div class="result-confidence">

            Confidence · {confidence * 100:.2f}%

        </div>

    </div>

    """)





def page_heading(label, title_first, title_second, description):

    st.html(f"""

<div style="animation:pageIn .45s ease;">

    <div class="eyebrow">{label}</div>

    <div class="page-title">{title_first}<br><span>{title_second}</span></div>

    <div class="page-description">{description}</div>

</div>

    """)





# ============================================================

# AUTHENTICATION

# ============================================================



def show_auth():

    """Desktop login screen: branding on the left, credentials on the right."""

    left, right = st.columns([1.12, 0.88], gap="small")



    # --------------------------------------------------------

    # LEFT: branding / project introduction

    # --------------------------------------------------------

    with left:

        st.html("""

        <div class="auth-brand-card">

            <div class="auth-glow"></div>

            <div class="auth-label">AI MEDIA SECURITY</div>



            <div class="auth-brand-center">

                <div class="auth-title">

                    SEE THROUGH<br>

                    <span>THE FAKE.</span>

                </div>

                <div class="auth-description">

                    DeepGuard uses computer vision and deep learning

                    to screen images and videos for potential manipulation.

                </div>

            </div>



            <div class="auth-footer">

                <span>ANALYSE</span>

                <i></i>

                <span>DETECT</span>

                <i></i>

                <span>PROTECT</span>

            </div>

        </div>

        """)



    # --------------------------------------------------------

    # RIGHT: REAL STREAMLIT LOGIN / REGISTER CONTROLS

    # --------------------------------------------------------

    with right:

        with st.container(border=True):

            st.html("""

            <div class="auth-form-head">

                <div class="auth-form-kicker">SECURE ACCESS</div>

                <div class="auth-form-title">Welcome to DeepGuard</div>

                <div class="auth-form-subtitle">

                    Sign in to access your AI-powered media analysis workspace.

                </div>

            </div>

            """)



            login_tab, register_tab = st.tabs(["LOGIN", "REGISTER"])



            with login_tab:

                username = st.text_input(

                    "Username",

                    key="login_username",

                    placeholder="Enter username",

                )

                password = st.text_input(

                    "Password",

                    type="password",

                    key="login_password",

                    placeholder="Enter password",

                )



                if st.button("LOGIN  →", key="login_button"):

                    if authenticate_user(username, password):

                        st.session_state.logged_in = True

                        st.session_state.username = username.strip()

                        st.session_state.page = "Dashboard"

                        st.rerun()

                    else:

                        st.error("Invalid username or password.")



                st.html("""

                <div class="auth-hint">

                    Enter your registered credentials to continue.

                </div>

                """)



            with register_tab:

                username = st.text_input(

                    "Username",

                    key="register_username",

                    placeholder="Choose username",

                )

                password = st.text_input(

                    "Password",

                    type="password",

                    key="register_password",

                    placeholder="Create password",

                )

                confirm_password = st.text_input(

                    "Confirm password",

                    type="password",

                    key="register_confirm",

                    placeholder="Repeat password",

                )



                if st.button("CREATE ACCOUNT  →", key="register_button"):

                    if not username.strip():

                        st.error("Username is required.")

                    elif password != confirm_password:

                        st.error("Passwords do not match.")

                    else:

                        success, message = register_user(username, password)

                        if success:

                            st.success(message)

                        else:

                            st.error(message)



    st.html("""

    <div class="auth-bottom-note">

        DEEPGUARD · AI MEDIA SECURITY · LOCAL AUTHENTICATION

    </div>

    """)





# ============================================================

# DASHBOARD

# ============================================================



def show_dashboard():

    show_navbar()



    left, right = st.columns([1.35, .65])



    with left:

        st.html(f"""

        <div class="dark-card" style="height:270px;padding:30px;">

            <div class="eyebrow">

                WELCOME BACK, {st.session_state.username.upper()}

            </div>

            <div class="page-title" style="font-size:48px;">

                Detect what<br><span>isn't real.</span>

            </div>

            <div class="page-description" style="max-width:620px;">

                Analyse digital images and videos for potential deepfake

                manipulation using a ResNet18 deep learning model.

            </div>

        </div>

        """)



    with right:

        st.html("""

        <div class="dark-card" style="height:270px;">

            <div class="card-icon">🛡️</div>

            <div class="card-title">DeepGuard System</div>

            <div class="card-description">

                Image analysis, face detection and video-frame aggregation

                in one compact workspace.

                <br><br>

                <b style="color:#a9a7ff;">Status</b><br>

                Model loaded · Ready

            </div>

        </div>

        """)



    a, b = st.columns(2)



    with a:

        st.html("""

        <div class="dark-card" style="height:165px;">

            <div class="card-icon">◈</div>

            <div class="card-title">Image Analysis</div>

            <div class="card-description">

                Upload a facial image and analyse it for potential

                manipulation.

            </div>

        </div>

        """)

        if st.button("OPEN IMAGE ANALYSIS →", key="open_image"):

            st.session_state.page = "Image Detection"

            st.rerun()



    with b:

        st.html("""

        <div class="dark-card" style="height:165px;">

            <div class="card-icon">◉</div>

            <div class="card-title">Video Analysis</div>

            <div class="card-description">

                Sample multiple frames, detect faces and aggregate

                predictions into a video-level result.

            </div>

        </div>

        """)

        if st.button("OPEN VIDEO ANALYSIS →", key="open_video"):

            st.session_state.page = "Video Detection"

            st.rerun()



    st.html("""

    <div style="

        margin-top:5px;

        padding:9px 14px;

        border-radius:12px;

        border:1px solid rgba(255,255,255,.05);

        background:rgba(12,13,21,.50);

        color:#6f7180;

        font-size:9px;

    ">

        <b style="color:#9c9dab;">SYSTEM NOTE</b>

        &nbsp; DeepGuard is an academic AI screening system.

        Detection results should not be considered definitive forensic evidence.

    </div>

    """)





# ============================================================

# IMAGE DETECTION

# ============================================================



def show_image_detection():

    show_navbar()



    uploaded_file = st.file_uploader(

        "Upload image",

        type=["jpg", "jpeg", "png"],

        key="image_uploader",

    )



    if uploaded_file is None:

        page_heading(

            "IMAGE ANALYSIS",

            "Analyse an",

            "image.",

            "Upload a facial image and let DeepGuard analyse its visual features.",

        )



        st.html("""

        <div class="dark-card" style="

            height:220px;

            margin-top:12px;

            display:flex;

            align-items:center;

            justify-content:center;

            text-align:center;

        ">

            <div>

                <div style="font-size:42px;">◈</div>

                <div class="card-title">Drop your image here</div>

                <div class="card-description">JPG · JPEG · PNG</div>

            </div>

        </div>

        """)

        return



    image = Image.open(uploaded_file).convert("RGB")



    with torch.no_grad():

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)



    fake_probability = probabilities[0][0].item()

    real_probability = probabilities[0][1].item()



    prediction, confidence, css_class, icon = prediction_values(

        fake_probability

    )



    # Everything is horizontal on desktop.

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])



    with c1:

        st.markdown("**INPUT IMAGE**")

        st.image(

            image,

            caption="Uploaded image",

            width=300,

        )



    with c2:

        st.markdown("**DETECTION RESULT**")

        result_card(

            prediction,

            confidence,

            css_class,

            icon,

        )



        st.metric(

            "MODEL CONFIDENCE",

            f"{confidence * 100:.2f}%",

        )



    with c3:

        st.markdown("**ANALYSIS**")



        st.metric(

            "FAKE PROBABILITY",

            f"{fake_probability * 100:.2f}%",

        )

        st.progress(fake_probability)



        st.metric(

            "REAL PROBABILITY",

            f"{real_probability * 100:.2f}%",

        )

        st.progress(real_probability)



    st.warning(

        "AI screening result only — not definitive proof of manipulation."

    )





# ============================================================

# VIDEO DETECTION

# ============================================================



def show_video_detection():

    show_navbar()



    uploaded_video = st.file_uploader(

        "Upload video",

        type=["mp4", "avi", "mov"],

        key="video_uploader",

    )



    if uploaded_video is None:

        page_heading(

            "VIDEO ANALYSIS",

            "Analyse a",

            "video.",

            "Sample multiple frames, detect faces and produce a video-level prediction.",

        )



        st.html("""

        <div class="dark-card" style="

            height:220px;

            margin-top:12px;

            display:flex;

            align-items:center;

            justify-content:center;

            text-align:center;

        ">

            <div>

                <div style="font-size:42px;">◉</div>

                <div class="card-title">Upload your video</div>

                <div class="card-description">MP4 · AVI · MOV</div>

            </div>

        </div>

        """)

        return



    left, right = st.columns([1.0, 1.25])



    with left:

        st.markdown("**VIDEO PREVIEW**")

        st.video(uploaded_video, start_time=0)



        analyze = st.button(

            "START VIDEO ANALYSIS →",

            key="start_video_analysis",

        )



    if not analyze:

        with right:

            st.html("""

            <div class="dark-card" style="height:300px;">

                <div class="card-icon">◎</div>

                <div class="card-title">Ready for analysis</div>

                <div class="card-description">

                    Up to 20 evenly distributed frames will be sampled.

                    Faces are detected and the largest detected face from

                    each usable frame is passed to the ResNet18 model.

                </div>

            </div>

            """)

        return



    # Save upload temporarily.

    suffix = os.path.splitext(uploaded_video.name)[1]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)



    try:

        temp_file.write(uploaded_video.getvalue())

        temp_file.close()



        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)



        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        duration = total_frames / fps if fps > 0 else 0



        sample_count = min(20, total_frames)



        if sample_count <= 0:

            st.error("Could not read this video.")

            cap.release()

            return



        frame_indices = np.linspace(

            0,

            max(total_frames - 1, 0),

            sample_count,

            dtype=int,

        )



        fake_probabilities = []

        detected_faces = 0

        progress = st.progress(0, text="Analysing frames...")



        for i, frame_index in enumerate(frame_indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

            success, frame = cap.read()



            if not success:

                progress.progress(

                    int((i + 1) / len(frame_indices) * 100),

                    text="Analysing frames...",

                )

                continue



            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



            faces = face_detector.detectMultiScale(

                gray,

                scaleFactor=1.1,

                minNeighbors=5,

                minSize=(60, 60),

            )



            if len(faces) == 0:

                progress.progress(

                    int((i + 1) / len(frame_indices) * 100),

                    text="Analysing frames...",

                )

                continue



            detected_faces += 1



            largest_face = max(

                faces,

                key=lambda rect: rect[2] * rect[3],

            )



            x, y, w, h = largest_face

            face = frame[y:y + h, x:x + w]



            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face_image = Image.fromarray(face)



            input_tensor = transform(face_image).unsqueeze(0).to(DEVICE)



            with torch.no_grad():

                output = model(input_tensor)

                probabilities = torch.softmax(output, dim=1)



            fake_probability = probabilities[0][0].item()

            fake_probabilities.append(fake_probability)



            progress.progress(

                int((i + 1) / len(frame_indices) * 100),

                text="Analysing frames...",

            )



        cap.release()



    finally:

        try:

            os.remove(temp_file.name)

        except OSError:

            pass



    if len(fake_probabilities) == 0:

        st.error("No detectable faces were found in the sampled frames.")

        return



    # Robust video-level aggregation
    median_fake_probability = float(np.median(fake_probabilities))
    fake_frame_ratio = float(
        np.mean(np.array(fake_probabilities) >= 0.65)
    )

    # Require strong probability and agreement across frames
    if median_fake_probability >= 0.70 and fake_frame_ratio >= 0.60:
        prediction = "FAKE"
        confidence = median_fake_probability
        css_class = "fake"
        icon = "⚠️"
    elif median_fake_probability <= 0.30 and fake_frame_ratio <= 0.30:
        prediction = "REAL"
        confidence = 1.0 - median_fake_probability
        css_class = "real"
        icon = "✓"
    else:
        prediction = "UNCERTAIN"
        confidence = max(
            median_fake_probability,
            1.0 - median_fake_probability
        )
        css_class = "uncertain"
        icon = "◐"

    average_fake_probability = median_fake_probability
    real_probability = 1.0 - average_fake_probability

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])



    with c1:

        result_card(

            prediction,

            confidence,

            css_class,

            icon,

        )



    with c2:

        st.metric("DURATION", f"{duration:.1f}s")

        st.metric("FRAMES ANALYSED", len(fake_probabilities))



    with c3:

        st.metric("FACES DETECTED", detected_faces)

        st.metric(

            "FAKE PROBABILITY",

            f"{average_fake_probability * 100:.2f}%",

        )

        st.metric(
            "FAKE FRAME AGREEMENT",
            f"{fake_frame_ratio * 100:.0f}%",
        )



    a, b = st.columns(2)



    with a:

        st.caption("Fake probability")

        st.progress(average_fake_probability)



    with b:

        st.caption("Real probability")

        st.progress(real_probability)



    st.warning(

        "Video result is an AI screening prediction and should not be treated "

        "as definitive forensic evidence."

    )





# ============================================================

# ABOUT

# ============================================================



def show_about():

    show_navbar()



    st.html("""

    <div style="animation:pageIn .45s ease;margin-bottom:10px;">

        <div class="eyebrow">ABOUT DEEPGUARD</div>

        <div class="page-title">

            Intelligence for<br>

            <span>digital authenticity.</span>

        </div>

        <div class="page-description">

            An academic deep learning project exploring the detection

            of manipulated facial media.

        </div>

    </div>

    """)



    c1, c2, c3 = st.columns(3)



    with c1:

        st.html("""

        <div class="dark-card" style="height:180px;">

            <div class="card-icon">🧠</div>

            <div class="card-title">ResNet18</div>

            <div class="card-description">

                Transfer learning with a ResNet18 convolutional neural network

                for binary real/fake classification.

            </div>

        </div>

        """)

    with c2:

        st.html("""

        <div class="dark-card" style="height:180px;">

            <div class="card-icon">👤</div>

            <div class="card-title">Face Detection</div>

            <div class="card-description">

                Facial regions are identified before deep learning analysis.

            </div>

        </div>

        """)



    with c3:

        st.html("""

        <div class="dark-card" style="height:180px;">

            <div class="card-icon">🎬</div>

            <div class="card-title">Video Sampling</div>

            <div class="card-description">

                Multiple video frames are analysed and aggregated into

                a video-level prediction.

            </div>

        </div>

        """)



    st.html("""

    <div class="dark-card" style="margin-top:5px;height:125px;">

        <div class="card-title">Dataset & Method</div>

        <div class="card-description">

            The current system uses FaceForensics++ real videos and

            DeepFake-manipulated videos. The dataset is separated into

            training, validation and test subsets at the video level.

        </div>

    </div>

    """)



    st.warning(

        "Deepfake detectors may not generalise perfectly to manipulation "

        "techniques or AI generators that were not represented in training."

    )





# ============================================================

# MAIN ROUTER

# ============================================================



if not st.session_state.logged_in:

    show_auth()

else:

    # Compact horizontal navigation.

    n1, n2, n3, n4, n5 = st.columns([1.15, 1.15, 1.15, 1.15, .8])



    with n1:

        if st.button("HOME", key="nav_home"):

            st.session_state.page = "Dashboard"

            st.rerun()



    with n2:

        if st.button("IMAGE", key="nav_image"):

            st.session_state.page = "Image Detection"

            st.rerun()



    with n3:

        if st.button("VIDEO", key="nav_video"):

            st.session_state.page = "Video Detection"

            st.rerun()



    with n4:

        if st.button("ABOUT", key="nav_about"):

            st.session_state.page = "About"

            st.rerun()



    with n5:

        if st.button("LOGOUT", key="logout"):

            st.session_state.logged_in = False

            st.session_state.username = ""

            st.session_state.page = "Dashboard"

            st.rerun()



    if st.session_state.page == "Dashboard":

        show_dashboard()

    elif st.session_state.page == "Image Detection":

        show_image_detection()

    elif st.session_state.page == "Video Detection":

        show_video_detection()

    elif st.session_state.page == "About":

        show_about()



    st.html("""

    <div style="

        text-align:center;

        color:#4f5160;

        font-size:8px;

        letter-spacing:1px;

        margin-top:4px;

    ">

        🛡️ DEEPGUARD · AI MEDIA SECURITY · RESNET18

    </div>

    """)
