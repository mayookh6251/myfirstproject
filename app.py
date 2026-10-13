

import os
import io
import logging
import time
import tempfile
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import torch
import streamlit as st

from auth import init_db, register_user, authenticate_user
from model_def import get_model, load_model_checkpoint, LABEL_MAP
from extract_faces import resolve_cascade_classifier, detect_and_crop_face
from dataset_loader import get_inference_transform
from evaluate_video import predict_video

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

#supabase client
@st.cache_resource
def get_supabase():
    from supabase import create_client

    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

def save_detection(username, file_name, media_type,
                   prediction, confidence=None):
    try:
        db = get_supabase()

        record = {
            "username": username,
            "file_name": file_name,
            "media_type": media_type,
            "prediction": prediction,
            "confidence": (
                float(confidence)
                if confidence is not None else None
            ),
        }

        db.table("detection_history").insert(
            record
        ).execute()

        return True

    except Exception:
        logger.exception("Failed to save detection history to Supabase.")
        st.warning("The analysis completed, but detection history could not be saved.")
        return False



# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Aletheia AI - Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize database
init_db()

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet18_deepfake.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")
SAMPLE_VIDEOS_DIR = os.path.join(BASE_DIR, "ffpp_videos")


def _load_safe_image(source) -> Image.Image:
    """Validate an uploaded/sample image and return an EXIF-corrected RGB copy."""
    try:
        with Image.open(source) as opened:
            if opened.format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError("Unsupported image format. Use JPEG, PNG, or WEBP.")
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions are invalid or exceed the {MAX_IMAGE_PIXELS:,}-pixel limit."
                )
            corrected = ImageOps.exif_transpose(opened)
            corrected.load()
            return corrected.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The image is corrupt or could not be decoded.") from exc


def _decode_for_opencv(image: Image.Image) -> np.ndarray:
    """Encode normalized RGB pixels and verify OpenCV can decode the image."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Image conversion did not produce valid RGB pixels.")
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("OpenCV could not encode the validated image.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0 or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError("OpenCV could not decode the validated image.")
    return decoded


def _detect_valid_faces(image: np.ndarray, cascade) -> list[tuple[int, int, int, int]]:
    if cascade is None or cascade.empty():
        raise RuntimeError("The face detector is unavailable.")
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("OpenCV returned an invalid image array.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray)
    raw_candidates = []
    for detection_image, scale_factor, min_neighbors, min_size in (
        (gray, 1.1, 5, (30, 30)),
        (equalized_gray, 1.03, 3, (15, 15)),
    ):
        detected = cascade.detectMultiScale(
            detection_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        raw_candidates.extend(detected)

    image_height, image_width = image.shape[:2]
    faces = []
    for candidate in raw_candidates:
        if len(candidate) != 4:
            continue
        x, y, width, height = (int(value) for value in candidate)
        if width <= 0 or height <= 0:
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_width, x + width), min(image_height, y + height)
        if x2 <= x1 or y2 <= y1:
            continue
        box = (x1, y1, x2 - x1, y2 - y1)
        if not any(
            min(box[2] * box[3], kept[2] * kept[3]) > 0
            and (
                max(0, min(x1 + box[2], kept[0] + kept[2]) - max(x1, kept[0]))
                * max(0, min(y1 + box[3], kept[1] + kept[3]) - max(y1, kept[1]))
            ) / min(box[2] * box[3], kept[2] * kept[3]) > 0.8
            for kept in faces
        ):
            faces.append(box)

    faces.sort(key=lambda box: box[2] * box[3], reverse=True)
    return faces


def _make_face_tensors(image: np.ndarray, faces, transform) -> list[torch.Tensor]:
    image_height, image_width = image.shape[:2]
    face_tensors = []
    for x, y, width, height in faces:
        margin_x, margin_y = int(width * 0.15), int(height * 0.15)
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2 = min(image_width, x + width + margin_x)
        y2 = min(image_height, y + height + margin_y)
        if x2 <= x1 or y2 <= y1:
            continue
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0 or face_crop.ndim != 3 or face_crop.shape[2] != 3:
            continue
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_tensor = transform(Image.fromarray(face_rgb))
        if (
            not isinstance(face_tensor, torch.Tensor)
            or face_tensor.shape != (3, 224, 224)
            or not torch.isfinite(face_tensor).all()
        ):
            raise ValueError("Face preprocessing returned an invalid model input.")
        face_tensors.append(face_tensor)
    return face_tensors


def _analyze_single_face(image: Image.Image, cascade, transform, model, device) -> dict:
    cv_image = _decode_for_opencv(image)
    faces = _detect_valid_faces(cv_image, cascade)
    if not faces:
        return {"status": "no_face"}
    if len(faces) > 1:
        return {"status": "multiple_faces", "face_count": len(faces)}

    face_tensors = _make_face_tensors(cv_image, faces, transform)
    if not face_tensors:
        raise ValueError("Face detection returned a box but no valid face crop.")

    batch = torch.stack(face_tensors).to(device)
    with torch.no_grad():
        start_time = time.time()
        logits = model(batch)
        latency = (time.time() - start_time) * 1000.0
        if logits.ndim != 2 or logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("The model returned invalid output for the detected face.")
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_index = int(np.argmax(probabilities))
    pred_class = "FAKE" if pred_index == 0 else "REAL"
    return {
        "status": "ok",
        "prediction": pred_class,
        "confidence": float(probabilities[pred_index]),
        "fake_prob": float(probabilities[0]),
        "real_prob": float(probabilities[1]),
        "latency": latency,
    }


def ensure_ascii_safe_copy(path: str) -> str:
    """Copy non-ASCII file paths to a temp folder so OpenCV can read them reliably."""
    if not path or not os.path.exists(path):
        return path

    abs_path = os.path.abspath(path)
    if all(ord(ch) < 128 for ch in abs_path):
        return abs_path

    safe_dir = os.path.join(tempfile.gettempdir(), "deepguard_ai")
    os.makedirs(safe_dir, exist_ok=True)
    safe_target = os.path.join(safe_dir, os.path.basename(abs_path))

    if not os.path.exists(safe_target) or os.path.getsize(safe_target) != os.path.getsize(abs_path):
        shutil.copy2(abs_path, safe_target)

    return safe_target


# ==============================================================================
# Cached Resources
# ==============================================================================
@st.cache_resource
def load_deepfake_detector():
    """Loads and caches the ResNet18 model and face cascade classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cascade_path = ensure_ascii_safe_copy(os.path.join(BASE_DIR, "cascade", "haarcascade_frontalface_default.xml"))
    
    try:
        model = load_model_checkpoint(MODEL_PATH, device=device)
        model_loaded = True
        model_error = None
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        model = None
        model_loaded = False
        model_error = str(e)

    try:
        cascade = resolve_cascade_classifier(cascade_path)
    except Exception as e:
        cascade = None
        st.error(f"Error loading face detector: {e}")

    transform = get_inference_transform()
    return model, cascade, transform, device, model_loaded, model_error


# ==============================================================================
# Futuristic CSS Styling
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Dark forensic theme */
    .stApp {
        background-color: #070b12;
        background-image:
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

    /* Glass cards */
    .glass-card {
        background: rgba(13, 21, 38, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 28px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        transition: all 0.25s ease;
        z-index: 1;
        margin-top: 20px;
    }
    
    
    .auth-panel {
        max-width: 440px;
        width: 100%;
        margin-left: auto;
        min-height: 330px;
        padding: 24px;
        box-sizing: border-box;
        background: rgba(13, 21, 38, 0.82);
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        margin-top:20px;
    }

    .st-key-auth_panel {
        max-width: 440px;
        width: 100%;
        margin-left: auto;
        min-height: 330px;
        padding: 24px;
        box-sizing: border-box;
        background: rgba(13, 21, 38, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        margin-top:20px;
    }

    .auth-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #00e5ff;
        margin-bottom: 4px;
    }

    .stTabs [data-testid="stBaseButton-secondary"], .stTabs [role="tab"] {
        min-height: 34px !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        border-radius: 12px !important;
        font-size: 0.78rem !important;
        letter-spacing: 1px;
        color: #94a3b8 !important;
        background: rgba(10, 16, 29, 0.75) !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: rgba(0, 229, 255, 0.16) !important;
        color: #ffffff !important;
    }
    .stForm {
        margin-top: 0 !important;
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.35);
        box-shadow: 0 10px 36px rgba(0, 229, 255, 0.12);
    }

    /* Hero Brand Text */
    .brand-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.06em;
        color: #00e5ff;
        margin-bottom: 8px;
    }
    .tagline {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.05em;
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .subtagline {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.7;
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

    /* Nav Bar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background: rgba(13, 21, 38, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    /* Button overrides */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.2) 0%, rgba(121, 40, 202, 0.25) 100%);
        color: #00e5ff;
        border: 1px solid #00e5ff;
        border-radius: 12px;
        padding: 10px 18px;
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
        transform: translateY(-1px);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.45);
    }

    /* Streamlit Input Styling */
    div[data-baseweb="input"] {
        background-color: rgba(10, 16, 29, 0.8) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 229, 255, 0.25) !important;
        color: #f1f5f9 !important;
        box-shadow: none !important;
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
    """, unsafe_allow_html=True)


# ==============================================================================
# Authentication State Management
# ==============================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


def render_login_page():
    """Renders the presentation-ready, true 2-column horizontal login screen."""
    inject_custom_css()

    col_hero, col_form = st.columns([1.3, 0.9], gap="large")

    # LEFT COLUMN: Visual Brand Hero
    with col_hero:
        st.markdown("""
        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; min-height: 410px; padding-top: 22px; padding-bottom: 22px;">
            <div style="display: inline-block; padding: 4px 12px; background: rgba(0, 229, 255, 0.1); border: 1px solid #00e5ff; border-radius: 20px; font-size: 0.75rem; letter-spacing: 2px; color: #00e5ff; margin-bottom: 14px; width: fit-content;">
                AI MEDIA FORENSICS SUITE
            </div>
            <div class="brand-title" style="font-size: 2.0rem;">ALETHEIA AI</div>
            <div class="tagline" style="font-size: 1.3rem;">SEE THROUGH<br/>THE FAKE.</div>
            <p class="subtagline" style="margin-top: 8px; margin-bottom: 0;">
                Aletheia AI analyzes visual media using state-of-the-art computer vision
                and deep transfer learning to protect media authenticity and expose synthetic face manipulation.
            </p>
            <div style="margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="display: flex; gap: 18px; color: #64748b; font-size: 0.82rem; letter-spacing: 2px; font-weight: 600;">
                    <span>• ANALYSE</span>
                    <span>• DETECT</span>
                    <span>• PROTECT</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT COLUMN: Secure Access Authentication Form
    
    # RIGHT COLUMN: Secure Access Authentication Form
    with col_form:
        with st.container(key="auth_panel"):
            st.markdown(
                """
                <div class="auth-title">
                    SECURE ACCESS
                </div>
                <div class="subtagline"
                     style="font-size: 0.85rem; margin-bottom: 16px;">
                    Authenticate to access Aletheia forensic tools
                </div>
                """,
                unsafe_allow_html=True,
            )

            tab_login, tab_register = st.tabs(
                ["[ LOGIN ]", "[ REGISTER ]"]
            )

            with tab_login:
                with st.form("login_form", clear_on_submit=False):
                    username_input = st.text_input(
                        "Username",
                        key="login_user",
                        placeholder="Enter your username",
                    )
                    password_input = st.text_input(
                        "Password",
                        type="password",
                        key="login_pass",
                        placeholder="Enter password",
                    )
                    submit_login = st.form_submit_button(
                        "LOGIN →",
                        use_container_width=True,
                    )

                    if submit_login:
                        success, msg = authenticate_user(
                            username_input, password_input
                        )
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username_input.strip()
                            st.session_state.current_page = "Dashboard"
                            st.rerun()
                        else:
                            st.error(msg)

                st.caption(
                    "Default college viva demo: **admin** / **admin123**"
                )

            with tab_register:
                with st.form("register_form", clear_on_submit=True):
                    new_username = st.text_input(
                        "Choose Username",
                        key="reg_user",
                        placeholder="Min 3 characters (letters, numbers)",
                    )
                    new_password = st.text_input(
                        "Choose Password",
                        type="password",
                        key="reg_pass",
                        placeholder="Min 6 characters",
                    )
                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                        key="reg_conf",
                        placeholder="Re-type password",
                    )
                    submit_reg = st.form_submit_button(
                        "CREATE ACCOUNT →",
                        use_container_width=True,
                    )

                    if submit_reg:
                        if new_password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            reg_success, reg_msg = register_user(
                                new_username, new_password
                            )
                            if reg_success:
                                st.success(
                                    f"{reg_msg} You can now log in."
                                )
                            else:
                                st.error(reg_msg)


        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# Main Authenticated Application Dashboard
# ==============================================================================
def render_navbar():
    """Renders the top navigation bar."""
    st.markdown(f"""
    <div class="navbar">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.6rem;">🛡️</span>
            <div>
                <span style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; font-weight: 800; color: #00e5ff; letter-spacing: 1px;">ALETHEIA AI</span>
                <span style="font-size: 0.75rem; color: #64748b; margin-left: 10px; border-left: 1px solid #334155; padding-left: 10px;">OPERATOR: <b style="color: #94a3b8;">{st.session_state.username.upper()}</b></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_cols = st.columns(7)
    with nav_cols[0]:
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    with nav_cols[1]:
        if st.button("🖼️ Image Detection", use_container_width=True):
            st.session_state.current_page = "Image Detection"
            st.rerun()
    with nav_cols[2]:
        if st.button("🎥 Video Detection", use_container_width=True):
            st.session_state.current_page = "Video Detection"
            st.rerun()
    with nav_cols[3]:
        if st.button("📊 Evaluation", use_container_width=True):
            st.session_state.current_page = "Model Evaluation"
            st.rerun()
    with nav_cols[4]:
        if st.button("ℹ️ About System", use_container_width=True):
            st.session_state.current_page = "About"
            st.rerun()
    

    with nav_cols[5]:
        if st.button("📜 History", use_container_width=True):
            st.session_state.current_page = "History"
            st.rerun()

    with nav_cols[6]:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()



# ==============================================================================
# Dashboard Page
# ==============================================================================
def render_dashboard(model_loaded: bool, device: torch.device):
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 24px; text-align: center; padding: 32px 20px;">
        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.8rem; font-weight: 800; color: #f8fafc; letter-spacing: 2px;">
            AI-POWERED MEDIA AUTHENTICITY ANALYSIS
        </div>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 8px;">
            Protecting digital ecosystems by analyzing facial manipulation cues through transfer learning CNNs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.3, 1.3, 1.0], gap="medium")

    with c1:
        st.markdown("""
        <div class="glass-card" style="min-height: 260px;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🖼️</div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; font-weight: 700; color: #00e5ff;">
                IMAGE DETECTION
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 10px; min-height: 60px;">
                Upload standalone portraits (JPG, PNG, WEBP). Analyzes facial region, classifies authenticity, and provides confidence probabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("LAUNCH IMAGE DETECTOR →", key="dash_img_btn", use_container_width=True):
            st.session_state.current_page = "Image Detection"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="glass-card" style="min-height: 260px;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🎥</div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; font-weight: 700; color: #7928ca;">
                VIDEO DETECTION
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 10px; min-height: 60px;">
                Analyze videos (MP4, AVI, MOV). Uniformly samples frames, crops faces, and aggregates probabilities across temporal frames.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("LAUNCH VIDEO DETECTOR →", key="dash_vid_btn", use_container_width=True):
            st.session_state.current_page = "Video Detection"
            st.rerun()

    with c3:
        status_text = "ONLINE" if model_loaded else "OFFLINE"
        status_color = "#00ff88" if model_loaded else "#ff3366"
        st.markdown(f"""
        <div class="glass-card" style="min-height: 260px;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 16px;">
                SYSTEM TELEMETRY
            </div>
            <div class="metric-box">
                <div class="metric-value">ResNet18</div>
                <div class="metric-label">NEURAL BACKBONE</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: #7928ca;">{str(device).upper()}</div>
                <div class="metric-label">INFERENCE DEVICE</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: {status_color};">{status_text}</div>
                <div class="metric-label">MODEL STATUS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# Image Detection Page
# ==============================================================================
def render_image_detection(model, cascade, transform, device, model_loaded: bool):
    st.markdown("""
    <div style="margin-bottom: 18px;">
        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: #00e5ff;">
            IMAGE AUTHENTICITY ANALYSIS
        </div>
        <div style="color: #94a3b8; font-size: 0.9rem;">
            Upload a portrait image or select a sample image for deep learning facial forensic inspection.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("The detection model is unavailable. See the model error above.")
        return

    # Compact 3-column layout
    col_upload, col_preview, col_results = st.columns([1.1, 1.2, 1.3], gap="medium")

    uploaded_file = None
    with col_upload:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 1. Media Input")
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported: JPG, JPEG, PNG, WEBP"
        )

        # Quick preset selector for immediate viva demo
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 12px 0;'>", unsafe_allow_html=True)
        st.caption("Or test with preloaded demonstration samples:")
        sample_choice = st.selectbox(
            "Select sample:",
            ["(None)", "Real Portrait Sample", "Fake Manipulated Sample", "Scene with No Face"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Resolve image source
    image_to_process = None
    if uploaded_file is not None:
        try:
            image_to_process = _load_safe_image(io.BytesIO(uploaded_file.getvalue()))
        except (ValueError, OSError):
            logger.exception("Image validation failed for uploaded file %r.", uploaded_file.name)
            st.error(
                "Image rejected. Upload a valid JPEG, PNG, or WEBP image smaller than "
                f"{MAX_IMAGE_PIXELS:,} pixels."
            )
    elif sample_choice != "(None)":
        sample_map = {
            "Real Portrait Sample": os.path.join(SAMPLE_IMAGES_DIR, "real_face_01.jpg"),
            "Fake Manipulated Sample": os.path.join(SAMPLE_IMAGES_DIR, "fake_face_01.jpg"),
            "Scene with No Face": os.path.join(SAMPLE_IMAGES_DIR, "no_face_01.jpg"),
        }
        target_path = sample_map[sample_choice]
        if os.path.exists(target_path):
            try:
                image_to_process = _load_safe_image(target_path)
            except (ValueError, OSError):
                logger.exception("Image validation failed for sample %r.", target_path)
                st.error("The selected sample image could not be loaded. Details were logged.")
        else:
            st.warning("The selected sample image is not available.")

    with col_preview:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 2. Inspection Preview")
        if image_to_process is not None:
            # Display compact image (max width ~320px)
            st.image(image_to_process, caption="Input Media", use_container_width=True)
        else:
            st.info("Upload an image on the left or select a sample to preview.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 3. AI Forensic Verdict")
        analyze_clicked = st.button("🔍 ANALYZE IMAGE", use_container_width=True)

        if image_to_process is not None and analyze_clicked:
            try:
                analysis = _analyze_single_face(image_to_process, cascade, transform, model, device)
            except (cv2.error, OSError, RuntimeError, ValueError):
                logger.exception(
                    "Image analysis failed for %r during face detection, preprocessing, or inference.",
                    uploaded_file.name if uploaded_file is not None else sample_choice,
                )
                st.error(
                    "Image analysis failed. Try a different supported image; "
                    "the detailed error was logged."
                )
            else:
                if analysis["status"] == "no_face":
                    st.warning(
                        "No face was detected. No authenticity verdict was generated; "
                        "try a clearer image with a visible face."
                    )
                elif analysis["status"] == "multiple_faces":
                    st.warning(
                        f"{analysis['face_count']} faces were detected. No authenticity verdict "
                        "was generated; upload an image with one clearly visible face."
                    )
                else:
                    pred_class = analysis["prediction"]
                    confidence = analysis["confidence"]
                    fake_prob = analysis["fake_prob"]
                    real_prob = analysis["real_prob"]
                    latency = analysis["latency"]
                    badge_html = (
                        '<div class="badge-fake">MANIPULATED / FAKE</div>'
                        if pred_class == "FAKE"
                        else '<div class="badge-real">AUTHENTIC / REAL</div>'
                    )

                    save_detection(
                        username=st.session_state.username,
                        file_name=uploaded_file.name if uploaded_file is not None else sample_choice,
                        media_type="image",
                        prediction=pred_class,
                        confidence=confidence,
                    )

                    st.markdown(
                        f"<div style='text-align: center; margin: 16px 0;'>{badge_html}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Confidence:** `{confidence * 100:.1f}%`")
                    st.caption("Faces analyzed: 1")
                    st.markdown(f"Fake Probability (`{fake_prob * 100:.1f}%`):")
                    st.progress(fake_prob)
                    st.markdown(f"Real Probability (`{real_prob * 100:.1f}%`):")
                    st.progress(real_prob)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-top: 14px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                        <span>Faces Analyzed: 1</span>
                        <span>Inference: {latency:.1f} ms</span>
                        <span>Device: {str(device).upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)
        elif image_to_process is not None:
            st.info("Image ready. Select Analyze Image to run the forensic inspection.")
        else:
            st.info("Awaiting image input...")

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# Video Detection Page
# ==============================================================================
def render_video_detection(model, cascade, transform, device, model_loaded: bool):
    st.markdown("""
    <div style="margin-bottom: 18px;">
        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: #00e5ff;">
            VIDEO-LEVEL MEDIA FORENSICS
        </div>
        <div style="color: #94a3b8; font-size: 0.9rem;">
            Analyzes multiple frames across video timeline, extracts faces, and aggregates probabilities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("The detection model is unavailable. See the model error above.")
        return

    col_input, col_analysis = st.columns([1.1, 1.4], gap="medium")

    video_source_path = None
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 1. Video Source")
        uploaded_video = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supports standard formats like MP4, AVI, MOV"
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 12px 0;'>", unsafe_allow_html=True)
        st.caption("Or test with demonstration videos:")
        sample_v_choice = st.selectbox(
            "Select sample video:",
            ["(None)", "Real Video Sample 01", "Fake DeepFake Sample 01"]
        )

        if uploaded_video is not None:
            # Save to temporary file for OpenCV access
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_source_path = tfile.name
        elif sample_v_choice != "(None)":
            sample_v_map = {
                "Real Video Sample 01": os.path.join(SAMPLE_VIDEOS_DIR, "real", "real_video_01.mp4"),
                "Fake DeepFake Sample 01": os.path.join(SAMPLE_VIDEOS_DIR, "fake", "fake_video_01.mp4"),
            }
            video_source_path = ensure_ascii_safe_copy(sample_v_map[sample_v_choice])

        # Compact video preview
        if video_source_path and os.path.exists(video_source_path):
            st.markdown("<div style='max-width: 360px;'>", unsafe_allow_html=True)
            st.video(video_source_path)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col_analysis:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 2. Analysis Configuration & Results")

        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            sample_frames_count = st.slider("Frames to sample", min_value=5, max_value=40, value=20, step=5)
        with cfg_col2:
            agg_method = st.selectbox("Aggregation Strategy", ["mean", "median", "majority"])

        analyze_clicked = st.button("🔍 ANALYZE VIDEO", use_container_width=True)

        if analyze_clicked and video_source_path:
            with st.spinner("Processing video frames and extracting facial features..."):
                prog_bar = st.progress(0)
                for pct in range(20, 90, 20):
                    time.sleep(0.05)
                    prog_bar.progress(pct)

                results = predict_video(
                    video_path=video_source_path,
                    model=model,
                    classifier=cascade,
                    transform=transform,
                    device=device,
                    sample_frames=sample_frames_count,
                    aggregation=agg_method
                )
                prog_bar.progress(100)

            if results["prediction"] == "NO_FACE":
                st.error("❌ " + results["message"])
            else:
                pred = results["prediction"]
                conf = results["confidence"]
                p_fake = results["fake_probability"]
                p_real = results["real_probability"]
                
                save_detection(
                    username=st.session_state.username,
                    file_name=(
                        uploaded_video.name
                        if uploaded_video is not None
                        else os.path.basename(video_source_path)
                    ),
                    media_type="video",
                    prediction=pred,
                    confidence=conf
                )


                if pred == "FAKE":
                    badge_html = '<div class="badge-fake">VIDEO VERDICT: FAKE</div>'
                elif pred == "REAL":
                    badge_html = '<div class="badge-real">VIDEO VERDICT: REAL</div>'
                else:
                    badge_html = '<div class="badge-uncertain">VIDEO VERDICT: UNCERTAIN</div>'

                st.markdown(f"<div style='text-align: center; margin: 18px 0;'>{badge_html}</div>", unsafe_allow_html=True)

                # Metric highlights
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"<div class='metric-box'><div class='metric-value'>{results['frames_analyzed']}</div><div class='metric-label'>FRAMES</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-box'><div class='metric-value'>{results['faces_detected']}</div><div class='metric-label'>FACES</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-box'><div class='metric-value'>{conf*100:.1f}%</div><div class='metric-label'>CONFIDENCE</div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='metric-box'><div class='metric-value'>{p_fake*100:.1f}%</div><div class='metric-label'>AVG FAKE PROB</div></div>", unsafe_allow_html=True)

                # Multi-frame timeline
                if len(results["frame_probabilities"]) > 1:
                    st.markdown("###### Temporal Frame-by-Frame Fake Probability")
                    chart_df = pd.DataFrame({
                        "Frame": [f"F#{i+1}" for i in range(len(results["frame_probabilities"]))],
                        "Fake Probability": results["frame_probabilities"]
                    }).set_index("Frame")
                    st.line_chart(chart_df, height=180)

        elif not video_source_path:
            st.info("Please select or upload a video file to begin analysis.")

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# Model Evaluation Page
# ==============================================================================
def render_evaluation_page():
    st.markdown("""
    <div style="margin-bottom: 18px;">
        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: #00e5ff;">
            EMPIRICAL MODEL EVALUATION & METRICS
        </div>
        <div style="color: #94a3b8; font-size: 0.9rem;">
            Academic evaluation results computed directly from the test dataset partition.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_loss = st.columns(2, gap="medium")

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    hist_path = os.path.join(RESULTS_DIR, "training_history.png")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")

    with col_cm:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Confusion Matrix (Test Split)")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix [0=FAKE, 1=REAL]", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found. Run `python evaluate.py` to generate.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_loss:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Training History Curves")
        if os.path.exists(hist_path):
            st.image(hist_path, caption="Loss & Accuracy across Epochs", use_container_width=True)
        else:
            st.info("Training history plot not found. Run `python train_transfer.py` to generate.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card" style="margin-top: 18px;">', unsafe_allow_html=True)
    st.markdown("##### Academic Metrics Report")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            st.code(f.read(), language="text")
    else:
        st.info("Metrics report file not found. Run `python evaluate.py` to generate.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# About Page
# ==============================================================================
def render_about_page():
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px;">
        <div class="brand-title">PROJECT SPECIFICATIONS</div>
        <div style="color: #94a3b8; font-size: 0.95rem; line-height: 1.7;">
            <b>Project Title:</b> Deepfake Detection: Protecting Images and Videos Using Deepfake Detection (Aletheia AI)<br/>
            <b>Academic Purpose:</b> Final-Year Computer Science Engineering Capstone Project<br/>
            <b>Core Architecture:</b> ResNet18 Transfer Learning with Custom Dropout & Binary Classification Head<br/>
            <b>Benchmark Dataset:</b> FaceForensics++ (DeepFakes Manipulation Category)
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("""
        <div class="glass-card" style="min-height: 280px;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.15rem; font-weight: 700; color: #00e5ff; margin-bottom: 12px;">
                FORENSIC PIPELINE ARCHITECTURE
            </div>
            <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.05rem; line-height: 1.8; color: #cbd5e1;">
                1. <b>Video Input:</b> Read video frames via OpenCV.<br/>
                2. <b>Temporal Sampling:</b> Extract uniformly spaced frames.<br/>
                3. <b>Face Localization:</b> Detect facial bounding box via Haar Cascade.<br/>
                4. <b>Margin Cropping & Preprocessing:</b> Crop face + 15% margin, resize to 224×224, apply ImageNet normalization.<br/>
                5. <b>ResNet18 Inference:</b> Extract deep representations, apply dropout (p=0.5), compute softmax probabilities.<br/>
                6. <b>Temporal Aggregation:</b> Aggregate probabilities across frames.<br/>
                7. <b>Uncertainty Gating:</b> Classify as REAL, FAKE, or UNCERTAIN.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card" style="min-height: 280px;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.15rem; font-weight: 700; color: #ffb800; margin-bottom: 12px;">
                ACADEMIC HONESTY & LIMITATIONS DISCLOSURE
            </div>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.7;">
                Deepfake detection is inherently <b>probabilistic</b>. Real-world performance depends heavily on:
            </p>
            <ul style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                <li>Training dataset distribution and variety of faces</li>
                <li>Video compression artifacts (e.g. H.264/H.265 compression loss)</li>
                <li>Novel GAN / Diffusion manipulation techniques not seen during training</li>
                <li>Extreme illumination angles, heavy motion blur, or partial facial occlusion</li>
            </ul>
            <p style="color: #64748b; font-size: 0.82rem; margin-top: 10px;">
                <i>Note: This system does not claim 100% universal accuracy on arbitrary wild media. The UNCERTAIN classification prevents false confidence on ambiguous samples.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# Detection History Page
# ==============================================================================

def render_detection_history():
    st.markdown(
        "## 📜 Detection History"
    )
    st.caption(
        "View your previously completed image and video analyses."
    )

    if st.button("🔄 Refresh History"):
        st.rerun()

    try:
        db = get_supabase()

        response = (
            db.table("detection_history")
            .select(
                "created_at, file_name, media_type, "
                "prediction, confidence"
            )
            .eq("username", st.session_state.username)
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        records = response.data

        if records:
            st.success(
                f"Found {len(records)} detection records."
            )

            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No detection history yet. "
                "Analyze an image or video first."
            )

    except Exception as e:
        st.error(
            f"Could not load detection history: {e}"
        )


# ==============================================================================
# Application Router
# ==============================================================================
def main():
    if not st.session_state.authenticated:
        render_login_page()
    else:
        inject_custom_css()
        render_navbar()

        model, cascade, transform, device, model_loaded, model_error = load_deepfake_detector()
        if model_error:
            st.error(f"Model unavailable: {model_error}")

        if st.session_state.current_page == "Dashboard":
            render_dashboard(model_loaded, device)
        elif st.session_state.current_page == "Image Detection":
            render_image_detection(model, cascade, transform, device, model_loaded)
        elif st.session_state.current_page == "Video Detection":
            render_video_detection(model, cascade, transform, device, model_loaded)
        elif st.session_state.current_page == "Model Evaluation":
            render_evaluation_page()
        elif st.session_state.current_page == "About":
            render_about_page()
        elif st.session_state.current_page == "History":
            render_detection_history()


if __name__ == "__main__":
    main()
