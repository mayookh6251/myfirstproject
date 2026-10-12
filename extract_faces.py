"""
DeepGuard AI - Face Detection and Cropping Pipeline
===================================================
Detects and extracts facial regions from extracted frames using OpenCV Haar Cascade.
Features:
- Selects the largest/most relevant face if multiple faces are detected
- Adds natural bounding box margin to preserve boundary blending artifacts
- Reports frames processed, faces detected, faces missed, and output count
- Uses explicit robust cascade path resolution with auto-download fallback
"""

import os
import glob
import argparse
import shutil
import tempfile
import urllib.request
import cv2
import numpy as np

DEFAULT_CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascade", "haarcascade_frontalface_default.xml")
CASCADE_GITHUB_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


def _copy_to_ascii_safe_path(path: str) -> str:
    """Copy a resource to a temp directory with ASCII-only characters for OpenCV compatibility."""
    if not path or not os.path.exists(path):
        return path

    abs_path = os.path.abspath(path)
    if all(ord(ch) < 128 for ch in abs_path):
        return abs_path

    safe_dir = os.path.join(tempfile.gettempdir(), "deepguard_ai")
    os.makedirs(safe_dir, exist_ok=True)
    safe_path = os.path.join(safe_dir, os.path.basename(abs_path))

    if not os.path.exists(safe_path) or os.path.getsize(safe_path) != os.path.getsize(abs_path):
        shutil.copy2(abs_path, safe_path)

    return safe_path


def resolve_cascade_classifier(explicit_path: str = DEFAULT_CASCADE_PATH) -> cv2.CascadeClassifier:
    """
    Resolves and loads the Haar cascade classifier file reliably.
    Downloads the official XML file automatically if missing.
    """
    search_paths = [
        explicit_path,
        os.path.join("cascade", "haarcascade_frontalface_default.xml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"),
    ]

    # Try cv2.data if it exists
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        search_paths.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    found_path = None
    for p in search_paths:
        if p and os.path.exists(p) and os.path.getsize(p) > 1000:
            found_path = p
            break

    if not found_path:
        print("[!] Haar cascade file not found locally. Auto-downloading official OpenCV weights...")
        os.makedirs(os.path.dirname(explicit_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(CASCADE_GITHUB_URL, explicit_path)
            found_path = explicit_path
            print(f"[OK] Downloaded cascade XML to {explicit_path}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to locate or download haarcascade_frontalface_default.xml: {e}")

    # OpenCV can fail to load XML from non-ASCII paths on Windows/OneDrive mounts.
    found_path = _copy_to_ascii_safe_path(found_path)

    classifier = cv2.CascadeClassifier(found_path)
    if classifier.empty():
        raise ValueError(f"Failed to initialize cv2.CascadeClassifier from: {found_path}")

    return classifier


def detect_and_crop_face(
    image: np.ndarray,
    classifier: cv2.CascadeClassifier,
    margin_ratio: float = 0.15
) -> np.ndarray:
    """
    Detects faces in an image and crops the largest face.
    Returns cropped BGR face image, or None if no face detected.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        # Fallback with relaxed parameters
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

    if len(faces) == 0:
        return None

    # Pick the largest face (w * h)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Apply bounding box margin
    img_h, img_w = image.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)

    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def process_frames_to_faces(
    frames_base_dir: str = "frames",
    faces_base_dir: str = "faces",
    cascade_path: str = DEFAULT_CASCADE_PATH,
    margin: float = 0.15
) -> None:
    """
    Iterates through all frames, detects faces, crops the largest face, and saves it.
    """
    print("=" * 60)
    print(" DeepGuard AI - Face Extraction Pipeline")
    print("=" * 60)
    print(f"[*] Frames source directory: {frames_base_dir}")
    print(f"[*] Faces output directory:  {faces_base_dir}")
    print(f"[*] Cascade path:            {cascade_path}")

    classifier = resolve_cascade_classifier(cascade_path)
    print("[OK] Face detector initialized successfully.")

    categories = ["real", "fake"]
    total_frames = 0
    total_detected = 0
    total_missed = 0
    total_saved = 0

    for category in categories:
        cat_frames_dir = os.path.join(frames_base_dir, category)
        cat_faces_dir = os.path.join(faces_base_dir, category)

        if not os.path.exists(cat_frames_dir):
            os.makedirs(cat_frames_dir, exist_ok=True)
            continue

        video_subdirs = [
            d for d in os.listdir(cat_frames_dir)
            if os.path.isdir(os.path.join(cat_frames_dir, d))
        ]

        print(f"\n[*] Processing [{category.upper()}] video directories: {len(video_subdirs)} found.")

        for v_sub in video_subdirs:
            in_video_dir = os.path.join(cat_frames_dir, v_sub)
            out_video_dir = os.path.join(cat_faces_dir, v_sub)
            os.makedirs(out_video_dir, exist_ok=True)

            frame_files = sorted(glob.glob(os.path.join(in_video_dir, "*.jpg")) + glob.glob(os.path.join(in_video_dir, "*.png")))
            
            for fpath in frame_files:
                total_frames += 1
                img = cv2.imread(fpath)
                if img is None:
                    total_missed += 1
                    continue

                face_crop = detect_and_crop_face(img, classifier, margin_ratio=margin)
                if face_crop is not None:
                    total_detected += 1
                    fname = os.path.basename(fpath).replace("frame_", "face_")
                    out_path = os.path.join(out_video_dir, fname)
                    if cv2.imwrite(out_path, face_crop):
                        total_saved += 1
                else:
                    total_missed += 1

    print("\n" + "=" * 60)
    print(" Face Extraction Summary Report:")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Faces detected:         {total_detected}")
    print(f"  Faces missed:           {total_missed}")
    print(f"  Total face images saved: {total_saved}")
    if total_frames > 0:
        det_rate = (total_detected / total_frames) * 100
        print(f"  Detection rate:         {det_rate:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from video frames.")
    parser.add_argument("--frames_dir", type=str, default="frames", help="Directory containing extracted frames")
    parser.add_argument("--faces_dir", type=str, default="faces", help="Output directory for face crops")
    parser.add_argument("--cascade_path", type=str, default=DEFAULT_CASCADE_PATH, help="Path to Haar cascade XML")
    parser.add_argument("--margin", type=float, default=0.15, help="Face crop margin ratio (default: 0.15)")
    args = parser.parse_args()

    process_frames_to_faces(
        frames_base_dir=args.frames_dir,
        faces_base_dir=args.faces_dir,
        cascade_path=args.cascade_path,
        margin=args.margin
    )
