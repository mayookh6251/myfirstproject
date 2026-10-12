"""
DeepGuard AI - Sample Dataset & Demonstration Generator
======================================================
Generates an immediate, presentation-ready benchmark dataset for college project
demonstrations and testing without needing to download massive external archives.

Generates:
- REAL videos in ffpp_videos/real/
- FAKE videos (with synthetic deepfake boundary/warping artifacts) in ffpp_videos/fake/
- Standalone test images in sample_images/ for quick Streamlit testing

Note:
When real FaceForensics++ videos are available, simply copy them into ffpp_videos/real/
and ffpp_videos/fake/ to train on the full research benchmark.
"""

import os
import urllib.request
import cv2
import numpy as np


def download_base_faces() -> list:
    """Downloads public domain / sample portrait images for demonstration generation."""
    os.makedirs("sample_images", exist_ok=True)
    sources = [
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", "sample_images/base_face_1.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg", "sample_images/base_face_2.jpg"),
    ]

    loaded_faces = []
    headers = {"User-Agent": "DeepGuardAI/1.0 (academic; student@test.edu)"}

    for url, save_path in sources:
        try:
            if not os.path.exists(save_path) or os.path.getsize(save_path) < 1000:
                print(f"[*] Downloading base face: {os.path.basename(save_path)}...")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    with open(save_path, "wb") as f:
                        f.write(resp.read())
            img = cv2.imread(save_path)
            if img is not None:
                # Resize to standard 480x480 resolution
                img = cv2.resize(img, (480, 480))
                loaded_faces.append(img)
        except Exception as e:
            print(f"[!] Warning: Could not download {url}: {e}")

    # Fallback if download fails (procedural realistic face synthesis)
    if len(loaded_faces) < 2:
        print("[*] Generating procedural fallback faces...")
        for i in range(2 - len(loaded_faces)):
            base = np.full((480, 480, 3), (215, 225, 235), dtype=np.uint8)
            cv2.ellipse(base, (240, 240), (140, 180), 0, 0, 360, (170, 190, 220), -1)
            loaded_faces.append(base)

    return loaded_faces


def create_real_video(base_face: np.ndarray, output_path: str, num_frames: int = 40, fps: float = 20.0):
    """Creates a realistic video with natural camera micro-motion and subtle illumination changes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = base_face.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(num_frames):
        # Subtle translation and slight zoom
        dx = int(3 * np.sin(i * 0.15))
        dy = int(2 * np.cos(i * 0.15))
        scale = 1.0 + 0.02 * np.sin(i * 0.1)

        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        frame = cv2.warpAffine(base_face, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Gentle natural lighting variation
        gain = 1.0 + 0.03 * np.sin(i * 0.2)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()


def create_fake_video(base_face: np.ndarray, output_path: str, num_frames: int = 40, fps: float = 20.0):
    """
    Creates a deepfake-manipulated video exhibiting classic FaceForensics++ artifacts:
    - Facial blending border seam (Gaussian blur edge mismatch)
    - Color tone temperature discrepancy inside facial mask
    - Affine warping jitter
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = base_face.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, int(h * 0.52)), (int(w * 0.28), int(h * 0.35)), 0, 0, 360, 255, -1)
    mask_blur = cv2.GaussianBlur(mask, (25, 25), 10) / 255.0
    mask_blur = np.repeat(mask_blur[:, :, np.newaxis], 3, axis=2)

    for i in range(num_frames):
        # Base background motion
        dx = int(2 * np.sin(i * 0.15))
        dy = int(2 * np.cos(i * 0.15))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        bg = cv2.warpAffine(base_face, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Fake face modification inside mask (color temperature shift + high-frequency noise + warping)
        manipulated = bg.copy().astype(np.float32)
        # Deepfake color mismatch: warmer tone and contrast alteration
        manipulated[:, :, 0] *= 0.88   # Blue
        manipulated[:, :, 2] *= 1.12   # Red
        
        # Subtle warp jitter inside facial mask
        jitter_x = int(3 * np.cos(i * 0.4))
        jitter_y = int(3 * np.sin(i * 0.4))
        M_fake = np.float32([[1, 0, jitter_x], [0, 1, jitter_y]])
        manipulated = cv2.warpAffine(manipulated, M_fake, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Blend fake face with background
        blended = (manipulated * mask_blur + bg.astype(np.float32) * (1.0 - mask_blur))
        frame = np.clip(blended, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()


def generate_all_samples():
    """Generates complete suite of real and fake videos plus standalone images."""
    print("=" * 65)
    print(" DeepGuard AI - Generating Demonstration Sample Videos & Images")
    print("=" * 65)

    base_faces = download_base_faces()
    print(f"[*] Base face templates prepared: {len(base_faces)}")

    # Generate 4 Real Videos
    for idx in range(1, 5):
        base = base_faces[(idx - 1) % len(base_faces)]
        # Add slight variation per video
        if idx > 2:
            base = cv2.flip(base, 1)
        out_v = f"ffpp_videos/real/real_video_0{idx}.mp4"
        create_real_video(base, out_v, num_frames=30)
        print(f"[OK] Generated REAL video: {out_v}")

    # Generate 4 Fake Videos
    for idx in range(1, 5):
        base = base_faces[(idx - 1) % len(base_faces)]
        if idx > 2:
            base = cv2.flip(base, 1)
        out_v = f"ffpp_videos/fake/fake_video_0{idx}.mp4"
        create_fake_video(base, out_v, num_frames=30)
        print(f"[OK] Generated FAKE video: {out_v}")

    # Save standalone test images
    sample_real_img = base_faces[0]
    cv2.imwrite("sample_images/real_face_01.jpg", sample_real_img)
    
    # Fake standalone image
    h, w = sample_real_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, int(h * 0.52)), (int(w * 0.28), int(h * 0.35)), 0, 0, 360, 255, -1)
    mask_blur = np.repeat((cv2.GaussianBlur(mask, (25, 25), 10) / 255.0)[:, :, np.newaxis], 3, axis=2)
    fake_img = sample_real_img.copy().astype(np.float32)
    fake_img[:, :, 0] *= 0.85
    fake_img[:, :, 2] *= 1.15
    fake_blended = np.clip(fake_img * mask_blur + sample_real_img.astype(np.float32) * (1.0 - mask_blur), 0, 255).astype(np.uint8)
    cv2.imwrite("sample_images/fake_face_01.jpg", fake_blended)

    # Image with no face
    no_face_img = np.full((400, 400, 3), (60, 60, 60), dtype=np.uint8)
    cv2.putText(no_face_img, "Background Scene", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.imwrite("sample_images/no_face_01.jpg", no_face_img)
    print("[OK] Standalone test images saved to sample_images/")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_samples()
