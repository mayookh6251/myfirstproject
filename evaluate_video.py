"""
DeepGuard AI - Video-Level Inference and Evaluation
===================================================
Classifies entire videos by aggregating predictions across sampled frames.
Supports:
- Uniform frame sampling
- Face detection and bounding box cropping per frame
- Multi-frame probability aggregation (Mean, Median, Majority Voting)
- Uncertainty thresholding:
    p_fake >= 0.65  -> FAKE
    p_fake <= 0.35  -> REAL
    otherwise       -> UNCERTAIN
- Batch evaluation on video test sets with metric reporting
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from PIL import Image

from model_def import load_model_checkpoint
from extract_faces import resolve_cascade_classifier, detect_and_crop_face
from dataset_loader import get_inference_transform


def predict_single_face(
    face_bgr: np.ndarray,
    model: torch.nn.Module,
    transform,
    device: torch.device
) -> Tuple[float, float]:
    """
    Runs model inference on a single cropped face image.
    Returns: (fake_prob: float, real_prob: float)
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Class 0: FAKE, Class 1: REAL
    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    return fake_prob, real_prob


def predict_video(
    video_path: str,
    model: torch.nn.Module,
    classifier: cv2.CascadeClassifier,
    transform,
    device: torch.device,
    sample_frames: int = 20,
    threshold_fake: float = 0.65,
    threshold_real: float = 0.35,
    aggregation: str = "mean"
) -> Dict:
    """
    Analyzes an entire video by sampling frames and aggregating face predictions.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Frame sampling indices
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
        indices = sorted(list(set(indices)))
    else:
        indices = []

    frame_fake_probs = []
    face_crops_preview = []
    analyzed_frames_count = 0
    faces_detected_count = 0

    if total_frames > 0:
        for f_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            analyzed_frames_count += 1
            face_crop = detect_and_crop_face(frame, classifier, margin_ratio=0.15)
            if face_crop is not None:
                faces_detected_count += 1
                p_fake, p_real = predict_single_face(face_crop, model, transform, device)
                frame_fake_probs.append(p_fake)
                if len(face_crops_preview) < 5:
                    face_crops_preview.append(face_crop)
    else:
        # Sequential read fallback
        step = 5
        curr = 0
        while analyzed_frames_count < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if curr % step == 0:
                analyzed_frames_count += 1
                face_crop = detect_and_crop_face(frame, classifier, margin_ratio=0.15)
                if face_crop is not None:
                    faces_detected_count += 1
                    p_fake, _ = predict_single_face(face_crop, model, transform, device)
                    frame_fake_probs.append(p_fake)
            curr += 1

    cap.release()

    if faces_detected_count == 0:
        return {
            "prediction": "NO_FACE",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "frames_analyzed": analyzed_frames_count,
            "faces_detected": 0,
            "frame_probabilities": [],
            "aggregation_method": aggregation,
            "message": "No detectable face found in sampled frames."
        }

    # Aggregate probabilities across faces
    if aggregation == "median":
        agg_fake_prob = float(np.median(frame_fake_probs))
    elif aggregation == "majority":
        votes_fake = sum(1 for p in frame_fake_probs if p >= 0.5)
        agg_fake_prob = votes_fake / len(frame_fake_probs)
    else:  # mean (default)
        agg_fake_prob = float(np.mean(frame_fake_probs))

    agg_real_prob = 1.0 - agg_fake_prob

    # Apply uncertainty thresholding
    if agg_fake_prob >= threshold_fake:
        prediction = "FAKE"
        confidence = agg_fake_prob
    elif agg_fake_prob <= threshold_real:
        prediction = "REAL"
        confidence = agg_real_prob
    else:
        prediction = "UNCERTAIN"
        confidence = max(agg_fake_prob, agg_real_prob)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": agg_fake_prob,
        "real_probability": agg_real_prob,
        "frames_analyzed": analyzed_frames_count,
        "faces_detected": faces_detected_count,
        "frame_probabilities": frame_fake_probs,
        "aggregation_method": aggregation,
        "thresholds": {"fake": threshold_fake, "real": threshold_real}
    }


def evaluate_video_directory(
    videos_dir: str,
    model_path: str = "models/best_resnet18_deepfake.pth",
    sample_frames: int = 20,
    aggregation: str = "mean"
) -> Dict:
    """
    Evaluates a collection of videos in real/ and fake/ subfolders.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" DeepGuard AI - Video-Level Batch Evaluation")
    print("=" * 65)
    print(f"[*] Videos Directory:     {videos_dir}")
    print(f"[*] Model Checkpoint:     {model_path}")
    print(f"[*] Aggregation Strategy: {aggregation}")
    print(f"[*] Sample Frames/Video:  {sample_frames}")

    model = load_model_checkpoint(model_path, device=device)
    classifier = resolve_cascade_classifier()
    transform = get_inference_transform()

    results = []
    y_true = []
    y_pred = []

    for category, label_int in [("fake", 0), ("real", 1)]:
        cat_dir = os.path.join(videos_dir, category)
        if not os.path.exists(cat_dir):
            continue

        v_files = sorted(
            glob.glob(os.path.join(cat_dir, "*.mp4")) +
            glob.glob(os.path.join(cat_dir, "*.avi")) +
            glob.glob(os.path.join(cat_dir, "*.mov"))
        )

        print(f"\n[*] Evaluating [{category.upper()}] videos: {len(v_files)} files.")

        for vf in v_files:
            try:
                res = predict_video(
                    video_path=vf,
                    model=model,
                    classifier=classifier,
                    transform=transform,
                    device=device,
                    sample_frames=sample_frames,
                    aggregation=aggregation
                )
                pred_label = res["prediction"]
                print(f"  {os.path.basename(vf):<30} -> {pred_label:<10} (Conf: {res['confidence']*100:5.1f}%, FakeP: {res['fake_probability']*100:5.1f}%)")
                
                if pred_label in ["FAKE", "REAL"]:
                    y_true.append(label_int)
                    y_pred.append(0 if pred_label == "FAKE" else 1)
                results.append((vf, category, res))
            except Exception as e:
                print(f"  [!] Failed on {os.path.basename(vf)}: {e}")

    if len(y_true) > 0:
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print("\n" + "=" * 65)
        print(f" Video-Level Evaluation Summary:")
        print(f" Evaluated Videos: {len(y_true)}")
        print(f" Video Accuracy:   {acc * 100:.2f}%")
        print("\n Confusion Matrix:")
        print(cm)
        print("\n Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"], zero_division=0))
        print("=" * 65)

    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video deepfake detection.")
    parser.add_argument("--video", type=str, default=None, help="Path to single video for inference")
    parser.add_argument("--videos_dir", type=str, default=None, help="Directory containing real/ and fake/ videos")
    parser.add_argument("--model_path", type=str, default="models/best_resnet18_deepfake.pth", help="Model checkpoint path")
    parser.add_argument("--sample_frames", type=int, default=20, help="Frames to sample per video")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "majority"], help="Aggregation strategy")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.video:
        print(f"[*] Analyzing video: {args.video}")
        m = load_model_checkpoint(args.model_path, device=dev)
        c = resolve_cascade_classifier()
        t = get_inference_transform()
        out = predict_video(args.video, m, c, t, dev, sample_frames=args.sample_frames, aggregation=args.aggregation)
        print("\n" + "=" * 50)
        print(" Prediction Result:")
        for k, v in out.items():
            if k != "frame_probabilities":
                print(f"  {k}: {v}")
        print("=" * 50)
    elif args.videos_dir:
        evaluate_video_directory(args.videos_dir, model_path=args.model_path, sample_frames=args.sample_frames, aggregation=args.aggregation)
    else:
        print("[!] Please specify --video <path> or --videos_dir <directory>.")
