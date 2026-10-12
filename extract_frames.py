"""
DeepGuard AI - Video Frame Extraction
=====================================
Extracts a configurable number of evenly spaced frames from FaceForensics++ videos.
Saves frames organized by category (real / fake) and original video name.
Robust against corrupted or unreadable video files.

Usage:
    python extract_frames.py --frames_per_video 20
"""

import os
import glob
import argparse
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frames_per_video: int = 20
) -> int:
    """
    Extracts evenly spaced frames from a single video file.
    Returns: Number of frames successfully saved.
    """
    if not os.path.exists(video_path):
        print(f"[!] Warning: File does not exist: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Warning: Could not open video (corrupted or unsupported format): {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        # Fallback: Sequential read if CAP_PROP_FRAME_COUNT is invalid
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        
        if not all_frames:
            print(f"[!] Warning: No readable frames found in {video_path}")
            return 0
            
        step = max(1, len(all_frames) // frames_per_video)
        selected_frames = all_frames[::step][:frames_per_video]
        os.makedirs(output_dir, exist_ok=True)
        saved = 0
        for idx, frame in enumerate(selected_frames):
            out_file = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            if cv2.imwrite(out_file, frame):
                saved += 1
        return saved

    # Standard fast seek extraction using linspace
    frame_indices = np.linspace(0, total_frames - 1, min(frames_per_video, total_frames), dtype=int)
    # Deduplicate indices
    frame_indices = sorted(list(set(frame_indices)))

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    for idx, target_frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame_idx))
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            out_file = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            if cv2.imwrite(out_file, frame):
                saved_count += 1

    cap.release()
    return saved_count


def process_video_directory(
    videos_base_dir: str = "ffpp_videos",
    frames_base_dir: str = "frames",
    frames_per_video: int = 20
) -> None:
    """
    Discovers all videos in 'real' and 'fake' subfolders and extracts frames.
    """
    print("=" * 60)
    print(" DeepGuard AI - Video Frame Extraction Pipeline")
    print("=" * 60)
    print(f"[*] Videos base directory:  {videos_base_dir}")
    print(f"[*] Frames output directory: {frames_base_dir}")
    print(f"[*] Target frames/video:     {frames_per_video}")

    categories = ["real", "fake"]
    grand_total_videos = 0
    grand_total_successful = 0
    grand_total_frames = 0

    for category in categories:
        cat_video_dir = os.path.join(videos_base_dir, category)
        cat_frames_dir = os.path.join(frames_base_dir, category)
        
        if not os.path.exists(cat_video_dir):
            os.makedirs(cat_video_dir, exist_ok=True)
            print(f"[i] Created missing video folder: {cat_video_dir}")
            continue

        # Find all videos
        video_files = []
        for ext in SUPPORTED_EXTENSIONS:
            video_files.extend(glob.glob(os.path.join(cat_video_dir, f"*{ext}")))
            video_files.extend(glob.glob(os.path.join(cat_video_dir, f"*{ext.upper()}")))
        video_files = sorted(list(set(video_files)))

        print(f"\n[*] Processing [{category.upper()}] videos: {len(video_files)} found.")
        
        cat_saved_frames = 0
        cat_success_count = 0

        for i, vpath in enumerate(video_files, 1):
            video_stem = os.path.splitext(os.path.basename(vpath))[0]
            video_out_dir = os.path.join(cat_frames_dir, video_stem)
            
            try:
                saved = extract_frames_from_video(vpath, video_out_dir, frames_per_video)
                if saved > 0:
                    cat_saved_frames += saved
                    cat_success_count += 1
                    print(f"  [{i}/{len(video_files)}] {os.path.basename(vpath)} -> {saved} frames extracted")
                else:
                    print(f"  [{i}/{len(video_files)}] {os.path.basename(vpath)} -> FAILED / 0 frames")
            except Exception as e:
                print(f"  [!] Error processing {os.path.basename(vpath)}: {str(e)}")

        grand_total_videos += len(video_files)
        grand_total_successful += cat_success_count
        grand_total_frames += cat_saved_frames

    print("\n" + "=" * 60)
    print(" Extraction Summary:")
    print(f"  Total videos discovered:  {grand_total_videos}")
    print(f"  Videos processed:         {grand_total_successful}")
    print(f"  Total frames extracted:   {grand_total_frames}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from FaceForensics++ videos.")
    parser.add_argument("--videos_dir", type=str, default="ffpp_videos", help="Input videos directory")
    parser.add_argument("--output_dir", type=str, default="frames", help="Output directory for extracted frames")
    parser.add_argument("--frames_per_video", type=int, default=20, help="Frames to extract per video (default: 20)")
    args = parser.parse_args()

    process_video_directory(
        videos_base_dir=args.videos_dir,
        frames_base_dir=args.output_dir,
        frames_per_video=args.frames_per_video
    )
