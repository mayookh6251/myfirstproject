"""
DeepGuard AI - Video-Level Dataset Preparation
==============================================
Partitions face crops into Train (70%), Validation (15%), and Test (15%) sets.
CRITICAL INTEGRITY PRINCIPLE:
Data is partitioned strictly by ORIGINAL VIDEO ID, guaranteeing that frames
from the same video NEVER appear across both training and evaluation splits (Zero Data Leakage).

Output structure:
    dataset/
      ├── train/ (real / fake)
      ├── val/   (real / fake)
      └── test/  (real / fake)
"""

import os
import glob
import shutil
import random
import json
import argparse
from typing import Dict, List, Tuple


def get_video_subdirs(category_faces_dir: str) -> List[str]:
    """Returns sorted list of video folder names inside category faces directory."""
    if not os.path.exists(category_faces_dir):
        return []
    return sorted([
        d for d in os.listdir(category_faces_dir)
        if os.path.isdir(os.path.join(category_faces_dir, d))
    ])


def partition_video_list(
    video_list: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits video IDs into train, val, and test subsets with zero intersection.
    """
    vids = list(video_list)
    rng = random.Random(seed)
    rng.shuffle(vids)

    n = len(vids)
    if n == 0:
        return [], [], []
    if n == 1:
        return vids, [], []
    if n == 2:
        return [vids[0]], [vids[1]], []

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    
    # Adjust to ensure all sets get elements if possible
    if n >= 3 and n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
        
    train_vids = vids[:n_train]
    val_vids = vids[n_train:n_train + n_val]
    test_vids = vids[n_train + n_val:]

    # Assert zero video leakage
    s_train, s_val, s_test = set(train_vids), set(val_vids), set(test_vids)
    assert len(s_train & s_val) == 0, "Data leakage detected: train and val overlap!"
    assert len(s_train & s_test) == 0, "Data leakage detected: train and test overlap!"
    assert len(s_val & s_test) == 0, "Data leakage detected: val and test overlap!"

    return train_vids, val_vids, test_vids


def copy_faces(
    source_faces_dir: str,
    target_category_dir: str,
    video_ids: List[str]
) -> int:
    """Copies all face crops from the specified video folders into target folder."""
    os.makedirs(target_category_dir, exist_ok=True)
    copied_count = 0

    for vid in video_ids:
        vid_dir = os.path.join(source_faces_dir, vid)
        if not os.path.exists(vid_dir):
            continue
            
        face_files = glob.glob(os.path.join(vid_dir, "*.jpg")) + glob.glob(os.path.join(vid_dir, "*.png"))
        for fpath in face_files:
            fname = f"{vid}_{os.path.basename(fpath)}"
            dst = os.path.join(target_category_dir, fname)
            shutil.copyfile(fpath, dst)
            copied_count += 1

    return copied_count


def prepare_dataset_splits(
    faces_base_dir: str = "faces",
    dataset_base_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    clean_existing: bool = True
) -> Dict:
    """
    Executes the video-level dataset splitting pipeline.
    """
    print("=" * 65)
    print(" DeepGuard AI - Video-Partitioned Dataset Preparation")
    print("=" * 65)
    print(f"[*] Faces source directory:   {faces_base_dir}")
    print(f"[*] Dataset target directory: {dataset_base_dir}")
    print(f"[*] Split ratios:             Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {(1-train_ratio-val_ratio)*100:.0f}%")
    print(f"[*] Random seed:              {seed}")

    real_vids = get_video_subdirs(os.path.join(faces_base_dir, "real"))
    fake_vids = get_video_subdirs(os.path.join(faces_base_dir, "fake"))

    print(f"\n[*] Total source videos found: {len(real_vids)} REAL, {len(fake_vids)} FAKE")

    if len(real_vids) == 0 and len(fake_vids) == 0:
        print("[!] No face folders found in source faces directory.")
        print("    Please run extract_frames.py and extract_faces.py first.")
        return {}

    # Partition video IDs
    train_real_vids, val_real_vids, test_real_vids = partition_video_list(real_vids, train_ratio, val_ratio, seed)
    train_fake_vids, val_fake_vids, test_fake_vids = partition_video_list(fake_vids, train_ratio, val_ratio, seed)

    splits = {
        "train": {"real_vids": train_real_vids, "fake_vids": train_fake_vids},
        "val":   {"real_vids": val_real_vids,   "fake_vids": val_fake_vids},
        "test":  {"real_vids": test_real_vids,  "fake_vids": test_fake_vids},
    }

    # Reset dataset directory structure if requested
    if clean_existing and os.path.exists(dataset_base_dir):
        for split in ["train", "val", "test"]:
            for cat in ["real", "fake"]:
                p = os.path.join(dataset_base_dir, split, cat)
                if os.path.exists(p):
                    shutil.rmtree(p)
                os.makedirs(p, exist_ok=True)

    summary_stats = {}

    for split, data in splits.items():
        real_dst = os.path.join(dataset_base_dir, split, "real")
        fake_dst = os.path.join(dataset_base_dir, split, "fake")

        num_real_imgs = copy_faces(os.path.join(faces_base_dir, "real"), real_dst, data["real_vids"])
        num_fake_imgs = copy_faces(os.path.join(faces_base_dir, "fake"), fake_dst, data["fake_vids"])

        summary_stats[split] = {
            "real_videos": len(data["real_vids"]),
            "fake_videos": len(data["fake_vids"]),
            "real_images": num_real_imgs,
            "fake_images": num_fake_imgs,
            "total_images": num_real_imgs + num_fake_imgs
        }

    # Save manifest for scientific reproducibility
    manifest_path = os.path.join(dataset_base_dir, "splits_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "seed": seed,
            "splits": splits,
            "summary": summary_stats
        }, f, indent=2)

    # Print clean summary table
    print("\n" + "=" * 65)
    print(f"{'Split':<10} | {'Real Vids':<10} | {'Fake Vids':<10} | {'Real Imgs':<10} | {'Fake Imgs':<10} | {'Total Imgs':<10}")
    print("-" * 65)
    for split, st in summary_stats.items():
        print(f"{split:<10} | {st['real_videos']:<10} | {st['fake_videos']:<10} | {st['real_images']:<10} | {st['fake_images']:<10} | {st['total_images']:<10}")
    print("=" * 65)
    print(f"[OK] Manifest saved to: {manifest_path}")
    print("[OK] Dataset preparation complete with ZERO video leakage.")

    return summary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition face crops into train/val/test by video.")
    parser.add_argument("--faces_dir", type=str, default="faces", help="Source faces directory")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Target dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    prepare_dataset_splits(
        faces_base_dir=args.faces_dir,
        dataset_base_dir=args.dataset_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
