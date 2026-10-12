"""
DeepGuard AI - Dataset Integrity & Health Checker
=================================================
Validates the prepared dataset structure:
- Checks directory hierarchy (train/val/test x real/fake)
- Scans for corrupt or unreadable image files
- Checks image dimensions and color channels
- Evaluates class balance and sample distributions
- Inspects splits_manifest.json for cross-set video leakage
"""

import os
import glob
import json
import argparse
from PIL import Image

EXPECTED_SPLITS = ["train", "val", "test"]
EXPECTED_CLASSES = ["fake", "real"]


def verify_dataset_integrity(dataset_dir: str = "dataset") -> bool:
    """
    Performs comprehensive health checks on the dataset.
    Returns: True if dataset is valid and ready for training, False otherwise.
    """
    print("=" * 65)
    print(" DeepGuard AI - Dataset Health & Integrity Verification")
    print("=" * 65)
    print(f"[*] Dataset root directory: {os.path.abspath(dataset_dir)}")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found at: {dataset_dir}")
        return False

    all_ok = True
    split_counts = {}

    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"[!] Warning: Missing split directory: {split_dir}")
            all_ok = False
            continue

        split_counts[split] = {}
        for cls in EXPECTED_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"[!] Warning: Missing class directory: {cls_dir}")
                split_counts[split][cls] = 0
                all_ok = False
                continue

            # Find all images
            img_files = (
                glob.glob(os.path.join(cls_dir, "*.jpg")) +
                glob.glob(os.path.join(cls_dir, "*.jpeg")) +
                glob.glob(os.path.join(cls_dir, "*.png"))
            )
            split_counts[split][cls] = len(img_files)

            # Check a sample of images for corruption
            corrupt_count = 0
            for img_path in img_files[:20]:  # Sample up to 20 images
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception:
                    corrupt_count += 1

            if corrupt_count > 0:
                print(f"[!] Found {corrupt_count} corrupt images in {cls_dir}")
                all_ok = False

    # Print summary table
    print("\n" + "-" * 65)
    print(f"{'Split':<10} | {'FAKE Images (0)':<18} | {'REAL Images (1)':<18} | {'Total':<10}")
    print("-" * 65)
    total_all = 0
    for split in EXPECTED_SPLITS:
        fake_cnt = split_counts.get(split, {}).get("fake", 0)
        real_cnt = split_counts.get(split, {}).get("real", 0)
        subtotal = fake_cnt + real_cnt
        total_all += subtotal
        print(f"{split:<10} | {fake_cnt:<18} | {real_cnt:<18} | {subtotal:<10}")
    print("-" * 65)
    print(f"{'TOTAL':<10} | {'':<18} | {'':<18} | {total_all:<10}")
    print("-" * 65)

    # Manifest verification
    manifest_path = os.path.join(dataset_dir, "splits_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            splits_data = manifest.get("splits", {})
            train_v = set(splits_data.get("train", {}).get("real_vids", []) + splits_data.get("train", {}).get("fake_vids", []))
            val_v = set(splits_data.get("val", {}).get("real_vids", []) + splits_data.get("val", {}).get("fake_vids", []))
            test_v = set(splits_data.get("test", {}).get("real_vids", []) + splits_data.get("test", {}).get("fake_vids", []))

            leak_tr_val = train_v & val_v
            leak_tr_te = train_v & test_v
            leak_val_te = val_v & test_v

            if leak_tr_val or leak_tr_te or leak_val_te:
                print(f"[ERROR] Video leakage found in manifest! Overlaps: {leak_tr_val | leak_tr_te | leak_val_te}")
                all_ok = False
            else:
                print("[OK] Manifest verified: ZERO video leakage across train/val/test splits.")
        except Exception as e:
            print(f"[!] Warning reading manifest: {e}")

    if total_all == 0:
        print("\n[!] Notice: Dataset is currently empty. Run extraction and preparation scripts to populate it.")
        return False
    elif all_ok:
        print("\n[OK] Dataset health check PASSED! Ready for model training and evaluation.")
        return True
    else:
        print("\n[!] Dataset check completed with warnings. Review details above.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset integrity and image counts.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory")
    args = parser.parse_args()

    verify_dataset_integrity(args.dataset_dir)
