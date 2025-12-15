
import argparse
import os
import cv2
import torch
import pyiqa
import glob
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Calculate Perceptual Score (PS)")
    parser.add_argument('--enhanced_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, default=None, help="Optional GT for Reference-based PS (LPIPS)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Metrics
    # If GT is present, use LPIPS (standard Perceptual Score in FR)
    # If GT is NOT present, use MUSIQ as a proxy for Perceptual Quality? 
    # Or maybe the user implies 'PS' is a specific metric name in pyiqa?
    # No, usually PS = Perceptual Score.
    # We will compute LPIPS if GT exists.
    # We will compute MUSIQ if GT does not exist (or both if user wants).
    # Since the user said "PS is usable for both", likely they mean "The script calculates PS".
    
    lpips_metric = None
    nr_metric = None
    
    if args.gt_dir:
        print("GT directory provided. Initializing LPIPS (FR)...")
        try:
            lpips_metric = pyiqa.create_metric('lpips', device=device)
        except Exception as e:
            print(f"Failed to load LPIPS: {e}")

    print("Initializing NR Perceptual Metric (MUSIQ) as fallback/supplement...")
    try:
        nr_metric = pyiqa.create_metric('musiq', device=device)
    except Exception as e:
        print(f"Failed to load MUSIQ: {e}")

    enhanced_files = sorted(glob.glob(os.path.join(args.enhanced_dir, '*')))
    enhanced_files = [f for f in enhanced_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not enhanced_files:
        print("No images found.")
        return

    print(f"Processing {len(enhanced_files)} images...")
    
    avg_fr_ps = 0
    avg_nr_ps = 0
    count = 0
    fr_count = 0

    for enh_path in enhanced_files:
        # Resize to 256x256 as requested
        img = cv2.imread(enh_path)
        if img is None: continue
        img = cv2.resize(img, (256, 256))
        
        # PyIQA expects path or tensor. Since we resized, we must pass tensor or save temp.
        # Passing tensor (HWC -> CHW, 0-1 RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # NR PS
        val_nr = 0
        if nr_metric:
            val_nr = nr_metric(img_tensor).item()
            avg_nr_ps += val_nr

        # FR PS
        val_fr = -1
        if lpips_metric and args.gt_dir:
            basename = os.path.basename(enh_path)
            gt_path = os.path.join(args.gt_dir, basename)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                if gt_img is not None:
                    gt_img = cv2.resize(gt_img, (256, 256))
                    gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                    gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0
                    gt_tensor = gt_tensor.unsqueeze(0).to(device)
                    
                    val_fr = lpips_metric(img_tensor, gt_tensor).item()
                    avg_fr_ps += val_fr
                    fr_count += 1

        # Output
        fr_str = f"LPIPS={val_fr:.4f}" if val_fr != -1 else "LPIPS=N/A"
        nr_str = f"MUSIQ={val_nr:.4f}" if nr_metric else "MUSIQ=N/A"
        print(f"{os.path.basename(enh_path)}: {fr_str}, {nr_str}")
        count += 1

    print("\nAverage Perceptual Scores:")
    if nr_metric:
        print(f"NR PS (MUSIQ): {avg_nr_ps/count:.4f}")
    if lpips_metric and fr_count > 0:
        print(f"FR PS (LPIPS): {avg_fr_ps/fr_count:.4f}")

if __name__ == "__main__":
    main()
