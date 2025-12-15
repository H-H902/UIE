
import os
import argparse
import cv2
import glob
import numpy as np
import torch
import pyiqa
from nr_utils import uciqe, uiqm
from skimage.metrics import peak_signal_noise_ratio as psnr

def main():
    parser = argparse.ArgumentParser(description="Calculate UCIQ, UIQM, NIQE, PS (PSNR), MUSIQ")
    parser.add_argument('--enhanced_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, default=None, help="Optional GT dir for PS(PSNR)")
    args = parser.parse_args()

    enhanced_files = sorted(glob.glob(os.path.join(args.enhanced_dir, '*')))
    # Filter for images
    enhanced_files = [f for f in enhanced_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not enhanced_files:
        print("No images found in enhanced_dir")
        return

    # Initialize PyIQA metrics
    # niqe = pyiqa.create_metric('niqe', device=torch.device('cpu')) 
    # musiq = pyiqa.create_metric('musiq', device=torch.device('cpu'))
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        niqe_metric = pyiqa.create_metric('niqe', device=device)
        musiq_metric = pyiqa.create_metric('musiq', device=device)
    except Exception as e:
        print(f"Error loading pyiqa models: {e}")
        return

    avg_uciqe = 0
    avg_uiqm = 0
    avg_niqe = 0
    avg_musiq = 0
    avg_ps = 0
    count = 0
    ps_count = 0

    print(f"Processing {len(enhanced_files)} images...")

    for enh_path in enhanced_files:
        img = cv2.imread(enh_path)
        if img is None:
            continue
            
        # UIQM and UCIQE
        try:
            val_uciqe = uciqe(img)
            val_uiqm = uiqm(img)
        except Exception as e:
            print(f"Error calculating UCIQE/UIQM for {enh_path}: {e}")
            val_uciqe = 0
            val_uiqm = 0

        # NIQE and MUSIQ (PyIQA expects path or tensor)
        # Pass path directly to avoid conversion issues if supported
        try:
            val_niqe = niqe_metric(enh_path).item()
            val_musiq = musiq_metric(enh_path).item()
        except Exception as e:
            print(f"Error pyiqa for {enh_path}: {e}")
            val_niqe = 0
            val_musiq = 0

        # PS (PSNR)
        val_ps = 0
        if args.gt_dir:
            basename = os.path.basename(enh_path)
            gt_path = os.path.join(args.gt_dir, basename)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                if gt_img is not None:
                    # Resize to match if needed, though usually mapped
                    h, w, c = img.shape
                    gt_img = cv2.resize(gt_img, (w, h))
                    val_ps = psnr(gt_img, img, data_range=255)
                    avg_ps += val_ps
                    ps_count += 1
        
        avg_uciqe += val_uciqe
        avg_uiqm += val_uiqm
        avg_niqe += val_niqe
        avg_musiq += val_musiq
        count += 1

        print(f"{os.path.basename(enh_path)}: UCIQE={val_uciqe:.4f}, UIQM={val_uiqm:.4f}, NIQE={val_niqe:.4f}, MUSIQ={val_musiq:.4f}, PS={val_ps:.4f}")

    if count > 0:
        print("\nAverage Metrics:")
        print(f"UCIQE: {avg_uciqe/count:.4f}")
        print(f"UIQM: {avg_uiqm/count:.4f}")
        print(f"NIQE: {avg_niqe/count:.4f}")
        print(f"MUSIQ: {avg_musiq/count:.4f}")
        if args.gt_dir and ps_count > 0:
            print(f"PS (PSNR): {avg_ps/ps_count:.4f}")
        else:
            print(f"PS (PSNR): N/A (No GT provided or found)")

if __name__ == "__main__":
    main()

