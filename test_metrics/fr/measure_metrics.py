
import os
import cv2
import numpy as np
import argparse
from glob import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def calc_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def main():
    parser = argparse.ArgumentParser(description="Calculate MSE, SSIM, MAE, PSNR for images in two folders.")
    parser.add_argument('--enhanced_dir', type=str, required=True, help='Path to the enhanced images folder')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to the ground truth images folder')
    args = parser.parse_args()

    enhanced_dir = args.enhanced_dir
    gt_dir = args.gt_dir

    # Support multiple extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.BMP', '*.bmp']
    enhanced_files = []
    for ext in extensions:
        enhanced_files.extend(glob(os.path.join(enhanced_dir, ext)))
    
    if not enhanced_files:
        print(f"No images found in {enhanced_dir}")
        return

    avg_mse = 0
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    count = 0

    print(f"Found {len(enhanced_files)} images. Starting evaluation...")

    for enh_path in enhanced_files:
        basename = os.path.basename(enh_path)
        gt_path = os.path.join(gt_dir, basename)

        if not os.path.exists(gt_path):
            # Try to find with different extensions if exact match not found? 
            # For now, strict name matching is safer as per standard datasets.
            print(f"Warning: GT image not found for {basename}. Skipping.")
            continue

        img_enh = read_image(enh_path)
        img_gt = read_image(gt_path)

        # Resize to 256x256
        img_enh = cv2.resize(img_enh, (256, 256))
        img_gt = cv2.resize(img_gt, (256, 256))

        # Calculate metrics
        # MSE and MAE on raw 0-255 values as requested ("MAE和MSE不做归一化处理")
        # Ensure float computation for accuracy
        img_enh_f = img_enh.astype(np.float32)
        img_gt_f = img_gt.astype(np.float32)

        m_mse = mse(img_gt_f, img_enh_f)
        m_mae = calc_mae(img_gt_f, img_enh_f)
        
        # PSNR and SSIM typically expect range or data_range designation. 
        # skimage psnr calculates based on data type range if data_range is not provided.
        # Since we converted to float32, we should specify data_range=255.
        m_psnr = psnr(img_gt, img_enh, data_range=255)
        
        # SSIM
        # multichannel is deprecated in newer versions, channel_axis is the new one.
        # We'll try to be compatible or check version, but simple `channel_axis` argument is common now.
        # If skimage is old, it might use `multichannel=True`.
        # Let's check imports or wrap in try/except for compatibility if needed.
        # For now, assuming relatively modern environment, using channel_axis=-1 for HWC images.
        try:
            m_ssim = ssim(img_gt, img_enh, data_range=255, channel_axis=-1)
        except TypeError:
             # Fallback for older scikit-image versions
            m_ssim = ssim(img_gt, img_enh, data_range=255, multichannel=True)

        avg_mse += m_mse
        avg_mae += m_mae
        avg_psnr += m_psnr
        avg_ssim += m_ssim
        count += 1
        
        # Optional: Print per-image stats? Maybe too verbose.
        # print(f"{basename}: PSNR={m_psnr:.4f}, SSIM={m_ssim:.4f}")

    if count == 0:
        print("No valid image pairs found.")
        return

    avg_mse /= count
    avg_mae /= count
    avg_psnr /= count
    avg_ssim /= count

    print("\nAverage Metrics:")
    print(f"MSE: {avg_mse:.4f}")
    print(f"MAE: {avg_mae:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
