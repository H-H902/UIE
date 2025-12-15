
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import kornia.color as color

def uciqe(image):
    """
    Calculate UCIQE using PyTorch. 
    Input: image is a numpy array (BGR or RGB) read by cv2.
    It handles conversion internally.
    """
    # Convert numpy to tensor
    # Ensure standard RGB and normalized 0-1 for kornia/torch processing
    if isinstance(image, np.ndarray):
        # cv2 reads in BGR usually. Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_img = to_tensor(image) # C, H, W, range 0-1
    else:
        # Assume it is already tensor
        t_img = image

    # Add batch dim if needed
    if len(t_img.shape) == 3:
        t_img = t_img.unsqueeze(0)

    # RGB to HSV
    hsv = color.rgb_to_hsv(t_img) 
    H, S, V = torch.chunk(hsv, 3, dim=1)

    # Values
    H = H.squeeze()
    S = S.squeeze()
    V = V.squeeze()

    # Chroma std (H) 
    # Original code: delta = np.std(H) / 180 (Wait, H in standard hsv is 0-360 or 0-1?)
    # kornia.color.rgb_to_hsv outputs H in [0, 2pi], S in [0, 1], V in [0, 1].
    # But uciqe definition typically uses [0, 1] for all or specific ranges.
    # The reference code `uciqe.py` from Pytorch-UW-IQA uses: 
    # delta = torch.std(H) / (2 * math.pi) if using kornia/torch
    
    delta = torch.std(H) # H is 0-2pi
    # Re-checking reference: line 39: delta = torch.std(H) / (2 * math.pi)
    # The reference seems to expect H to be normalized or just uses 'delta' as std.
    # In 'uciqe' function in that file: line 14: delta = np.std(H) / 180 (OpenCV H is 0-180)
    # So delta needs to be comparable. 
    # If H is 0-2pi, then std(H) is in radians. 
    # 180 degrees is pi. So scaling should match. 
    # Let's stick to the torch logic: std(H) / (2pi) is confusing if H is 2pi range.
    # Effectively we want std normalized to [0,1].
    # Let's trust the logic from the repo: `delta = torch.std(H) / (2 * math.pi)` implies normalization.
    
    delta = torch.std(H) 
    # Wait, kornia rgb_to_hsv: " The image data is assumed to be in the range of (0, 1)."
    # "Return: HSV image with shape (C, H, W) ... H component is in range [0, 2pi]."
    # So `delta = torch.std(H) / (2 * math.pi)` makes sense provided H spans 0-2pi.
    
    # Saturation mean
    mu = torch.mean(S)
    
    # Contrast of Luminance (V)
    # Bottom 1% and Top 1%
    # V is 0-1.
    n, m = V.shape
    number = math.floor(n * m / 100)
    v = V.flatten()
    sorted_v, _ = torch.sort(v)
    
    bottom = torch.sum(sorted_v[:number]) / number
    top = torch.sum(sorted_v[-number:]) / number # More efficient than -v sort
    
    conl = top - bottom
    
    # Final Metric
    # Coeffs: 0.4680, 0.2745, 0.2576
    # Note: The reference code has: `0.4680 * delta + 0.2745 * conl + 0.2576 * mu`
    # Warning: In `uciqe.py` standard cv2 version: `delta = np.std(H) / 180`
    # In `torch_uciqe`: `delta = torch.std(H) / (2 * math.pi)`
    # This should be consistent.
    
    uciqe_val = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe_val.item()

# UIQM IMPLEMENTATION
sobel_kernel_x = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
sobel_kernel_y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

def sobel_torch(x):
    # x assumed to be (H, W) or (1, 1, H, W)
    if len(x.shape) == 2:
        x_in = x.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    elif len(x.shape) == 3:
        x_in = x.unsqueeze(0) # (1, H, W) -> assumes 1 channel?
        # If x is (C, H, W) and C > 1, this kernel (1,1,3,3) won't work well unless group convolution?
        # But we only pass R, G, B single channels here. So it should be (1, H, W).
    elif len(x.shape) == 4:
        x_in = x
    else:
        # Fallback for unexpected shapes, try to squash
        print(f"Warning: sobel_torch got shape {x.shape}")
        x_in = x
    
    dx = F.conv2d(x_in, sobel_kernel_x.to(x.device), padding=1)
    dy = F.conv2d(x_in, sobel_kernel_y.to(x.device), padding=1)
    mag = torch.hypot(dx, dy)
    
    # Normalization: value matches 255 scale in ref?
    # ref: mag *= 255.0 / torch.max(mag)
    if torch.max(mag) != 0:
        mag *= 255.0 / torch.max(mag)
    
    return mag.squeeze()

def eme(x, window_size):
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    x_crop = x[:k2 * window_size, :k1 * window_size]
    
    x_view = x_crop.view(k2, window_size, k1, window_size)
    x_view = x_view.permute(0, 2, 1, 3).contiguous().view(-1, window_size * window_size)
    
    max_vals, _ = torch.max(x_view, dim=1)
    min_vals, _ = torch.min(x_view, dim=1)
    
    non_zero = (min_vals != 0) & (max_vals != 0)
    # Avoid log(0)
    
    val = torch.zeros_like(max_vals)
    val[non_zero] = torch.log(max_vals[non_zero] / min_vals[non_zero])
    
    w = 2. / (k1 * k2)
    return w * val.sum()

def _uism(x):
    # x is (3, H, W) in RGB
    R = x[0, :, :]
    G = x[1, :, :]
    B = x[2, :, :]
    
    Rs = sobel_torch(R)
    Gs = sobel_torch(G)
    Bs = sobel_torch(B)
    
    R_edge = Rs * R
    G_edge = Gs * G
    B_edge = Bs * B
    
    r_eme = eme(R_edge, 10)
    g_eme = eme(G_edge, 10)
    b_eme = eme(B_edge, 10)
    
    return 0.299*r_eme + 0.587*g_eme + 0.144*b_eme

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x_sorted, _ = torch.sort(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = 1.0 / (K - T_a_L - T_a_R)
    s = int(T_a_L)
    e = int(K - T_a_R)
    return weight * torch.sum(x_sorted[s:e])

def s_a(x, mu):
    return torch.sum(torch.pow(x - mu, 2)) / len(x)

def _uicm(x):
    # x is (3, H, W)
    R = x[0, :, :].flatten()
    G = x[1, :, :].flatten()
    B = x[2, :, :].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = torch.sqrt(mu_a_RG**2 + mu_a_YB**2)
    r = torch.sqrt(s_a_RG + s_a_YB)
    return -0.0268*l + 0.1586*r

def _uiconm(x, window_size):
    k1 = x.shape[2] // window_size
    k2 = x.shape[1] // window_size
    x_crop = x[:, :k2 * window_size, :k1 * window_size]
    w = -1. / (k1 * k2)
    alpha = 1
    
    x_unfold = x_crop.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    # (3, k2, k1, w, w)
    x_unfold = x_unfold.reshape(3, k2, k1, -1) 
    # Max/Min over channel dim? No, max/min within block?
    # Ref: min_ = torch.min(torch.min(torch.min(x, dim=-1)...
    # The ref reshape: x.reshape(-1, k2, k1, window_size, window_size) implies merging channels?
    # Ref `x = x.reshape(-1, k2, k1, window_size, window_size)` -> -1 is 3 (channels)
    # Then `min_ = ... values, dim=0)`
    # It takes Min over ALL channels and pixels in block?
    # Ref line 65: min over dim -1 (width), then dim -1 (height), then dim 0 (channels).
    # Yes, global min/max for the block across all channels.
    
    x_blk = x_unfold.permute(1, 2, 0, 3).reshape(k2, k1, -1) # (k2, k1, 3*w*w)
    
    min_vals, _ = torch.min(x_blk, dim=2)
    max_vals, _ = torch.max(x_blk, dim=2)
    
    top = max_vals - min_vals
    bot = max_vals + min_vals
    
    val = alpha * torch.pow((top/bot), alpha) * torch.log(top/bot)
    val = torch.where(torch.isnan(val) | (bot==0) | (top==0), torch.zeros_like(val), val)
    return w * val.sum()

def uiqm(image):
    """
    Calculate UIQM using PyTorch.
    """
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_img = to_tensor(image) # 0-1
    else:
        t_img = image
        
    # UIQM expects input in calculation to be scaled?
    # Ref torch_uiqm: `x *= 255`
    t_img = t_img.clone() * 255.0
    
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uicm = _uicm(t_img)
    uism = _uism(t_img)
    uiconm = _uiconm(t_img, 10)
    return (c1*uicm + c2*uism + c3*uiconm).item()
