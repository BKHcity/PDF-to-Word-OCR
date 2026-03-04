"""
Utility functions for Vietnamese OCR preprocessing
"""

import cv2
import numpy as np
from PIL import Image
import random
import math
import Levenshtein as lev

# ---------- image helpers ----------

def to_gray(img):
    """Convert image to grayscale"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def clahe_equalize(img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def denoise(img):
    """Apply fast NL means denoising"""
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

def robust_deskew(img):
    """Robust skew correction using minAreaRect"""
    gray = img.copy()
    try:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        return img
    bw = 255 - bw
    coords = np.column_stack(np.where(bw > 0))
    if coords.shape[0] < 20:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

def adaptive_binarize(img):
    """Apply adaptive thresholding"""
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,9)

# normalize so ink=1 and values in [0,1]
def normalize_for_model(img):
    """Normalize image for model input (ink=1, background=0)"""
    arr = img.astype(np.float32)
    arr = 255.0 - arr
    arr = arr / 255.0
    return arr

def resize_keep_aspect(img, height=32, max_width=256, interp=cv2.INTER_AREA):
    """Resize image keeping aspect ratio"""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((height, 1), dtype=np.uint8) * 255
    new_h = height
    new_w = max(1, int(w * (new_h / h)))
    if new_w > max_width:
        new_w = max_width
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def pad_width(img, max_width):
    """Pad image width to max_width"""
    h, w = img.shape[:2]
    if w >= max_width:
        return cv2.resize(img, (max_width, h))
    pad = np.ones((h, max_width - w), dtype=img.dtype) * 255
    return np.concatenate([img, pad], axis=1)

# elastic augmentation
def elastic_transform(image, alpha, sigma, random_state=None):
    """Apply elastic deformation for data augmentation"""
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = (random_state.rand(*shape) * 2 - 1)
    dy = (random_state.rand(*shape) * 2 - 1)
    dx = cv2.GaussianBlur(dx, (17,17), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (17,17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

# ------------ metrics ------------

def cer(pred, gt):
    """Character Error Rate"""
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    dist = lev.distance(pred, gt)
    return dist / max(1, len(gt))

def wer(pred, gt):
    """Word Error Rate"""
    ps = pred.split()
    gs = gt.split()
    if len(gs) == 0:
        return 1.0 if len(ps) > 0 else 0.0
    dist = lev.distance(" ".join(ps), " ".join(gs))
    return dist / max(1, len(gs))






