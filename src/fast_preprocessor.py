"""
⚡ LIGHTWEIGHT PREPROCESSING
Minimal image enhancement for speed (200ms)
- Resize optimization
- Smart contrast adjustment
- Fast denoising

Without heavyweight operations (CLAHE, bilateral filter)
"""

import numpy as np
from PIL import Image
import cv2


class FastImagePreprocessor:
    """Ultra-fast preprocessing for OCR"""
    
    def __init__(self, target_dpi=300, max_height=1024):
        self.target_dpi = target_dpi
        self.max_height = max_height
    
    def preprocess(self, image_pil, aggressive=False):
        """
        Fast preprocessing pipeline:
        1. Resize if too large
        2. Contrast enhancement
        3. Optional light denoise
        
        ~150-200ms execution time
        """
        # 1. RESIZE OPTIMIZATION (fastest speed gain)
        img = image_pil.copy()
        h, w = img.height, img.width
        
        if h > self.max_height:
            ratio = self.max_height / h
            new_w = int(w * ratio)
            img = img.resize((new_w, self.max_height), Image.Resampling.LANCZOS)
        
        # 2. CONVERT TO NUMPY
        img_np = np.array(img.convert('L'))  # Grayscale
        
        # 3. FAST CONTRAST ENHANCEMENT
        # Using simple linear stretching (100x faster than CLAHE)
        p2, p98 = np.percentile(img_np, (2, 98))
        img_np = np.clip((img_np - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        
        # 4. OPTIONAL LIGHT DENOISING (if image is very noisy)
        if aggressive and img_np.std() > 80:  # High noise indicator
            # Use Gaussian blur (faster than bilateral)
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
            # Sharpen immediately after
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            img_np = cv2.filter2D(img_np, -1, kernel / 2)
        
        # 5. BACK TO PIL (for model input)
        return Image.fromarray(img_np).convert('RGB')
    
    def preprocess_batch(self, images_pil, aggressive=False):
        """Preprocess multiple images"""
        return [self.preprocess(img, aggressive) for img in images_pil]


# Standalone functions for quick use
def fast_resize(image_pil, max_height=1024):
    """Resize image if too large"""
    h = image_pil.height
    if h > max_height:
        ratio = max_height / h
        new_w = int(image_pil.width * ratio)
        return image_pil.resize((new_w, max_height), Image.Resampling.LANCZOS)
    return image_pil


def fast_contrast(img_np):
    """Ultra-fast contrast enhancement (linear stretching)"""
    p2, p98 = np.percentile(img_np, (2, 98))
    return np.clip((img_np - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)


def fast_denoise_light(img_np):
    """Very light denoising using Gaussian blur + sharpen"""
    blurred = cv2.GaussianBlur(img_np, (3, 3), 0)
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel / 2)
    return np.uint8(sharpened * 0.7 + blurred * 0.3)  # Blend for stability
