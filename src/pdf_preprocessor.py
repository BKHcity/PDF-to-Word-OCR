"""
Advanced PDF Image Enhancement for Vietnamese OCR
Tối ưu hóa ảnh PDF để nhận dạng dấu tiếng Việt chính xác hơn
"""

import cv2
import numpy as np
from PIL import Image
import torch


class PDFImageEnhancer:
    """
    Enhance PDF images for better Vietnamese OCR recognition
    - Increase contrast (dấu rõ hơn)
    - Remove noise
    - Normalize lighting
    - Binarization (if needed)
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def enhance_contrast_adaptive(self, image_array: np.ndarray, clip_limit=2.0, tile_size=8) -> np.ndarray:
        """
        Adaptive histogram equalization to boost diacritics visibility
        diacritics (dấu) là những dòng/điểm nhỏ → CLAHE giúp rõ hơn
        """
        # Convert to uint8 if needed
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(image_array)
        
        if self.debug:
            print(f"  [CLAHE] Applied adaptive contrast enhancement")
        
        return enhanced
    
    def enhance_contrast_global(self, image_array: np.ndarray, alpha=1.5, beta=0) -> np.ndarray:
        """
        Global contrast enhancement: new = alpha * pixel + beta
        Làm dấu rõ hơn bằng cách tăng độ tương phản
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Normalize to [-0.5, 0.5]
        normalized = (image_array / 255.0) - 0.5
        
        # Apply contrast scaling
        enhanced = (normalized * alpha) + 0.5
        
        # Clamp and convert back
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        if self.debug:
            print(f"  [Contrast] Applied global contrast (alpha={alpha})")
        
        return enhanced
    
    def denoise_bilateral(self, image_array: np.ndarray, diameter=9, sigma_color=75, sigma_space=75) -> np.ndarray:
        """
        Bilateral filter to remove noise while preserving edges (dấu)
        Giữ lại dấu nhưng xóa nhiễu
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        denoised = cv2.bilateralFilter(image_array, diameter, sigma_color, sigma_space)
        
        if self.debug:
            print(f"  [Denoise] Applied bilateral filter")
        
        return denoised
    
    def denoise_morphological(self, image_array: np.ndarray, kernel_size=3) -> np.ndarray:
        """
        Morphological operations to remove small noise
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Morphological opening (erode → dilate) removes small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
        
        if self.debug:
            print(f"  [Morpho] Applied morphological opening")
        
        return opened
    
    def sharpen_image(self, image_array: np.ndarray, strength=1.5) -> np.ndarray:
        """
        Unsharp masking to sharpen edges and diacritics
        Làm cho dấu sắc hơn
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image_array, (0, 0), 1.0)
        
        # Unsharp mask: sharpened = image + strength * (image - blurred)
        sharpened = cv2.addWeighted(image_array, 1 + strength, blurred, -strength, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        if self.debug:
            print(f"  [Sharpen] Applied unsharp masking (strength={strength})")
        
        return sharpened
    
    def normalize_lighting(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalize uneven lighting using morphological closing
        Sửa chữ quá sáng/quá tối
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Apply morphological closing to estimate background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        background = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)
        
        # Normalize by dividing by background (avoid division by zero)
        background = np.maximum(background, 30)  # Minimum background value
        normalized = (image_array.astype(float) / background.astype(float) * 200).astype(np.uint8)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        if self.debug:
            print(f"  [Lighting] Applied lighting normalization")
        
        return normalized
    
    def binarize_otsu(self, image_array: np.ndarray) -> np.ndarray:
        """
        Otsu's automatic binarization
        Chuyển thành black/white (tùy chọn)
        """
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        _, binary = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if self.debug:
            print(f"  [Binarize] Applied Otsu's binarization")
        
        return binary
    
    def enhance_full_pipeline(self, image_pil: Image.Image, 
                             apply_clahe=True, 
                             apply_denoise=True,
                             apply_sharpen=True,
                             apply_lighting_norm=True,
                             apply_binarize=False) -> Image.Image:
        """
        Full enhancement pipeline for PDF text recognition
        1. CLAHE → tăng contrast (dấu rõ)
        2. Denoise → xóa nhiễu
        3. Sharpen → làm sắc dấu
        4. Lighting normalize → sửa sáng/tối
        5. Binarize (optional) → black/white
        """
        # Convert to numpy
        if image_pil.mode != 'L':
            image_array = np.array(image_pil.convert('L')).astype(np.float32) / 255.0
        else:
            image_array = np.array(image_pil).astype(np.float32) / 255.0
        
        if self.debug:
            print(f"[Pipeline] Starting enhancement (input shape: {image_array.shape})")
        
        # Step 1: CLAHE for contrast
        if apply_clahe:
            image_array = self.enhance_contrast_adaptive(image_array)
            image_array = image_array.astype(np.float32) / 255.0
        
        # Step 2: Denoise
        if apply_denoise:
            image_array = self.denoise_bilateral(image_array)
            image_array = image_array.astype(np.float32) / 255.0
        
        # Step 3: Sharpen
        if apply_sharpen:
            image_array = self.sharpen_image(image_array, strength=1.0)
            image_array = image_array.astype(np.float32) / 255.0
        
        # Step 4: Lighting normalization
        if apply_lighting_norm:
            image_array = self.normalize_lighting(image_array)
            image_array = image_array.astype(np.float32) / 255.0
        
        # Step 5: Binarize (optional, usually not needed)
        if apply_binarize:
            image_array = self.binarize_otsu(image_array)
            image_array = image_array.astype(np.float32) / 255.0
        
        # Convert back to PIL
        enhanced_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        result_pil = Image.fromarray(enhanced_uint8, mode='L')
        
        if self.debug:
            print(f"[Pipeline] Enhancement complete!")
        
        return result_pil
    
    def enhance_for_crnn(self, image_pil: Image.Image) -> Image.Image:
        """
        Optimized enhancement specifically for CRNN OCR
        - Diacritics visibility: critical
        - Character shape: important
        """
        return self.enhance_full_pipeline(
            image_pil,
            apply_clahe=True,        # Critical for diacritics
            apply_denoise=True,      # Remove PDF artifacts
            apply_sharpen=True,      # Sharpen diacritics
            apply_lighting_norm=True, # Handle uneven lighting
            apply_binarize=False     # Keep grayscale for CRNN
        )


# Test function
if __name__ == "__main__":
    # Test with a sample image
    from PIL import Image
    
    # Create a test image (white with black text)
    test_img = Image.new('L', (200, 100), color=255)
    
    enhancer = PDFImageEnhancer(debug=True)
    enhanced = enhancer.enhance_for_crnn(test_img)
    
    print("\n✅ PDF Image Enhancement Module Ready!")
    print("Usage:")
    print("  enhancer = PDFImageEnhancer(debug=True)")
    print("  enhanced_image = enhancer.enhance_for_crnn(image_pil)")
