"""
Image Enhancement Module for Vietnamese OCR
Handles blurry, shaky, low-contrast, and poorly lit images
"""

import cv2
import numpy as np
from typing import Tuple

class ImageEnhancer:
    """Advanced image enhancement for OCR preprocessing"""

    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive image enhancement pipeline

        Args:
            image: Grayscale image array

        Returns:
            Enhanced image array
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        enhanced = image.copy()

        # Step 1: Noise reduction
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Step 2: Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)

        # Step 3: Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Step 4: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # Step 5: Adaptive thresholding for better binarization
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return enhanced

    @staticmethod
    def enhance_blurry_image(image: np.ndarray) -> np.ndarray:
        """Special handling for blurry images"""
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        # Apply bilateral filter to reduce noise while keeping edges
        enhanced = cv2.bilateralFilter(unsharp_mask, 9, 75, 75)

        return enhanced

    @staticmethod
    def enhance_low_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance low contrast images"""
        # Apply CLAHE with higher clip limit
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)

        # Stretch contrast
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        return enhanced

    @staticmethod
    def detect_image_quality(image: np.ndarray) -> dict:
        """Analyze image quality metrics"""
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

        # Contrast measurement
        contrast = image.std()

        # Brightness
        brightness = np.mean(image)

        return {
            'blur_score': laplacian_var,
            'contrast': contrast,
            'brightness': brightness,
            'is_blurry': laplacian_var < 100,
            'is_low_contrast': contrast < 30,
            'is_dark': brightness < 100,
            'is_bright': brightness > 200
        }






