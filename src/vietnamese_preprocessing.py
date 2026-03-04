"""
Vietnamese Text Preprocessing - Enhanced for Diacritics
Xử lý ảnh nâng cao đặc biệt cho dấu tiếng Việt
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Union, Tuple


class VietnameseImagePreprocessor:
    """
    Advanced image preprocessing specifically for Vietnamese diacritics
    Xử lý ảnh nâng cao cho dấu tiếng Việt
    """
    
    @staticmethod
    def enhance_diacritics(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Enhance diacritics (accents/tone marks) visibility
        Tăng cường độ rõ của dấu tiếng Việt
        
        Args:
            image: Grayscale image (numpy array)
            strength: Enhancement strength (1.0-3.0)
        
        Returns:
            Enhanced image with clearer diacritics
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Apply bilateral filter - preserve edges (important for diacritics)
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. Enhance contrast using adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=strength * 2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # 3. Sharpen image to make diacritics more distinct
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. Apply unsharp masking for fine details
        gaussian = cv2.GaussianBlur(sharpened, (0, 0), 3)
        unsharp = cv2.addWeighted(sharpened, 1.5, gaussian, -0.5, 0)
        
        return unsharp
    
    @staticmethod
    def remove_noise_preserve_diacritics(image: np.ndarray) -> np.ndarray:
        """
        Remove noise while preserving small diacritics marks
        Loại nhiễu nhưng giữ nguyên dấu tiếng Việt
        """
        # Use morphological opening with small kernel to remove noise
        # but preserve small marks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Opening: erosion followed by dilation
        denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Median filter with small kernel to smooth without losing details
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    @staticmethod
    def correct_contrast_for_vietnamese(image: np.ndarray) -> np.ndarray:
        """
        Correct contrast specifically optimized for Vietnamese text
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Find peaks (text and background)
        hist_smooth = cv2.GaussianBlur(hist, (5, 1), 0)
        
        # Adaptive thresholding for better separation
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check if we need to invert
        text_pixels = np.sum(binary == 0)
        bg_pixels = np.sum(binary == 255)
        
        if text_pixels > bg_pixels:
            # Text is black, background is white - correct
            result = binary
        else:
            # Invert
            result = 255 - binary
        
        return result
    
    @staticmethod
    def enhance_for_handwriting(image: Union[Image.Image, np.ndarray],
                                aggressive: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline for Vietnamese handwriting
        Pipeline hoàn chỉnh cho chữ viết tay tiếng Việt
        
        Args:
            image: PIL Image or numpy array
            aggressive: Use more aggressive enhancement (for difficult images)
        
        Returns:
            Preprocessed numpy array ready for OCR
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Denoise while preserving diacritics
        denoised = VietnameseImagePreprocessor.remove_noise_preserve_diacritics(gray)
        
        # Step 2: Enhance diacritics
        strength = 2.0 if aggressive else 1.5
        enhanced = VietnameseImagePreprocessor.enhance_diacritics(denoised, strength)
        
        # Step 3: Correct contrast
        contrasted = VietnameseImagePreprocessor.correct_contrast_for_vietnamese(enhanced)
        
        # Step 4: Final sharpening for diacritics
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        final = cv2.filter2D(contrasted, -1, kernel)
        
        return final
    
    @staticmethod
    def enhance_for_printed_text(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocessing for printed Vietnamese text (from PDFs, scans)
        Xử lý cho chữ in tiếng Việt
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 2: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Adaptive thresholding (better for printed text)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 4: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def auto_rotate(image: np.ndarray) -> np.ndarray:
        """
        Auto-rotate image to correct orientation
        Tự động xoay ảnh về hướng đúng
        """
        # Detect text orientation using projection profile
        h, w = image.shape[:2]
        
        # Try multiple angles
        best_angle = 0
        best_score = 0
        
        for angle in range(-10, 11, 1):
            # Rotate
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate horizontal projection profile
            projection = np.sum(rotated < 128, axis=1)
            
            # Score is variance of projection (higher = more aligned)
            score = np.var(projection)
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        # Apply best rotation
        if best_angle != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            result = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
            return result
        
        return image
    
    @staticmethod
    def remove_borders(image: np.ndarray, threshold: int = 240) -> np.ndarray:
        """
        Remove white borders around text
        Loại bỏ viền trắng xung quanh text
        """
        # Find bounding box of non-white pixels
        mask = image < threshold
        coords = cv2.findNonZero(mask.astype(np.uint8))
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add small padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return image[y:y+h, x:x+w]
        
        return image
    
    @staticmethod
    def process_for_ocr(image: Union[Image.Image, np.ndarray],
                       image_type: str = 'auto',
                       aggressive: bool = False) -> np.ndarray:
        """
        Main preprocessing function - auto-detect and process
        
        Args:
            image: Input image
            image_type: 'handwritten', 'printed', or 'auto'
            aggressive: Use aggressive enhancement for difficult images
        
        Returns:
            Preprocessed image ready for OCR
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Ensure grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Auto-detect image type if needed
        if image_type == 'auto':
            # Simple heuristic: check edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Handwriting usually has higher edge density
            if edge_density > 0.1:
                image_type = 'handwritten'
            else:
                image_type = 'printed'
            
            print(f"📊 Auto-detected: {image_type} (edge density: {edge_density:.3f})")
        
        # Apply appropriate preprocessing
        if image_type == 'handwritten':
            processed = VietnameseImagePreprocessor.enhance_for_handwriting(
                gray, aggressive=aggressive
            )
        else:
            processed = VietnameseImagePreprocessor.enhance_for_printed_text(gray)
        
        # Common post-processing
        processed = VietnameseImagePreprocessor.auto_rotate(processed)
        processed = VietnameseImagePreprocessor.remove_borders(processed)
        
        return processed


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def preprocess_image(image: Union[Image.Image, np.ndarray, str],
                    image_type: str = 'auto',
                    aggressive: bool = False) -> np.ndarray:
    """
    Quick preprocessing function
    
    Usage:
        processed = preprocess_image("image.png")
        processed = preprocess_image(pil_image, image_type='handwritten')
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image)
    
    preprocessor = VietnameseImagePreprocessor()
    return preprocessor.process_for_ocr(image, image_type, aggressive)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print("🖼️  Vietnamese Image Preprocessing - Demo")
    
    test_image = "test_image.png"
    
    if Path(test_image).exists():
        print(f"\n📸 Loading {test_image}...")
        
        # Load image
        img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
        
        # Process
        preprocessor = VietnameseImagePreprocessor()
        
        # Try both modes
        handwritten = preprocessor.enhance_for_handwriting(img, aggressive=False)
        printed = preprocessor.enhance_for_printed_text(img)
        
        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(handwritten, cmap='gray')
        axes[1].set_title('Handwritten Mode')
        axes[1].axis('off')
        
        axes[2].imshow(printed, cmap='gray')
        axes[2].set_title('Printed Mode')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('preprocessing_demo.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved preprocessing_demo.png")
        
    else:
        print(f"⚠️  {test_image} not found")




