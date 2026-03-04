"""
Image Preprocessing Module for Vietnamese OCR
Xử lý ảnh để cải thiện chất lượng OCR
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Union, Tuple


class ImagePreprocessor:
    """Xử lý ảnh để tối ưu hóa cho OCR"""
    
    @staticmethod
    def to_cv2(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    
    @staticmethod
    def to_pil(image: np.ndarray) -> Image.Image:
        """Convert OpenCV format to PIL Image"""
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def auto_deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """
        Tự động chỉnh góc nghiêng (deskew) ảnh
        Hữu ích khi chụp ảnh bị nghiêng
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough transform để detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate median angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) <= max_angle:
                angles.append(angle)
        
        if not angles:
            return image
        
        angle = np.median(angles)
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Remove noise từ ảnh
        strength: 1-20 (cao hơn = loại bỏ nhiều noise nhưng có thể làm mờ chi tiết)
        """
        return cv2.fastNlMeansDenoising(
            image, 
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    @staticmethod
    def increase_contrast(image: np.ndarray, alpha: float = 1.5, beta: float = 0) -> np.ndarray:
        """
        Tăng độ tương phản (Contrast)
        alpha > 1: Tăng contrast
        beta: Brightness adjustment
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def enhance_light_text(image: np.ndarray) -> np.ndarray:
        """
        Enhance ảnh có chữ màu nhạt (xanh nhạt, xám nhạt, etc.)
        Đặc biệt tốt cho chữ viết tay màu nhạt trên nền trắng/kem
        
        Strategy: Find the text color and maximize contrast
        """
        # Convert to float for better precision
        img_float = image.astype(np.float32) / 255.0
        
        # Get grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find the darkest pixels (likely text)
        # Use percentile to find text color threshold
        dark_threshold = np.percentile(gray, 10)  # Darkest 10%
        light_threshold = np.percentile(gray, 90)  # Lightest 90%
        
        # The difference between text and background
        contrast_range = light_threshold - dark_threshold
        
        if contrast_range < 30:
            # Very low contrast - need aggressive enhancement
            # Method: Stretch the histogram to full range
            gray_stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Invert so text becomes dark
            # Find if text is darker or lighter than background
            # (for light green text on cream, text is slightly darker)
            mean_val = np.mean(gray)
            
            # Apply strong CLAHE
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray_stretched)
            
            # Another round of normalization
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
            # Invert to get dark text
            enhanced = 255 - enhanced
            
            # Apply threshold to clean up
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # If mostly white (text was inverted wrong), invert again
            if np.mean(binary) > 200:
                binary = 255 - binary
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Make sure text is black on white
            if np.mean(binary) < 128:
                binary = 255 - binary
            
            result = binary
        else:
            # Normal contrast - just enhance
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            result = clahe.apply(gray)
            result = 255 - result  # Invert
            _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(result) < 128:
                result = 255 - result
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def detect_light_text(image: np.ndarray) -> bool:
        """
        Detect if image has light-colored text on white/light background
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate mean and std
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Check saturation (colored text)
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)
        
        # Light text on light background: 
        # - high gray mean (bright overall)
        # - low gray std (low contrast in grayscale)
        # - some saturation (colored text like green, blue)
        is_light_bg = mean_val > 180
        is_low_contrast = std_val < 60  # Increased threshold
        has_color = mean_sat > 10  # Has some color
        
        return is_light_bg and is_low_contrast and has_color
    
    @staticmethod
    def auto_threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Convert to binary image (sử dụng threshold)
        method: 'otsu', 'adaptive', 'simple'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        else:  # simple
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def upscale(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Phóng to ảnh để tăng độ chi tiết (tốt cho text nhỏ)
        scale: 1.5, 2.0, 3.0, etc.
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use INTER_CUBIC for better quality upscaling
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """
        Loại bỏ bóng & thắp sáng không đều
        Sử dụng morphological operations
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Morphological opening (remove small objects)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Morphological closing (remove small holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        # Remove shadows
        result = cv2.divide(gray, closing, scale=255)
        
        # Convert back to BGR
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def crop_document(image: np.ndarray) -> np.ndarray:
        """
        Tự động crop document (loại bỏ khoảng trắng thừa)
        Hữu ích cho ảnh chụp có background
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours of document
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1], w + 2*padding)
        h = min(image.shape[0], h + 2*padding)
        
        cropped = image[y:y+h, x:x+w]
        return cropped if cropped.size > 0 else image
    
    @staticmethod
    def enhance_text(image: np.ndarray) -> np.ndarray:
        """
        Enhance text visibility
        Kết hợp nhiều kỹ thuật để tối ưu text recognition
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Thicken text slightly (dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        
        # Convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def auto_preprocess(image: Union[Image.Image, np.ndarray], 
                       quality: str = 'medium') -> np.ndarray:
        """
        Tự động xử lý ảnh theo quality level
        
        quality:
        - 'light': Chỉ basic denoising (nhanh, đủ cho ảnh tốt)
        - 'medium': Denoise + enhance text (cân bằng tốc độ & chất lượng)
        - 'heavy': Denoise + crop + enhance + threshold (chậm, tốt cho ảnh xấu)
        """
        # Convert to CV2
        cv_image = ImagePreprocessor.to_cv2(image)
        
        # Auto-detect light text (e.g., light green/gray text on white background)
        if ImagePreprocessor.detect_light_text(cv_image):
            print("  🔍 Detected light-colored text, applying special enhancement...")
            return ImagePreprocessor.enhance_light_text(cv_image)
        
        if quality == 'light':
            # Just denoise
            return ImagePreprocessor.denoise(cv_image, strength=5)
        
        elif quality == 'medium':
            # Denoise + enhance text
            denoised = ImagePreprocessor.denoise(cv_image, strength=7)
            return ImagePreprocessor.enhance_text(denoised)
        
        elif quality == 'heavy':
            # Full preprocessing pipeline
            # 1. Denoise
            denoised = ImagePreprocessor.denoise(cv_image, strength=10)
            
            # 2. Auto deskew
            deskewed = ImagePreprocessor.auto_deskew(denoised)
            
            # 3. Crop document
            cropped = ImagePreprocessor.crop_document(deskewed)
            
            # 4. Remove shadows
            no_shadows = ImagePreprocessor.remove_shadows(cropped)
            
            # 5. Enhance text
            enhanced = ImagePreprocessor.enhance_text(no_shadows)
            
            # 6. Upscale if needed
            h, w = enhanced.shape[:2]
            if w < 300 or h < 300:  # Small image
                enhanced = ImagePreprocessor.upscale(enhanced, scale=2.0)
            
            return enhanced
        
        else:  # Default to medium
            return ImagePreprocessor.auto_preprocess(cv_image, quality='medium')
    
    @staticmethod
    def detect_text_lines(image: np.ndarray, min_height: int = 20) -> list:
        """
        Detect text lines in an image and return list of cropped line images
        Useful for VietOCR which works best on single lines
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate horizontally to connect text on same line
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Find contours (text lines)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= min_height and w > 50:  # Filter small noise
                # Add padding
                padding = 5
                y1 = max(0, y - padding)
                y2 = min(image.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(image.shape[1], x + w + padding)
                
                line_img = image[y1:y2, x1:x2]
                lines.append({
                    'image': line_img,
                    'bbox': (x1, y1, x2, y2),
                    'y': y  # For sorting
                })
        
        # Sort by Y position (top to bottom)
        lines.sort(key=lambda x: x['y'])
        
        return lines


def get_preprocessing_options():
    """Get available preprocessing options for UI"""
    return {
        'none': {'name': 'No Processing', 'description': 'Sử dụng ảnh gốc'},
        'light': {'name': 'Light (Fast)', 'description': 'Loại bỏ noise nhẹ'},
        'medium': {'name': 'Medium (Balanced)', 'description': 'Denoise + Enhance Text'},
        'heavy': {'name': 'Heavy (Best Quality)', 'description': 'Full pipeline'},
    }
