"""
Advanced Image Processor for Vietnamese OCR
Multi-scale processing, deep cleaning, and intelligent text extraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)


class AdvancedImageProcessor:
    """
    Advanced image processing with multi-scale OCR approach
    - Deep cleaning and enhancement
    - Multi-scale processing (zoom in/out)
    - Color-based text extraction
    - Intelligent result selection
    """
    
    def __init__(self):
        self.scale_factors = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        self.enhancement_methods = [
            'original',
            'invert',
            'high_contrast',
            'color_isolation',
            'adaptive_threshold',
            'morphological',
            'unsharp_mask',
            'denoised'
        ]
    
    def process_for_ocr(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Process image with multiple methods and scales
        Returns list of (processed_image, method_name) tuples
        """
        results = []
        
        # 1. Deep clean the original
        cleaned = self.deep_clean(image)
        results.append((cleaned, "deep_cleaned"))
        
        # 2. Try different enhancement methods
        for method in self.enhancement_methods:
            try:
                enhanced = self.apply_enhancement(image, method)
                if enhanced is not None:
                    results.append((enhanced, method))
            except Exception as e:
                logger.debug(f"Enhancement {method} failed: {e}")
        
        # 3. Multi-scale versions of best cleaned image
        for scale in self.scale_factors:
            if scale != 1.0:
                try:
                    scaled = self.scale_image(cleaned, scale)
                    results.append((scaled, f"scale_{scale}"))
                except Exception as e:
                    logger.debug(f"Scale {scale} failed: {e}")
        
        return results
    
    def deep_clean(self, image: np.ndarray) -> np.ndarray:
        """
        Deep cleaning pipeline for maximum text clarity
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Step 1: Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Step 2: White balance / color correction
        corrected = self.white_balance(denoised)
        
        # Step 3: Increase sharpness
        sharpened = self.sharpen(corrected)
        
        # Step 4: Enhance contrast
        enhanced = self.enhance_contrast(sharpened)
        
        return enhanced
    
    def white_balance(self, image: np.ndarray) -> np.ndarray:
        """Auto white balance using gray world algorithm"""
        result = image.copy().astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        if avg_b > 0:
            result[:, :, 0] *= avg_gray / avg_b
        if avg_g > 0:
            result[:, :, 1] *= avg_gray / avg_g
        if avg_r > 0:
            result[:, :, 2] *= avg_gray / avg_r
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen using unsharp mask"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE on LAB color space"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def apply_enhancement(self, image: np.ndarray, method: str) -> Optional[np.ndarray]:
        """Apply specific enhancement method"""
        
        if method == 'original':
            return image.copy()
        
        elif method == 'invert':
            return self.invert_smart(image)
        
        elif method == 'high_contrast':
            return self.high_contrast(image)
        
        elif method == 'color_isolation':
            return self.isolate_text_colors(image)
        
        elif method == 'adaptive_threshold':
            return self.adaptive_threshold(image)
        
        elif method == 'morphological':
            return self.morphological_clean(image)
        
        elif method == 'unsharp_mask':
            return self.unsharp_mask(image)
        
        elif method == 'denoised':
            return cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21)
        
        return None
    
    def invert_smart(self, image: np.ndarray) -> np.ndarray:
        """
        Smart inversion - detect if image is light or dark
        and invert appropriately to make text dark on light background
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        
        # If image is light (white background), invert to make light text visible
        if mean_val > 127:
            inverted = cv2.bitwise_not(image)
            # Convert to grayscale and enhance
            gray_inv = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_inv)
            # Threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert back so text is dark
            binary = cv2.bitwise_not(binary)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return image.copy()
    
    def high_contrast(self, image: np.ndarray) -> np.ndarray:
        """Maximum contrast enhancement"""
        # Convert to PIL for better enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def isolate_text_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Isolate common text colors (including light colors like cyan, light green, light blue)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for various text colors
        masks = []
        
        # Dark colors (black, dark blue, dark green, etc.)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
        masks.append(dark_mask)
        
        # Blue/Cyan range (light blue text)
        blue_mask = cv2.inRange(hsv, (80, 20, 50), (130, 255, 255))
        masks.append(blue_mask)
        
        # Green range (light green text)
        green_mask = cv2.inRange(hsv, (35, 20, 50), (85, 255, 255))
        masks.append(green_mask)
        
        # Red/Pink range
        red_mask1 = cv2.inRange(hsv, (0, 20, 50), (15, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 20, 50), (180, 255, 255))
        masks.append(red_mask1)
        masks.append(red_mask2)
        
        # Combine all masks
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Dilate to connect text
        kernel = np.ones((2, 2), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # Create white background, black text
        result = np.ones_like(image) * 255
        result[combined_mask > 0] = [0, 0, 0]
        
        return result
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for variable lighting"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def morphological_clean(self, image: np.ndarray) -> np.ndarray:
        """Morphological operations to clean text"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Close small gaps in text
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Invert back
        result = cv2.bitwise_not(opened)
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    def unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp mask for edge enhancement"""
        gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp
    
    def scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by factor"""
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use different interpolation based on scale
        if scale > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    def extract_text_region(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and extract only the text region, removing excessive whitespace
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect text regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get bounding box of all text
        x_min = image.shape[1]
        y_min = image.shape[0]
        x_max = 0
        y_max = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Crop
        return image[y_min:y_max, x_min:x_max]
    
    def process_light_text(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Special processing for light colored text on white/light background
        Returns multiple processed versions
        """
        results = []
        
        # Method 1: Color channel analysis
        b, g, r = cv2.split(image)
        
        # For cyan/light blue text: high in B and G, low in R relative to white
        cyan_diff = cv2.subtract(cv2.add(b, g), r)
        _, cyan_mask = cv2.threshold(cyan_diff, 10, 255, cv2.THRESH_BINARY)
        cyan_result = np.ones_like(image) * 255
        cyan_result[cyan_mask > 0] = [0, 0, 0]
        results.append((cyan_result, "cyan_extraction"))
        
        # Method 2: Saturation-based (colored text has more saturation than white)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Any saturation indicates color
        _, sat_mask = cv2.threshold(s, 5, 255, cv2.THRESH_BINARY)
        sat_result = np.ones_like(image) * 255
        sat_result[sat_mask > 0] = [0, 0, 0]
        results.append((sat_result, "saturation_extraction"))
        
        # Method 3: LAB color space - A and B channels detect color
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        
        # Deviation from neutral (128) indicates color
        a_dev = np.abs(a.astype(np.int16) - 128).astype(np.uint8)
        b_dev = np.abs(b_ch.astype(np.int16) - 128).astype(np.uint8)
        color_dev = cv2.add(a_dev, b_dev)
        _, lab_mask = cv2.threshold(color_dev, 5, 255, cv2.THRESH_BINARY)
        lab_result = np.ones_like(image) * 255
        lab_result[lab_mask > 0] = [0, 0, 0]
        results.append((lab_result, "lab_extraction"))
        
        # Method 4: Edge enhancement then threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert (light text becomes dark)
        inverted = cv2.bitwise_not(gray)
        # Enhance
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(inverted)
        # High contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=3.0, beta=-100)
        _, edge_binary = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)
        results.append((cv2.cvtColor(edge_binary, cv2.COLOR_GRAY2BGR), "edge_enhancement"))
        
        # Method 5: Difference from white
        white = np.ones_like(image) * 255
        diff = cv2.absdiff(image, white)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)
        # Morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        diff_result = np.ones_like(image) * 255
        diff_result[diff_mask > 0] = [0, 0, 0]
        results.append((diff_result, "white_difference"))
        
        return results
    
    def get_best_result(self, results: List[Tuple[str, float, str]]) -> Tuple[str, float, str]:
        """
        Select best OCR result from multiple attempts
        results: List of (text, confidence, method)
        """
        if not results:
            return ("", 0.0, "none")
        
        # Filter out garbage results
        valid_results = []
        for text, conf, method in results:
            if self.is_valid_vietnamese(text) and len(text.strip()) > 0:
                valid_results.append((text, conf, method))
        
        if not valid_results:
            # Return longest result if no valid Vietnamese
            return max(results, key=lambda x: len(x[0]))
        
        # Score each result
        scored = []
        for text, conf, method in valid_results:
            score = self.score_result(text, conf)
            scored.append((text, conf, method, score))
        
        # Return highest scored
        best = max(scored, key=lambda x: x[3])
        return (best[0], best[1], best[2])
    
    def score_result(self, text: str, confidence: float) -> float:
        """Score a result based on various factors"""
        score = confidence
        
        # Bonus for Vietnamese characters
        vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
        text_lower = text.lower()
        viet_count = sum(1 for c in text_lower if c in vietnamese_chars)
        score += viet_count * 0.05
        
        # Bonus for reasonable length
        if 5 <= len(text) <= 200:
            score += 0.1
        
        # Bonus for having spaces (multiple words)
        if ' ' in text:
            score += 0.1
        
        # Penalty for too many non-alphabetic characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.5:
            score -= 0.2
        
        return score
    
    def is_valid_vietnamese(self, text: str) -> bool:
        """Check if text looks like valid Vietnamese"""
        if not text or len(text.strip()) < 2:
            return False
        
        # Check for common garbage patterns
        garbage_patterns = [
            'glangers', 'lorem', 'ipsum', 'quack',
            'www.', 'http', '.com', '.net'
        ]
        text_lower = text.lower()
        for pattern in garbage_patterns:
            if pattern in text_lower:
                return False
        
        # Check character distribution
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return False
        
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / total_chars
        
        # Text should be mostly alphabetic
        return alpha_ratio > 0.6


class MultiScaleOCR:
    """
    Multi-scale OCR processor that tries multiple scales and methods
    """
    
    def __init__(self, ocr_engine):
        self.ocr = ocr_engine
        self.processor = AdvancedImageProcessor()
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    def recognize_multi_scale(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Try OCR at multiple scales and return best result
        """
        results = []
        debug_info = {
            'scales_tried': [],
            'methods_tried': [],
            'all_results': []
        }
        
        # Check if light image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        is_light = mean_val > 180 and std_val < 50
        
        if is_light:
            logger.info(f"Detected light image (mean={mean_val:.0f}, std={std_val:.0f})")
            # Process light text specially
            light_results = self.processor.process_light_text(image)
            for processed, method in light_results:
                debug_info['methods_tried'].append(method)
                try:
                    text, conf = self._run_ocr(processed)
                    if text and len(text.strip()) > 0:
                        results.append((text, conf, method))
                        debug_info['all_results'].append({
                            'method': method,
                            'text': text,
                            'confidence': conf
                        })
                except Exception as e:
                    logger.debug(f"OCR failed for {method}: {e}")
        
        # Also try multi-scale on cleaned image
        cleaned = self.processor.deep_clean(image)
        
        for scale in self.scales:
            debug_info['scales_tried'].append(scale)
            try:
                scaled = self.processor.scale_image(cleaned, scale)
                text, conf = self._run_ocr(scaled)
                if text and len(text.strip()) > 0:
                    results.append((text, conf, f"scale_{scale}"))
                    debug_info['all_results'].append({
                        'method': f'scale_{scale}',
                        'text': text,
                        'confidence': conf
                    })
            except Exception as e:
                logger.debug(f"OCR failed for scale {scale}: {e}")
        
        # Get best result
        if results:
            best_text, best_conf, best_method = self.processor.get_best_result(results)
            debug_info['best_method'] = best_method
            return best_text, best_conf, debug_info
        
        return "", 0.0, debug_info
    
    def _run_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR on image"""
        # This will be implemented to use actual OCR engine
        # For now, placeholder
        return self.ocr.recognize(image)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Main entry point for image enhancement
    Returns the best enhanced version for OCR
    """
    processor = AdvancedImageProcessor()
    
    # Deep clean
    cleaned = processor.deep_clean(image)
    
    # Check if light image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    if mean_val > 180 and std_val < 50:
        # Light image - use color extraction
        light_results = processor.process_light_text(image)
        # Return the saturation extraction as it's most general
        for result, method in light_results:
            if method == "saturation_extraction":
                return result
        # Fallback
        return light_results[0][0] if light_results else cleaned
    
    return cleaned
