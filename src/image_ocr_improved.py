"""
Improved Image OCR - Multiple Preprocessing Strategies
Cải thiện OCR cho ảnh với nhiều chiến lược tiền xử lý
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Tuple, Optional, Union
import torch
import torch.nn.functional as F


class ImprovedImagePreprocessor:
    """Preprocessor với nhiều chiến lược cho chữ in"""
    
    @staticmethod
    def create_variations(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Tạo nhiều variations của ảnh với các preprocessing khác nhau
        Returns: [(name, processed_image), ...]
        """
        variations = []
        
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Original + Basic Enhancement
        enhanced = cv2.equalizeHist(gray)
        variations.append(("enhanced", enhanced))
        
        # 2. High Contrast Adaptive
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        adaptive = clahe.apply(gray)
        variations.append(("adaptive", adaptive))
        
        # 3. Otsu Binarization
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variations.append(("otsu", otsu))
        
        # 4. Adaptive Threshold
        adaptive_th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        variations.append(("adaptive_th", adaptive_th))
        
        # 5. High Contrast + Sharpening
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(adaptive, -1, kernel)
        variations.append(("sharpened", sharpened))
        
        # 6. Morphology Cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        variations.append(("morph", morph))
        
        return variations
    
    @staticmethod
    def preprocess_for_printed_text(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Best preprocessing cho chữ in"""
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Step 1: Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 2: Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 4: Adaptive threshold
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary


def multi_preprocessing_ocr(ocr_system, image: Union[Image.Image, np.ndarray],
                            engines: Optional[List[str]] = None,
                            voting_method: str = 'weighted') -> dict:
    """
    Thử OCR với nhiều preprocessing variations, chọn kết quả tốt nhất
    
    Returns:
        {
            'text': best_text,
            'confidence': best_confidence,
            'engine': best_engine,
            'preprocessing': best_preprocessing,
            'all_results': all_results
        }
    """
    
    # Convert to numpy
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Create preprocessing variations
    preprocessor = ImprovedImagePreprocessor()
    variations = preprocessor.create_variations(img_array)
    
    all_results = []
    best_result = None
    best_conf = 0.0
    
    print(f"\n🔍 Trying {len(variations)} preprocessing strategies...")
    
    for name, processed in variations:
        try:
            # Convert back to PIL for OCR
            pil_image = Image.fromarray(processed)
            
            # Run OCR
            result = ocr_system.recognize(
                pil_image,
                engines=engines,
                voting_method=voting_method
            )
            
            # Check if better
            if result.confidence > best_conf and result.text.strip() and len(result.text.strip()) > 1:
                best_result = {
                    'text': result.text,
                    'confidence': result.confidence,
                    'engine': result.best_engine,
                    'preprocessing': name
                }
                best_conf = result.confidence
                
                print(f"  ✅ {name}: '{result.text[:50]}...' (conf: {result.confidence:.2f})")
            else:
                print(f"  ⚠️  {name}: Low quality result")
            
            all_results.append({
                'preprocessing': name,
                'text': result.text,
                'confidence': result.confidence,
                'engine': result.best_engine
            })
            
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    if best_result:
        print(f"\n✅ Best: {best_result['preprocessing']} → '{best_result['text'][:80]}...'")
        best_result['all_results'] = all_results
        return best_result
    else:
        # Fallback to simple enhancement
        print("\n⚠️  All strategies failed, using simple enhancement...")
        enhanced = preprocessor.preprocess_for_printed_text(img_array)
        pil_enhanced = Image.fromarray(enhanced)
        
        result = ocr_system.recognize(
            pil_enhanced,
            engines=engines,
            voting_method=voting_method
        )
        
        return {
            'text': result.text,
            'confidence': result.confidence,
            'engine': result.best_engine,
            'preprocessing': 'fallback_enhanced',
            'all_results': all_results
        }


def ensemble_with_preprocessing(ocr_system, image: Union[Image.Image, np.ndarray],
                               engines: Optional[List[str]] = None) -> str:
    """
    Quick function - Ensemble OCR với best preprocessing
    Returns: text only
    """
    result = multi_preprocessing_ocr(ocr_system, image, engines)
    return result.get('text', '')




