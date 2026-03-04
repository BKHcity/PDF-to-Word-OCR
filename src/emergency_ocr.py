"""
Emergency OCR - Fallback khi tất cả engines fail
Sử dụng pytesseract trực tiếp hoặc improved CRNN
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional


def emergency_tesseract_ocr(image: Union[Image.Image, np.ndarray]) -> str:
    """
    Emergency OCR sử dụng pytesseract trực tiếp
    Works ngay cả khi Tesseract không được cài trong system PATH
    """
    try:
        import pytesseract
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Try multiple Tesseract paths
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        
        # Try Vietnamese first, then English
        try:
            text = pytesseract.image_to_string(pil_image, lang='vie', config='--psm 6')
            if text.strip():
                return text.strip()
        except:
            pass
        
        try:
            text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6')
            if text.strip():
                return text.strip()
        except:
            pass
        
        # Last resort: no language specified
        try:
            text = pytesseract.image_to_string(pil_image, config='--psm 6')
            if text.strip():
                return text.strip()
        except:
            pass
        
        return ""
        
    except ImportError:
        return ""
    except Exception as e:
        print(f"Emergency Tesseract error: {e}")
        return ""


def emergency_preprocessing_and_ocr(image: Union[Image.Image, np.ndarray]) -> str:
    """
    Emergency OCR với aggressive preprocessing
    """
    
    # Convert to numpy
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    results = []
    
    # Strategy 1: High contrast + Tesseract
    try:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        text = emergency_tesseract_ocr(sharpened)
        if text:
            results.append(text)
    except:
        pass
    
    # Strategy 2: Otsu + Tesseract
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = emergency_tesseract_ocr(binary)
        if text:
            results.append(text)
    except:
        pass
    
    # Strategy 3: Adaptive + Tesseract
    try:
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        text = emergency_tesseract_ocr(adaptive)
        if text:
            results.append(text)
    except:
        pass
    
    # Strategy 4: Inverted Otsu + Tesseract
    try:
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = emergency_tesseract_ocr(binary_inv)
        if text:
            results.append(text)
    except:
        pass
    
    # Return longest result
    if results:
        return max(results, key=len)
    
    return ""


def last_resort_ocr(image: Union[Image.Image, np.ndarray, str]) -> dict:
    """
    Last resort OCR - thử mọi cách có thể
    
    Returns:
        {
            'text': recognized_text,
            'method': method_used,
            'success': boolean
        }
    """
    
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image)
    
    print("\n🚨 EMERGENCY OCR - Last Resort")
    print("="*70)
    
    # Try 1: Emergency Tesseract
    print("\n1️⃣ Trying emergency Tesseract...")
    text1 = emergency_tesseract_ocr(image)
    if text1:
        print(f"   ✅ Success: '{text1[:100]}...'")
        return {
            'text': text1,
            'method': 'emergency_tesseract',
            'success': True
        }
    else:
        print("   ❌ Failed")
    
    # Try 2: Emergency preprocessing + Tesseract
    print("\n2️⃣ Trying emergency preprocessing + Tesseract...")
    text2 = emergency_preprocessing_and_ocr(image)
    if text2:
        print(f"   ✅ Success: '{text2[:100]}...'")
        return {
            'text': text2,
            'method': 'emergency_preprocessing',
            'success': True
        }
    else:
        print("   ❌ Failed")
    
    # Try 3: Google Cloud Vision API (if credentials available)
    print("\n3️⃣ Checking for Google Cloud Vision...")
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        
        # Convert to bytes
        import io
        if isinstance(image, Image.Image):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            content = img_byte_arr.getvalue()
        else:
            _, buffer = cv2.imencode('.png', image)
            content = buffer.tobytes()
        
        gimage = vision.Image(content=content)
        response = client.text_detection(gimage)
        
        if response.text_annotations:
            text3 = response.text_annotations[0].description
            print(f"   ✅ Success: '{text3[:100]}...'")
            return {
                'text': text3,
                'method': 'google_cloud_vision',
                'success': True
            }
    except:
        print("   ⚠️  Not available (credentials not found)")
    
    # All failed
    print("\n❌ All emergency methods failed!")
    print("\n💡 SOLUTIONS:")
    print("   1. Install Tesseract OCR (5 minutes)")
    print("      Download: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   2. Or use online OCR:")
    print("      - https://www.onlineocr.net/")
    print("      - https://www.newocr.com/")
    
    return {
        'text': "",
        'method': 'all_failed',
        'success': False
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        result = last_resort_ocr(sys.argv[1])
        
        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)
        print(f"Text: {result['text']}")
        print(f"Method: {result['method']}")
        print(f"Success: {result['success']}")
    else:
        print("Usage: python emergency_ocr.py <image_path>")




