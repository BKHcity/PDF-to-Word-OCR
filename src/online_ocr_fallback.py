"""
Online OCR Fallback - Sử dụng OCR APIs miễn phí
Khi tất cả local engines fail, fallback to online services
"""

import io
import base64
import requests
from PIL import Image
import numpy as np
from typing import Union, Optional


def ocr_space_api(image: Union[Image.Image, np.ndarray], 
                  language: str = 'vie',
                  api_key: str = 'helloworld') -> dict:
    """
    OCR.space API - Free OCR service
    https://ocr.space/ocrapi
    
    Free tier: 500 requests/day
    """
    
    try:
        # Convert to bytes
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # API request
        url = 'https://api.ocr.space/parse/image'
        
        payload = {
            'apikey': api_key,
            'language': language,  # 'vie' for Vietnamese
            'isOverlayRequired': False,
            'OCREngine': 2,  # Engine 2 is better for Asian languages
        }
        
        files = {
            'file': ('image.png', img_bytes, 'image/png')
        }
        
        response = requests.post(url, files=files, data=payload, timeout=30)
        result = response.json()
        
        if result.get('IsErroredOnProcessing'):
            error_msg = result.get('ErrorMessage', ['Unknown error'])[0]
            return {
                'text': '',
                'success': False,
                'error': error_msg,
                'service': 'ocr.space'
            }
        
        # Extract text
        parsed_text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
        
        return {
            'text': parsed_text.strip(),
            'success': True,
            'confidence': 0.85,
            'service': 'ocr.space'
        }
        
    except Exception as e:
        return {
            'text': '',
            'success': False,
            'error': str(e),
            'service': 'ocr.space'
        }


def api_ninjas_ocr(image: Union[Image.Image, np.ndarray],
                   api_key: str = 'YOUR_API_KEY') -> dict:
    """
    API Ninjas OCR - Free OCR service
    https://api-ninjas.com/api/imagetotext
    
    Free tier: 50,000 requests/month
    """
    
    try:
        # Convert to base64
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        url = 'https://api.api-ninjas.com/v1/imagetotext'
        
        response = requests.post(
            url,
            headers={'X-Api-Key': api_key},
            files={'image': img_base64},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text = '\n'.join([item.get('text', '') for item in result])
            
            return {
                'text': text.strip(),
                'success': True,
                'confidence': 0.85,
                'service': 'api-ninjas'
            }
        else:
            return {
                'text': '',
                'success': False,
                'error': f'API error: {response.status_code}',
                'service': 'api-ninjas'
            }
            
    except Exception as e:
        return {
            'text': '',
            'success': False,
            'error': str(e),
            'service': 'api-ninjas'
        }


def online_ocr_fallback(image: Union[Image.Image, np.ndarray]) -> dict:
    """
    Try multiple free online OCR services as fallback
    
    Returns:
        {
            'text': recognized_text,
            'success': boolean,
            'service': service_name,
            'confidence': float
        }
    """
    
    print("\n🌐 Trying online OCR services...")
    
    # Try OCR.space (free, no API key needed)
    print("  1️⃣ Trying OCR.space...")
    result = ocr_space_api(image, language='vie')
    
    if result['success'] and result['text']:
        print(f"     ✅ Success: '{result['text'][:100]}...'")
        return result
    else:
        print(f"     ❌ Failed: {result.get('error', 'No text')}")
    
    # All failed
    print("\n❌ All online services failed!")
    
    return {
        'text': '',
        'success': False,
        'service': 'none',
        'confidence': 0.0,
        'error': 'All services failed. Please install Tesseract or VietOCR.'
    }


def smart_ocr_with_fallback(ocr_system, image: Union[Image.Image, np.ndarray],
                            engines: Optional[list] = None,
                            voting_method: str = 'weighted',
                            use_online: bool = True) -> dict:
    """
    Smart OCR với online fallback
    
    Pipeline:
    1. Try local engines
    2. If fail → Try online services
    3. Return best result
    """
    
    # Try local first
    print("\n🔍 Trying local OCR engines...")
    try:
        result = ocr_system.recognize(
            image,
            engines=engines,
            voting_method=voting_method
        )
        
        # Check if result is good
        if result.text and len(result.text.strip()) > 2:
            print(f"✅ Local OCR success: '{result.text[:100]}...'")
            return {
                'text': result.text,
                'success': True,
                'confidence': result.confidence,
                'engine': result.best_engine,
                'service': 'local',
                'all_results': result.all_results
            }
        else:
            print(f"⚠️  Local OCR found minimal text: '{result.text}'")
    except Exception as e:
        print(f"❌ Local OCR error: {e}")
    
    # Fallback to online
    if use_online:
        print("\n🌐 Falling back to online OCR...")
        online_result = online_ocr_fallback(image)
        
        if online_result['success']:
            return online_result
    
    # All failed
    return {
        'text': '',
        'success': False,
        'confidence': 0.0,
        'engine': 'none',
        'service': 'none',
        'error': 'Both local and online OCR failed. Install Tesseract for 95%+ accuracy.'
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
        if Path(img_path).exists():
            img = Image.open(img_path)
            
            print("="*80)
            print("TESTING ONLINE OCR FALLBACK")
            print("="*80)
            
            result = online_ocr_fallback(img)
            
            print("\n" + "="*80)
            print("RESULT")
            print("="*80)
            print(f"Success: {result['success']}")
            print(f"Service: {result['service']}")
            print(f"Text: {result.get('text', '')[:200]}...")
        else:
            print(f"File not found: {img_path}")
    else:
        print("Usage: python online_ocr_fallback.py <image_path>")




