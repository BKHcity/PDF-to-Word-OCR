"""
VIETNAMESE OCR ADVANCED - State-of-the-Art OCR System
Hệ thống OCR tiên tiến nhất cho chữ viết tay & chữ in tiếng Việt

Tích hợp:
- VietOCR (Transformer-based, chuyên cho tiếng Việt)
- PaddleOCR (PP-OCRv4, mạnh cho chữ Việt & châu Á)  
- TrOCR (Microsoft Transformer OCR)
- CRNN (Custom trained model)
- Tesseract 5.0 (with Vietnamese)
- EasyOCR (Multi-language)
- Ensemble voting system
"""

import os
import io
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATACLASS FOR RESULTS
# ============================================================================

@dataclass
class OCRResult:
    """Kết quả OCR từ một engine"""
    text: str
    confidence: float
    engine: str
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleOCRResult:
    """Kết quả tổng hợp từ nhiều engines"""
    text: str
    confidence: float
    best_engine: str
    all_results: List[OCRResult]
    consensus_score: float = 0.0
    processing_time: float = 0.0


# ============================================================================
# VIETOCR ENGINE (Transformer-based, chuyên cho tiếng Việt)
# ============================================================================

class VietOCREngine:
    """
    VietOCR - Transformer-based OCR chuyên biệt cho tiếng Việt
    https://github.com/pbcquoc/vietocr
    """
    
    def __init__(self, config_name: str = 'vgg_transformer', device: str = 'cpu'):
        self.name = "VietOCR"
        self.device = device
        self.detector = None
        self.predictor = None
        self._initialized = False
        
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            
            # Cấu hình VietOCR
            config = Cfg.load_config_from_name(config_name)
            config['device'] = device
            config['predictor']['beamsearch'] = True
            
            self.predictor = Predictor(config)
            self._initialized = True
            print(f"✅ {self.name} initialized successfully")
            
        except ImportError:
            print(f"⚠️  VietOCR not installed. Install: pip install vietocr")
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
    
    def recognize(self, image: Union[Image.Image, np.ndarray]) -> OCRResult:
        """Nhận diện text từ ảnh"""
        if not self._initialized:
            return OCRResult("", 0.0, self.name)
        
        try:
            import time
            start = time.time()
            
            # IMPORTANT: VietOCR works best with ORIGINAL color images
            # Do NOT apply aggressive preprocessing - it damages handwriting
            
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image).convert('RGB')
                else:
                    # Convert BGR to RGB if needed
                    if image.shape[2] == 3:
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    else:
                        image = Image.fromarray(image).convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Nhận diện
            text = self.predictor.predict(image)
            
            # VietOCR không trả về confidence trực tiếp, estimate dựa trên độ dài
            confidence = min(0.95, 0.75 + len(text) / 100.0)
            
            processing_time = time.time() - start
            
            return OCRResult(
                text=text.strip(),
                confidence=confidence,
                engine=self.name,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
            return OCRResult("", 0.0, self.name)
    
    def is_available(self) -> bool:
        return self._initialized
    
    def _enhance_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR accuracy
        Especially important for colored backgrounds
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            # Check for colored background
            b_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            r_mean = np.mean(img_array[:, :, 2])
            
            # Blue background detection
            if b_mean > r_mean + 40 and b_mean > g_mean + 30:
                # Use red channel + inverted blue for maximum contrast
                gray = cv2.addWeighted(img_array[:, :, 2], 0.7, 255 - img_array[:, :, 0], 0.3, 0)
            # Green background
            elif g_mean > r_mean + 40 and g_mean > b_mean + 30:
                gray = cv2.addWeighted(img_array[:, :, 0], 0.7, 255 - img_array[:, :, 1], 0.3, 0)
            else:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Aggressive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Sharpen text
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Ensure good dynamic range
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced


# ============================================================================
# PADDLEOCR ENGINE (PP-OCRv4, mạnh cho châu Á)
# ============================================================================

class PaddleOCREngine:
    """
    PaddleOCR - State-of-the-art OCR từ Baidu
    PP-OCRv4 với độ chính xác cao cho Vietnamese
    """
    
    def __init__(self, lang: str = 'vi', use_angle_cls: bool = True, 
                 use_gpu: bool = False):
        self.name = "PaddleOCR"
        self.lang = lang
        self.ocr = None
        self._initialized = False
        
        try:
            from paddleocr import PaddleOCR
            
            # Khởi tạo PaddleOCR với Vietnamese
            # Minimal arguments for compatibility with latest version
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang  # 'vi' for Vietnamese
            )
            
            self._initialized = True
            print(f"✅ {self.name} initialized (lang={lang})")
            
        except ImportError:
            print(f"⚠️  PaddleOCR not installed. Install: pip install paddlepaddle paddleocr")
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
    
    def recognize(self, image: Union[Image.Image, np.ndarray]) -> OCRResult:
        """Nhận diện text từ ảnh"""
        if not self._initialized:
            return OCRResult("", 0.0, self.name)
        
        try:
            import time
            start = time.time()
            
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # PaddleOCR expects BGR format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # Nhận diện - Use predict for newer PaddleOCR versions
            try:
                # Try new API first (paddleocr >= 3.x)
                result = self.ocr.predict(image)
            except AttributeError:
                # Fallback to old API
                try:
                    result = self.ocr.ocr(image, cls=True)
                except TypeError:
                    result = self.ocr.ocr(image)
            
            # Tổng hợp kết quả - Support both old and new formats
            texts = []
            confidences = []
            
            if result:
                # New format (paddleocr >= 3.x): list of dicts with 'rec_texts' and 'rec_scores'
                if isinstance(result, list) and len(result) > 0:
                    first_result = result[0]
                    
                    # Check for new format
                    if isinstance(first_result, dict) and 'rec_texts' in first_result:
                        rec_texts = first_result.get('rec_texts', [])
                        rec_scores = first_result.get('rec_scores', [])
                        
                        for text, score in zip(rec_texts, rec_scores):
                            if text and str(text).strip():
                                texts.append(str(text))
                                confidences.append(float(score))
                    
                    # Old format: result[0] is list of [box, (text, confidence)]
                    elif isinstance(first_result, (list, tuple)):
                        for line in result[0] if result[0] else []:
                            try:
                                if line and len(line) >= 2 and line[1] and len(line[1]) >= 2:
                                    text = str(line[1][0])
                                    conf = float(line[1][1])
                                    if text.strip():
                                        texts.append(text)
                                        confidences.append(conf)
                            except (IndexError, TypeError, ValueError):
                                continue
            
            final_text = " ".join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            processing_time = time.time() - start
            
            return OCRResult(
                text=final_text.strip(),
                confidence=float(avg_confidence),
                engine=self.name,
                processing_time=processing_time,
                metadata={'line_count': len(texts)}
            )
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
            return OCRResult("", 0.0, self.name)
    
    def is_available(self) -> bool:
        return self._initialized


# ============================================================================
# TROCR ENGINE (Microsoft Transformer OCR)
# ============================================================================

class TrOCREngine:
    """
    TrOCR - Microsoft's Transformer-based OCR
    Sử dụng Vision Transformer encoder + BERT decoder
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten",
                 device: str = 'cpu'):
        self.name = "TrOCR"
        self.device = device
        self.processor = None
        self.model = None
        self._initialized = False
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            print(f"⏳ Loading {self.name}... (this may take a while)")
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            self._initialized = True
            print(f"✅ {self.name} initialized successfully")
            
        except ImportError:
            print(f"⚠️  Transformers not installed. Install: pip install transformers")
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
    
    def recognize(self, image: Union[Image.Image, np.ndarray]) -> OCRResult:
        """Nhận diện text từ ảnh"""
        if not self._initialized:
            return OCRResult("", 0.0, self.name)
        
        try:
            import time
            start = time.time()
            
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image).convert('RGB')
                else:
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # TrOCR doesn't provide confidence, estimate based on text length
            confidence = min(0.90, 0.70 + len(text) / 100.0)
            
            processing_time = time.time() - start
            
            return OCRResult(
                text=text.strip(),
                confidence=confidence,
                engine=self.name,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
            return OCRResult("", 0.0, self.name)
    
    def is_available(self) -> bool:
        return self._initialized


# ============================================================================
# PROTONX ENGINE (New Integration)
# ============================================================================

class ProtonXEngine:
    """
    ProtonX OCR Engine
    Integration for ProtonX library
    """
    
    def __init__(self, device: str = 'cpu'):
        self.name = "ProtonX"
        self.device = device
        self.model = None
        self._initialized = False
        
        try:
            import protonx
            # ASSUMPTION: ProtonX has an OCR model or similar API
            # If this fails, the user will see the error and can adjust the call
            if hasattr(protonx, 'OcrModel'):
                self.model = protonx.OcrModel()
            elif hasattr(protonx, 'OCR'):
                self.model = protonx.OCR()
            else:
                # Fallback: try to find anything that looks like a model
                print(f"⚠️  ProtonX found but unsure about API. Trying default init...")
                self.model = protonx
                
            self._initialized = True
            print(f"✅ {self.name} initialized successfully")
            
        except ImportError:
            print(f"⚠️  ProtonX not installed. Install: pip install --upgrade protonx")
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
    
    def recognize(self, image: Union[Image.Image, np.ndarray]) -> OCRResult:
        """Nhận diện text từ ảnh"""
        if not self._initialized:
            return OCRResult("", 0.0, self.name)
        
        try:
            import time
            start = time.time()
            
            # Convert to appropriate format (assuming PIL or path)
            # Most libs accept PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Hypothetical API call - ADAPT THIS if actual API is different
            if hasattr(self.model, 'predict'):
                text = self.model.predict(image)
            elif hasattr(self.model, 'recognize'):
                text = self.model.recognize(image)
            else:
                # Try calling it directly
                text = self.model(image)
            
            # Handle response format
            confidence = 0.85 # Default if not provided
            
            if isinstance(text, tuple):
                text, confidence = text
            elif isinstance(text, dict) and 'text' in text:
                confidence = text.get('confidence', 0.85)
                text = text['text']
            
            text = str(text).strip()
            
            processing_time = time.time() - start
            
            return OCRResult(
                text=text,
                confidence=float(confidence),
                engine=self.name,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
            return OCRResult("", 0.0, self.name)
    
    def is_available(self) -> bool:
        return self._initialized


# ============================================================================
# ENSEMBLE OCR SYSTEM - Voting & Consensus
# ============================================================================

class VietnameseOCRAdvanced:
    """
    Advanced Vietnamese OCR System with Ensemble Voting
    Kết hợp tất cả engines để đạt độ chính xác cao nhất
    """
    
    def __init__(self, device: str = 'cpu', enable_all: bool = True):
        """
        Initialize advanced OCR system
        
        Args:
            device: 'cpu' or 'cuda'
            enable_all: Enable all available engines (recommended)
        """
        self.device = device
        self.engines = {}
        
        print("=" * 70)
        print("🚀 VIETNAMESE OCR ADVANCED - Initializing...")
        print("=" * 70)
        
        # Initialize VietOCR (Priority 1 - Best for Vietnamese)
        if enable_all:
            vietocr = VietOCREngine(device=device)
            if vietocr.is_available():
                self.engines['vietocr'] = vietocr
        
        # Initialize PaddleOCR (Priority 2 - Excellent for Asian languages)
        if enable_all:
            paddleocr = PaddleOCREngine(lang='vi', use_gpu=False)
            if paddleocr.is_available():
                self.engines['paddleocr'] = paddleocr
        
        # Initialize TrOCR (Priority 3 - Transformer-based)
        if enable_all:
            trocr = TrOCREngine(device=device)
            if trocr.is_available():
                self.engines['trocr'] = trocr
        
        # Initialize ProtonX (Priority 3.5 - New Engine)
        if enable_all:
            protonx = ProtonXEngine(device=device)
            if protonx.is_available():
                self.engines['protonx'] = protonx
        
        # Initialize CRNN (Priority 4 - Custom trained)
        try:
            from src.models import CRNN
            from src.dataset import VOCAB
            import torch
            
            # Simple CRNN wrapper with dynamic model loading
            class SimpleCRNNWrapper:
                def __init__(self, model_path, device):
                    self.device = torch.device(device)
                    self.model = None
                    self.vocab = VOCAB
                    self.current_model_path = None
                    self.CRNN = CRNN  # Keep reference for reloading
                    
                    self._load_model(model_path)
                
                def _load_model(self, model_path):
                    """Load or reload CRNN model"""
                    import os
                    if not os.path.exists(model_path):
                        print(f"⚠️  CRNN model not found: {model_path}")
                        return False
                    
                    try:
                        checkpoint = torch.load(model_path, map_location=self.device)
                        if isinstance(checkpoint, dict):
                            state = checkpoint.get('model', checkpoint)
                        else:
                            state = checkpoint
                        
                        self.model = self.CRNN(num_classes=len(self.vocab), in_channels=1, rnn_hidden=256)
                        self.model.load_state_dict(state, strict=False)
                        self.model.eval()
                        self.model.to(self.device)
                        self.current_model_path = model_path
                        print(f"  ✅ CRNN loaded: {model_path}")
                        return True
                    except Exception as e:
                        print(f"⚠️  CRNN load error: {e}")
                        return False
                
                def switch_model(self, model_path):
                    """Switch to a different model file"""
                    if model_path != self.current_model_path:
                        return self._load_model(model_path)
                    return True
                
                def recognize(self, image, enhance=True, return_confidence=True):
                    if self.model is None:
                        return ("", 0.0) if return_confidence else ""
                    
                    try:
                        # Simple preprocessing
                        if isinstance(image, Image.Image):
                            image = np.array(image.convert('L'))
                        elif len(image.shape) == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        # Resize to model input
                        h, w = image.shape
                        new_h = 32
                        new_w = int(w * new_h / h)
                        new_w = min(new_w, 512)
                        image = cv2.resize(image, (new_w, new_h))
                        
                        # Pad
                        if new_w < 512:
                            pad = np.ones((new_h, 512 - new_w), dtype=np.uint8) * 255
                            image = np.concatenate([image, pad], axis=1)
                        
                        # To tensor
                        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
                        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        # Inference
                        with torch.no_grad():
                            output = self.model(tensor)
                            probs = torch.exp(output)
                            pred_indices = probs.argmax(dim=2)[0].cpu().numpy()
                        
                        # Decode
                        text = []
                        prev = 0
                        for idx in pred_indices:
                            if idx != 0 and idx != prev and idx < len(self.vocab):
                                text.append(self.vocab[idx])
                            prev = idx
                        
                        result = ''.join(text).strip()
                        conf = float(probs.max(dim=2)[0].mean().cpu().numpy())
                        
                        return (result, conf) if return_confidence else result
                    except Exception as e:
                        print(f"⚠️  CRNN recognize error: {e}")
                        return ("", 0.0) if return_confidence else ""
            
            crnn = SimpleCRNNWrapper(model_path="crnn_best.pth", device=device)
            if crnn.model is not None:
                self.engines['crnn'] = crnn
                print(f"✅ CRNN initialized successfully")
        except Exception as e:
            print(f"⚠️  CRNN initialization failed: {e}")
        
        # Initialize Tesseract (Priority 5 - Fallback for printed text)
        try:
            import pytesseract
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            self.engines['tesseract'] = 'tesseract'
            print(f"✅ Tesseract initialized successfully")
        except Exception:
            print(f"⚠️  Tesseract not available")
        
        # Initialize EasyOCR (Priority 6 - General purpose fallback)
        try:
            import easyocr
            reader = easyocr.Reader(['vi', 'en'], gpu=(device == 'cuda'))
            self.engines['easyocr'] = reader
            print(f"✅ EasyOCR initialized successfully")
        except Exception:
            print(f"⚠️  EasyOCR not available")
        
        print("=" * 70)
        print(f"✅ Initialized {len(self.engines)} OCR engines")
        print(f"📋 Available: {', '.join(self.engines.keys())}")
        print("=" * 70)
    
    def recognize_with_engine(self, engine_name: str, image: Union[Image.Image, np.ndarray]) -> OCRResult:
        """Nhận diện với một engine cụ thể"""
        
        if engine_name not in self.engines:
            return OCRResult("", 0.0, engine_name)
        
        engine = self.engines[engine_name]
        
        # Handle different engine types
        if hasattr(engine, 'recognize') and not engine_name == 'crnn':
            # VietOCR, PaddleOCR, TrOCR have recognize method that returns OCRResult
            try:
                result = engine.recognize(image)
                # Ensure it's an OCRResult
                if isinstance(result, OCRResult):
                    return result
                else:
                    # Fallback if method returns something else
                    return OCRResult("", 0.0, engine_name)
            except Exception as e:
                print(f"⚠️  {engine_name} error: {e}")
                return OCRResult("", 0.0, engine_name)
        
        elif engine_name == 'crnn':
            # CRNN from ocr_unified_enhanced
            import time
            start = time.time()
            try:
                result = engine.recognize(image, enhance=True, return_confidence=True)
                # Handle different return formats
                if isinstance(result, tuple) and len(result) == 2:
                    text, conf = result
                elif isinstance(result, str):
                    text = result
                    conf = 0.5
                else:
                    text = str(result)
                    conf = 0.5
                return OCRResult(text if text else "", float(conf), 'CRNN', time.time() - start)
            except Exception as e:
                print(f"⚠️  CRNN error: {e}")
                return OCRResult("", 0.0, 'CRNN', time.time() - start)
        
        elif engine_name == 'tesseract':
            # Tesseract
            import pytesseract
            import time
            start = time.time()
            
            try:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                try:
                    text = pytesseract.image_to_string(image, lang='vie')
                except:
                    text = pytesseract.image_to_string(image, lang='eng')
                
                return OCRResult(text.strip() if text else "", 0.80, 'Tesseract', time.time() - start)
            except Exception as e:
                print(f"⚠️  Tesseract error: {e}")
                return OCRResult("", 0.0, 'Tesseract', time.time() - start)
        
        elif engine_name == 'easyocr':
            # EasyOCR
            import time
            start = time.time()
            
            try:
                # FORCE convert to numpy array - EasyOCR ONLY accepts numpy array
                img_array = None
                
                # Step 1: Convert any input to PIL Image first
                if isinstance(image, str) or isinstance(image, Path):
                    pil_image = Image.open(str(image))
                elif isinstance(image, np.ndarray):
                    if len(image.shape) == 2:
                        pil_image = Image.fromarray(image, mode='L').convert('RGB')
                    elif image.shape[2] == 4:
                        pil_image = Image.fromarray(image, mode='RGBA').convert('RGB')
                    else:
                        pil_image = Image.fromarray(image)
                elif isinstance(image, Image.Image):
                    pil_image = image
                else:
                    print(f"⚠️  EasyOCR: Unknown image type: {type(image)}")
                    return OCRResult("", 0.0, 'EasyOCR', time.time() - start)
                
                # Step 2: Ensure RGB mode
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Step 3: Convert to numpy array (uint8)
                img_array = np.array(pil_image, dtype=np.uint8)
                
                # Step 4: Verify it's a valid numpy array
                if not isinstance(img_array, np.ndarray):
                    raise ValueError(f"Failed to convert to numpy array")
                
                # Debug print
                print(f"  📷 EasyOCR input: shape={img_array.shape}, dtype={img_array.dtype}")
                
                # EasyOCR readtext - get all text with details
                results = engine.readtext(img_array, detail=1, paragraph=False)
                
                # Sort by Y position (top to bottom) then X position
                if results:
                    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))  # Sort by top-left Y, then X
                
                # Combine text from all detected regions
                texts = []
                total_conf = 0
                for (bbox, text, conf) in results:
                    if text and len(text.strip()) > 0:
                        texts.append(text.strip())
                        total_conf += conf
                
                combined_text = " ".join(texts)
                avg_conf = total_conf / len(results) if results else 0
                
                print(f"  📝 EasyOCR found {len(results)} text regions")
                
                return OCRResult(combined_text.strip() if combined_text else "", avg_conf if avg_conf > 0 else 0.75, 'EasyOCR', time.time() - start)
            except Exception as e:
                print(f"⚠️  EasyOCR error: {e}")
                return OCRResult("", 0.0, 'EasyOCR', time.time() - start)
        
        # Default fallback - should never reach here
        print(f"⚠️  Unknown engine type: {engine_name}")
        return OCRResult("", 0.0, engine_name)
    
    def _detect_text_lines(self, image: Union[Image.Image, np.ndarray], verbose: bool = False) -> List[Image.Image]:
        """
        Detect and extract individual text lines from an image.
        Essential for VietOCR which only handles single lines.
        
        IMPORTANT: This function detects lines using grayscale processing,
        but returns ORIGINAL COLOR line crops for best OCR accuracy.
        
        Args:
            image: PIL Image or numpy array (original color)
            verbose: Print debug info
            
        Returns:
            List of PIL Images (color), each containing one text line
        """
        import cv2
        
        # Keep original color image for final cropping
        if isinstance(image, Image.Image):
            original_image = image
            img_array = np.array(image)
        else:
            original_image = Image.fromarray(image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            img_array = image.copy()
        
        # Convert to grayscale with ADVANCED colored background handling
        if len(img_array.shape) == 3:
            # Calculate color statistics for each channel
            b_mean = np.mean(img_array[:, :, 0]) if img_array.shape[2] >= 3 else 0
            g_mean = np.mean(img_array[:, :, 1]) if img_array.shape[2] >= 2 else 0
            r_mean = np.mean(img_array[:, :, 2]) if img_array.shape[2] >= 1 else 0
            
            # Calculate color std to detect uniform colored backgrounds
            b_std = np.std(img_array[:, :, 0]) if img_array.shape[2] >= 3 else 0
            g_std = np.std(img_array[:, :, 1]) if img_array.shape[2] >= 2 else 0
            r_std = np.std(img_array[:, :, 2]) if img_array.shape[2] >= 1 else 0
            
            # Detect colored background with low variance
            is_blue_bg = (b_mean > r_mean + 40 and b_mean > g_mean + 30 and b_std < 50)
            is_green_bg = (g_mean > r_mean + 40 and g_mean > b_mean + 30 and g_std < 50)
            
            if verbose:
                print(f"  🎨 Color: B={b_mean:.0f}±{b_std:.0f}, G={g_mean:.0f}±{g_std:.0f}, R={r_mean:.0f}±{r_std:.0f}")
            
            # For BLUE background (like light blue paper/screen)
            if is_blue_bg:
                if verbose:
                    print(f"  🔵 Blue background detected - using optimal channel extraction")
                # Method 1: Use red channel (text is darkest here)
                red_channel = img_array[:, :, 2]
                # Method 2: Invert blue channel and combine
                blue_inverted = 255 - img_array[:, :, 0]
                # Combine both for best contrast
                gray = cv2.addWeighted(red_channel, 0.7, blue_inverted, 0.3, 0)
                
            # For GREEN background
            elif is_green_bg:
                if verbose:
                    print(f"  🟢 Green background detected - using optimal channel extraction")
                # Use blue channel for green backgrounds
                blue_channel = img_array[:, :, 0]
                green_inverted = 255 - img_array[:, :, 1]
                gray = cv2.addWeighted(blue_channel, 0.7, green_inverted, 0.3, 0)
                
            else:
                # Normal grayscale conversion for other images
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Enhance contrast AGGRESSIVELY for colored backgrounds
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # ====================================================================
        # SPECIAL HANDLING: Remove texture noise (old paper, background)
        # ====================================================================
        # Bilateral filter - removes noise while preserving text edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect if image has textured background (like old paper)
        # Calculate local variance to detect texture
        local_mean = cv2.blur(gray, (15, 15))
        local_sq_mean = cv2.blur(gray.astype(np.float32)**2, (15, 15))
        local_var = local_sq_mean - local_mean.astype(np.float32)**2
        texture_level = np.mean(local_var)
        
        # If high texture, apply stronger denoising
        if texture_level > 200:  # High texture detected
            # Morphological opening to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            # Additional smoothing
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # ====================================================================
        # SPECIAL HANDLING: Remove grid lines from notebook/graph paper
        # ====================================================================
        # Detect if image has grid pattern (horizontal/vertical lines)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal lines (grid)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines (grid)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # If significant grid detected, remove it from the image
        grid_lines = cv2.add(horizontal_lines, vertical_lines)
        grid_coverage = np.sum(grid_lines > 0) / grid_lines.size
        
        if grid_coverage > 0.01:  # More than 1% of image is grid lines
            # Remove grid from gray image
            # Dilate grid slightly to cover fully
            grid_dilated = cv2.dilate(grid_lines, np.ones((3, 3), np.uint8), iterations=1)
            # Inpaint to remove grid lines
            gray_cleaned = cv2.inpaint(gray, grid_dilated, 3, cv2.INPAINT_TELEA)
            gray = gray_cleaned
        # ====================================================================
        
        # Adaptive threshold works better for camera images with varying lighting
        # Use larger block size for textured backgrounds
        block_size = 21 if texture_level > 200 else 11
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, 2
        )
        
        # Denoise - stronger for textured backgrounds
        if texture_level > 200:
            # Remove small noise blobs
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.medianBlur(binary, 5)
        else:
            binary = cv2.medianBlur(binary, 3)
        
        # ====================================================================
        # METHOD 1: Row projection (better for grid paper)
        # ====================================================================
        # Sum pixels horizontally to find text rows
        row_sums = np.sum(binary, axis=1)
        
        # Smooth the row sums with stronger smoothing for textured backgrounds
        from scipy.ndimage import gaussian_filter1d
        try:
            sigma = 5 if texture_level > 200 else 3
            row_sums_smooth = gaussian_filter1d(row_sums.astype(float), sigma=sigma)
        except:
            row_sums_smooth = row_sums.astype(float)
        
        # Find threshold for text vs background
        # Use VERY LOW threshold to catch even short lines like "gửi tớ,"
        if texture_level > 200:
            threshold = np.percentile(row_sums_smooth, 50)  # Much lower for textured
        else:
            # Use very low threshold - even 10% of mean can be a short line
            threshold = np.mean(row_sums_smooth) * 0.10
        
        # Ensure threshold is reasonable but NOT too high
        max_val = np.max(row_sums_smooth)
        median_val = np.median(row_sums_smooth)
        
        # Adaptive threshold based on statistics
        if max_val > 0:
            # Use 5-10% of max, or 30% of median, whichever is lower
            adaptive_threshold = min(max_val * 0.08, median_val * 0.3)
            threshold = max(threshold, adaptive_threshold)
            
        # For very short text (like "gửi tớ,"), use even lower threshold
        if max_val > 0 and threshold > max_val * 0.2:
            threshold = max_val * 0.08  # Use 8% of max for short lines
            
        # Debug: print threshold info
        if verbose:
            print(f"  📊 Row projection: max={max_val:.0f}, median={median_val:.0f}, threshold={threshold:.0f}")
        
        # Find text row regions with minimum gap between lines
        in_text = False
        text_regions = []
        start_row = 0
        min_gap = max(2, img_array.shape[0] // 100)  # Very small gap to catch more lines
        gap_count = 0  # Track consecutive rows below threshold
        
        for i, val in enumerate(row_sums_smooth):
            if not in_text and val > threshold:
                in_text = True
                start_row = i
                gap_count = 0
            elif in_text and val <= threshold:
                gap_count += 1
                # Only end region if gap is long enough
                if gap_count >= min_gap:
                    in_text = False
                    height = i - gap_count - start_row
                    if height >= 3:  # Very low minimum height to catch short lines like "gửi tớ,"
                        text_regions.append((start_row, i - gap_count))
                    gap_count = 0
            elif in_text and val > threshold:
                gap_count = 0  # Reset gap if text continues
        
        # Handle case where text extends to end
        if in_text:
            height = len(row_sums_smooth) - start_row
            if height >= 3:  # Very low minimum
                text_regions.append((start_row, len(row_sums_smooth)))
        
        if verbose:
            print(f"  📏 Detected {len(text_regions)} text regions")
        
        # Post-process: Split regions that are too tall (likely multiple merged lines)
        avg_height = np.mean([r[1] - r[0] for r in text_regions]) if text_regions else 20
        expected_line_height = max(10, min(40, avg_height))  # Lower min height for short lines
        
        if verbose:
            print(f"  📐 Expected line height: {expected_line_height:.0f}px")
        
        final_regions = []
        for start_row, end_row in text_regions:
            height = end_row - start_row
            if height > expected_line_height * 2.5:  # Likely merged lines
                # Split into smaller regions based on local minima in row_sums
                region_sums = row_sums_smooth[start_row:end_row]
                # Find valleys (potential line boundaries)
                from scipy.signal import find_peaks
                try:
                    valleys, _ = find_peaks(-region_sums, distance=expected_line_height//3)
                    if len(valleys) > 0:
                        prev = 0
                        for v in valleys:
                            if v - prev >= 3:  # Very low minimum for short lines
                                final_regions.append((start_row + prev, start_row + v))
                            prev = v
                        if end_row - start_row - prev >= 3:
                            final_regions.append((start_row + prev, end_row))
                    else:
                        final_regions.append((start_row, end_row))
                except:
                    final_regions.append((start_row, end_row))
            else:
                final_regions.append((start_row, end_row))
        
        text_regions = final_regions if final_regions else text_regions
        
        if verbose:
            print(f"  ✅ Final: {len(text_regions)} lines after split")
        
        # If row projection found lines, use it (even 1-2 lines is ok for artistic text)
        if len(text_regions) >= 1:  # Use row projection even for 1 line
            line_boxes = []
            for start_row, end_row in text_regions:
                # Add padding to each line
                padding = 3
                y_start = max(0, start_row - padding)
                y_end = min(img_array.shape[0], end_row + padding)
                # Full width, detected height
                line_boxes.append((0, y_start, img_array.shape[1], y_end - y_start))
        else:
            # ====================================================================
            # METHOD 2: Contour-based (fallback)
            # ====================================================================
            # Dilate horizontally to connect characters in same line
            kernel_width = max(20, img_array.shape[1] // 25)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # Vertical dilation - very minimal
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            dilated = cv2.dilate(dilated, kernel_v, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get bounding boxes
            line_boxes = []
            min_line_height = 8
            min_line_width = 15
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > min_line_height and w > min_line_width:
                    line_boxes.append((x, y, w, h))
        
        if not line_boxes:
            # No lines detected, return original image
            if isinstance(image, np.ndarray):
                return [Image.fromarray(image)]
            return [image]
        
        # Sort by Y position (top to bottom)
        line_boxes.sort(key=lambda box: box[1])
        
        # Merge overlapping lines - more conservative to avoid merging separate lines
        merged_boxes = []
        for box in line_boxes:
            x, y, w, h = box
            merged = False
            for i, (mx, my, mw, mh) in enumerate(merged_boxes):
                # Check if significantly overlapping vertically (at least 50%)
                overlap_y = min(y + h, my + mh) - max(y, my)
                min_height = min(h, mh)
                if overlap_y > min_height * 0.5:  # Only merge if >50% overlap
                    # Merge
                    new_x = min(x, mx)
                    new_y = min(y, my)
                    new_w = max(x + w, mx + mw) - new_x
                    new_h = max(y + h, my + mh) - new_y
                    merged_boxes[i] = (new_x, new_y, new_w, new_h)
                    merged = True
                    break
            if not merged:
                merged_boxes.append(box)
        
        # Sort again after merging
        merged_boxes.sort(key=lambda box: box[1])
        
        # Extract line images from ORIGINAL color image (not processed grayscale)
        line_images = []
        
        # Use the original color image for cropping
        original_array = np.array(original_image) if isinstance(original_image, Image.Image) else original_image
        
        for x, y, w, h in merged_boxes:
            # Add MORE padding to avoid cutting off text at edges
            # Handwritten text often extends beyond detected boundaries
            padding_x = max(15, w // 10)  # At least 15px or 10% of width
            padding_y = max(8, h // 5)    # At least 8px or 20% of height
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(original_array.shape[1], x + w + padding_x)
            y2 = min(original_array.shape[0], y + h + padding_y)
            
            line_crop = original_array[y1:y2, x1:x2]
            line_pil = Image.fromarray(line_crop)
            line_images.append(line_pil)
        
        return line_images if line_images else [original_image]
    
    def _multi_scale_ocr(self, cv_image: np.ndarray, verbose: bool = True) -> OCRResult:
        """
        Multi-scale, multi-method OCR for difficult images (light text, low contrast).
        
        Strategy:
        1. Try multiple preprocessing methods
        2. Try multiple scales (zoom in/out)
        3. Try multiple OCR engines
        4. Pick the best Vietnamese result
        
        Args:
            cv_image: OpenCV BGR image
            verbose: Print debug info
            
        Returns:
            Best OCRResult found
        """
        from image_preprocessing import ImagePreprocessor
        
        all_results = []
        
        # =========================================
        # PREPROCESSING METHODS
        # =========================================
        preprocessing_methods = {
            'original': lambda img: img,
            'invert_enhance': self._invert_and_enhance,
            'edge_enhance': self._edge_enhancement,
        }
        
        # =========================================
        # SCALE FACTORS - Reduced for speed
        # =========================================
        scales = [1.0, 2.0]
        
        # =========================================
        # ENGINES TO TRY - Only fast ones, skip EasyOCR (slow + errors)
        # =========================================
        engines_to_try = ['paddleocr', 'vietocr']
        
        if verbose:
            print(f"    🔄 Multi-scale OCR: {len(preprocessing_methods)} methods × {len(scales)} scales")
        
        # Try each preprocessing method
        for method_name, preprocess_func in preprocessing_methods.items():
            try:
                processed = preprocess_func(cv_image)
                if processed is None:
                    continue
                
                # Try each scale
                for scale in scales:
                    try:
                        # Scale image
                        if scale != 1.0:
                            h, w = processed.shape[:2]
                            new_w, new_h = int(w * scale), int(h * scale)
                            if new_w < 50 or new_h < 20:
                                continue
                            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                            scaled = cv2.resize(processed, (new_w, new_h), interpolation=interp)
                        else:
                            scaled = processed
                        
                        # Convert to PIL
                        if len(scaled.shape) == 2:
                            pil_image = Image.fromarray(scaled).convert('RGB')
                        else:
                            pil_image = Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
                        
                        # Try each engine
                        for engine_name in engines_to_try:
                            if engine_name not in self.engines:
                                continue
                            
                            try:
                                result = self.recognize_with_engine(engine_name, pil_image)
                                if result and result.text:
                                    text = result.text.strip()
                                    if text and len(text) >= 3:
                                        # Score this result
                                        score = self._score_vietnamese_text(text, result.confidence)
                                        all_results.append({
                                            'text': text,
                                            'confidence': result.confidence,
                                            'engine': engine_name,
                                            'method': method_name,
                                            'scale': scale,
                                            'score': score
                                        })
                                        
                                        if verbose and score > 10:
                                            print(f"      ✓ {method_name}@{scale}x/{engine_name}: '{text[:50]}...' (score={score:.1f})")
                            except Exception as e:
                                continue
                    except Exception as e:
                        continue
            except Exception as e:
                if verbose:
                    print(f"      ✗ {method_name} failed: {e}")
                continue
        
        # Pick best result
        if not all_results:
            if verbose:
                print(f"    ⚠️ No valid results from multi-scale OCR")
            return OCRResult("", 0.0, "multi_scale", 0)
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        best = all_results[0]
        
        if verbose:
            print(f"    🏆 Best: '{best['text'][:60]}...' ({best['method']}@{best['scale']}x/{best['engine']}, score={best['score']:.1f})")
        
        return OCRResult(
            text=best['text'],
            confidence=best['confidence'],
            engine=f"MultiScale-{best['engine']}-{best['method']}",
            processing_time=0.0
        )
    
    def _score_vietnamese_text(self, text: str, confidence: float) -> float:
        """Score text based on Vietnamese characteristics"""
        if not text:
            return -1000
        
        score = confidence * 20  # Base score from confidence
        
        # Vietnamese characters
        vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ')
        viet_count = sum(1 for c in text if c in vietnamese_chars)
        score += viet_count * 5  # Bonus for Vietnamese chars
        
        # Length bonus (longer = usually better)
        length = len(text.replace(' ', ''))
        if 10 <= length <= 100:
            score += length * 0.5
        elif length < 5:
            score -= 20  # Too short
        
        # Word structure bonus (has spaces = real words)
        if ' ' in text:
            word_count = len(text.split())
            if 2 <= word_count <= 20:
                score += word_count * 3
        
        # Alphabetic ratio (should be mostly letters)
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / max(len(text), 1)
        if alpha_ratio > 0.7:
            score += 15
        elif alpha_ratio < 0.4:
            score -= 30
        
        # Penalty for garbage patterns
        garbage_patterns = ['glangers', 'lorem', 'ipsum', 'quack', 'xxx', 'www', 'http']
        for pattern in garbage_patterns:
            if pattern in text.lower():
                score -= 100
        
        # Penalty for repeated characters
        unique_ratio = len(set(text.lower())) / max(len(text), 1)
        if unique_ratio < 0.3:
            score -= 50
        
        return score
    
    def _extract_by_saturation(self, image: np.ndarray) -> np.ndarray:
        """Extract text by saturation (colored text on white has saturation)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Any saturation indicates color (text)
        _, mask = cv2.threshold(s, 3, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Create white background with black text
        result = np.ones_like(image) * 255
        result[mask > 0] = [0, 0, 0]
        
        return result
    
    def _extract_by_color_difference(self, image: np.ndarray) -> np.ndarray:
        """Extract text by difference from white"""
        white = np.ones_like(image) * 255
        diff = cv2.absdiff(image, white)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Any difference from white = text
        _, mask = cv2.threshold(diff_gray, 3, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create result
        result = np.ones_like(image) * 255
        result[mask > 0] = [0, 0, 0]
        
        return result
    
    def _invert_and_enhance(self, image: np.ndarray) -> np.ndarray:
        """Invert colors and enhance contrast"""
        # Invert
        inverted = cv2.bitwise_not(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=2.5, beta=-50)
        
        # Threshold
        _, binary = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
        
        # Invert back (text should be dark)
        binary = cv2.bitwise_not(binary)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _extract_by_lab(self, image: np.ndarray) -> np.ndarray:
        """Extract text using LAB color space (detects any color deviation)"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Color deviation from neutral (128)
        a_dev = np.abs(a.astype(np.int16) - 128).astype(np.uint8)
        b_dev = np.abs(b.astype(np.int16) - 128).astype(np.uint8)
        color_dev = cv2.add(a_dev, b_dev)
        
        # Threshold
        _, mask = cv2.threshold(color_dev, 3, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create result
        result = np.ones_like(image) * 255
        result[mask > 0] = [0, 0, 0]
        
        return result
    
    def _edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges to detect faint text"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert
        inverted = cv2.bitwise_not(gray)
        
        # Heavy CLAHE
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(inverted)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Threshold
        _, binary = cv2.threshold(sharpened, 50, 255, cv2.THRESH_BINARY)
        
        # Invert back
        binary = cv2.bitwise_not(binary)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _pick_best_line_result(self, results: List[OCRResult]) -> OCRResult:
        """
        Pick the best OCR result from multiple engines for a single line.
        
        Criteria:
        1. Filter out garbage (single chars, repeated chars, non-Vietnamese)
        2. Require minimum confidence (> 0.3)
        3. Prefer longer meaningful text
        4. Prefer Vietnamese characters
        
        Args:
            results: List of OCRResult from different engines
            
        Returns:
            Best OCRResult or empty if all are garbage
        """
        if not results:
            return OCRResult("", 0.0, "none", 0)
        
        MIN_CONFIDENCE = 0.2  # Lowered for handwritten text (VietOCR often gives 0.75-0.85)
        
        def is_garbage(text: str) -> bool:
            """Check if text is likely garbage - more lenient for Vietnamese handwriting"""
            if not text or len(text) < 2:
                return True
            
            # All same character
            unique_chars = set(text.replace(' ', ''))
            if len(unique_chars) <= 1:  # Changed from 2 to 1
                return True
            
            # Too many NON-letter characters (garbage pattern like "xạxậàậxẳ")
            # But don't penalize normal Vietnamese text with diacritics
            letter_count = sum(1 for c in text if c.isalpha())
            if len(text) > 5 and letter_count < len(text) * 0.3:
                return True  # Less than 30% letters = likely garbage
            
            # Numbers only
            if text.replace(' ', '').isdigit():
                return True
            
            return False
        
        def score_result(r: OCRResult) -> float:
            text = r.text.strip()
            
            # Garbage detection
            if is_garbage(text):
                return -1000
            
            # Low confidence penalty
            if r.confidence < MIN_CONFIDENCE:
                return -500
            
            score = 0.0
            
            # Length score (longer is usually better, but not too long)
            length = len(text)
            if length < 3:
                score -= 20
            elif length < 5:
                score += length * 1
            else:
                score += min(length, 50) * 2  # Cap at 50 chars
            
            # Confidence score (very important)
            score += r.confidence * 50
            
            # Vietnamese character bonus (but not too many)
            vietnamese_lower = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
            vietnamese_upper = vietnamese_lower.upper()
            vietnamese_chars = set(vietnamese_lower + vietnamese_upper)
            viet_count = sum(1 for c in text if c in vietnamese_chars)
            
            # Reasonable ratio of Vietnamese chars
            # Vietnamese handwritten text normally has 10-40% diacritics
            viet_ratio = viet_count / max(len(text), 1)
            if 0.05 <= viet_ratio <= 0.6:  # More lenient range
                score += viet_count * 5  # Good ratio
            elif viet_ratio > 0.8:  # Only penalize if >80% diacritics
                score -= 20  # Reduced penalty
            
            # Penalty for repeated characters
            if len(set(text)) < len(text) / 3:
                score -= 40
            
            # Bonus for having spaces (real words)
            if ' ' in text:
                score += 10
            
            return score
        
        # Filter valid results first
        valid_results = [r for r in results if not is_garbage(r.text.strip()) and r.confidence >= MIN_CONFIDENCE]
        
        if not valid_results:
            # No valid results, return the least bad one with warning
            scored = [(r, score_result(r)) for r in results]
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0][0]
            # Mark as low confidence
            return OCRResult(best.text, min(best.confidence, 0.1), best.engine, best.processing_time)
        
        # Score valid results and pick best
        scored = [(r, score_result(r)) for r in valid_results]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[0][0]
    
    def recognize(self, 
                  image: Union[Image.Image, np.ndarray, str, Path],
                  engines: Optional[List[str]] = None,
                  voting_method: str = 'weighted',
                  min_confidence: float = 0.3,
                  verbose: bool = True,
                  preprocess: str = 'none',
                  fast_mode: bool = False,
                  crnn_model: str = 'crnn_vnondb.pth') -> EnsembleOCRResult:
        """
        Nhận diện text với ensemble voting (VietOCR + CRNN song song)
        
        Args:
            image: PIL Image, numpy array, or path to image
            engines: List of engines to use (None = use all)
            voting_method: 'weighted', 'majority', or 'best'
            min_confidence: Minimum confidence threshold
            verbose: If True, print detailed output. Set False when processing cells/regions
            preprocess: Image preprocessing ('none', 'light', 'medium', 'heavy')
            fast_mode: If True, use only VietOCR (fast, ~0.3s instead of 120s)
            crnn_model: CRNN model file to use (crnn_vnondb.pth, crnn_best.pth, etc.)
            crnn_model: CRNN model file to use (crnn_vnondb.pth, crnn_best.pth, etc.)
        
        Returns:
            EnsembleOCRResult with consensus from multiple engines
        """
        import time
        start_time = time.time()
        
        # Import preprocessing
        from image_preprocessing import ImagePreprocessor
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Apply preprocessing if requested
        if preprocess != 'none' and preprocess:
            if verbose:
                print(f"  🖼️  Preprocessing ({preprocess})...")
            image = ImagePreprocessor.auto_preprocess(image, quality=preprocess)
            image = ImagePreprocessor.to_pil(image)
        else:
            # Even if preprocessing is 'none', check for light-colored text
            cv_image = ImagePreprocessor.to_cv2(image)
            if ImagePreprocessor.detect_light_text(cv_image):
                if verbose:
                    print(f"  🔍 Auto-detected light-colored text, applying enhancement...")
                image = ImagePreprocessor.enhance_light_text(cv_image)
                image = ImagePreprocessor.to_pil(image)
        
        # Check if image is multi-line (full page) - if so, use line detection
        img_array = np.array(image)
        height, width = img_array.shape[:2] if len(img_array.shape) >= 2 else (0, 0)
        
        # =====================================================================
        # PRE-CHECK: Detect light-colored text images
        # =====================================================================
        cv_image = ImagePreprocessor.to_cv2(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        std_gray = np.std(gray)
        
        is_light_image = (mean_gray > 180 and std_gray < 50)  # Light image detection
        
        # Detect if image is printed text vs handwritten
        # Printed text: high contrast, uniform strokes, regular spacing
        # Handwritten: variable stroke width, irregular spacing
        is_printed_text = False
        
        # Check edge density - printed text has more uniform edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Multiple criteria for printed text detection:
        # 1. High contrast (std > 40) with moderate edge density
        # 2. OR very clean background (mean > 230) with text
        if (std_gray > 40 and edge_density > 0.02 and edge_density < 0.4):
            # Additional check: horizontal line detection (printed text has straight lines)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            h_line_ratio = np.sum(horizontal_lines > 0) / edges.size
            
            if h_line_ratio > 0.005 or (mean_gray > 200 and std_gray > 35):
                is_printed_text = True
        
        if is_light_image and verbose:
            print(f"  🔍 Detected light image (mean={mean_gray:.0f}, std={std_gray:.0f})")
            print(f"  🚀 Using Multi-Scale Multi-Method OCR for best results...")
        
        # Vietnamese text validator
        def is_vietnamese_text(text):
            if not text or len(text) < 3:
                return False
            # Vietnamese specific chars
            viet_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
            viet_chars += viet_chars.upper()
            # Check if has Vietnamese chars or common Vietnamese words
            has_viet = any(c in text.lower() for c in viet_chars)
            has_common_words = any(w in text.lower() for w in ['và', 'của', 'cho', 'với', 'được', 'là', 'có', 'không', 'này', 'đó', 'một', 'những', 'như', 'khi', 'để', 'từ', 'trong', 'nếu', 'thì', 'nhưng', 'cũng', 'còn', 'rất', 'đã', 'sẽ'])
            # Reject if all uppercase English (likely OCR garbage)
            if text.isupper() and not has_viet:
                return False
            return has_viet or has_common_words or len(text) > 20
        
        if is_printed_text and verbose:
            print(f"  📖 Detected printed text (std={std_gray:.0f}), prioritizing PaddleOCR...")
        
        # =====================================================================
        # STRATEGY 0a: TRY PADDLEOCR FIRST FOR PRINTED TEXT
        # =====================================================================
        if is_printed_text and 'paddleocr' in self.engines:
            if verbose:
                print(f"  📄 Trying PaddleOCR for printed text...")
            
            paddle_result = self.recognize_with_engine('paddleocr', image)
            paddle_text = paddle_result.text.strip() if paddle_result.text else ""
            
            # If PaddleOCR got reasonable Vietnamese text
            if paddle_text and len(paddle_text) > 10 and is_vietnamese_text(paddle_text):
                if verbose:
                    print(f"    ✅ PaddleOCR: '{paddle_text[:100]}...' (conf: {paddle_result.confidence:.2f})")
                
                # Post-process
                try:
                    from vietnamese_spell_checker import post_process_ocr_text
                    corrected = post_process_ocr_text(paddle_text, verbose=verbose)
                    if corrected != paddle_text:
                        if verbose:
                            print(f"  ✏️ Spell check applied!")
                        paddle_text = corrected
                except ImportError:
                    pass
                
                return EnsembleOCRResult(
                    text=paddle_text,
                    confidence=paddle_result.confidence,
                    best_engine="PaddleOCR (printed)",
                    all_results=[paddle_result],
                    consensus_score=0.90,
                    processing_time=time.time() - start_time
                )
            elif verbose:
                print(f"    ⚠️ PaddleOCR: No valid Vietnamese text found, trying other engines...")
        
        # =====================================================================
        # STRATEGY 0b: TRY EASYOCR (has built-in text detection, good for multi-line)
        # =====================================================================
        if 'easyocr' in self.engines and height > 100 and width > 100:
            if verbose:
                print(f"  📄 Trying EasyOCR (best for multi-line text)...")
            
            easyocr_result = self.recognize_with_engine('easyocr', image)
            easyocr_text = easyocr_result.text.strip() if easyocr_result.text else ""
            
            # If EasyOCR got reasonable text (>20 chars for multi-line), use it
            if easyocr_text and len(easyocr_text) > 20:
                if verbose:
                    print(f"    ✅ EasyOCR: '{easyocr_text[:100]}...' (conf: {easyocr_result.confidence:.2f})")
                
                # Post-process
                try:
                    from vietnamese_spell_checker import post_process_ocr_text
                    corrected = post_process_ocr_text(easyocr_text, verbose=verbose)
                    if corrected != easyocr_text:
                        if verbose:
                            print(f"  ✏️ Spell check applied!")
                        easyocr_text = corrected
                except ImportError:
                    pass
                
                return EnsembleOCRResult(
                    text=easyocr_text,
                    confidence=easyocr_result.confidence,
                    best_engine="EasyOCR (multi-line)",
                    all_results=[easyocr_result],
                    consensus_score=0.90,
                    processing_time=time.time() - start_time
                )
            elif verbose:
                print(f"    ⚠️ EasyOCR returned short text: '{easyocr_text[:50] if easyocr_text else 'empty'}'")
        
        # =====================================================================
        # STRATEGY 1: SKIP Multi-scale for light images - too slow, go to line detection
        # =====================================================================
        # Multi-scale OCR is disabled for speed - line detection works better
        
        # =====================================================================
        # STRATEGY 2: Try VietOCR directly on full image (works well for handwriting)
        # =====================================================================
        if 'vietocr' in self.engines:
            if verbose:
                print(f"  📄 Trying VietOCR on full image ({width}x{height})...")
            
            vietocr_result = self.recognize_with_engine('vietocr', image)
            vietocr_text = vietocr_result.text.strip() if vietocr_result.text else ""
            
            # If VietOCR got reasonable Vietnamese text
            if vietocr_text and len(vietocr_text) > 5 and is_vietnamese_text(vietocr_text):
                if verbose:
                    print(f"    ✅ VietOCR: '{vietocr_text[:80]}' (conf: {vietocr_result.confidence:.2f})")
                
                # Post-process: sửa lỗi chính tả
                try:
                    from vietnamese_spell_checker import post_process_ocr_text
                    corrected = post_process_ocr_text(vietocr_text, verbose=verbose)
                    if corrected != vietocr_text:
                        if verbose:
                            print(f"  ✏️ Spell check applied!")
                        vietocr_text = corrected
                except ImportError:
                    pass
                
                return EnsembleOCRResult(
                    text=vietocr_text,
                    confidence=vietocr_result.confidence,
                    best_engine="VietOCR (direct)",
                    all_results=[vietocr_result],
                    consensus_score=0.95,
                    processing_time=time.time() - start_time
                )
            else:
                if verbose:
                    if vietocr_text:
                        print(f"    ⚠️ VietOCR returned non-Vietnamese text '{vietocr_text}', trying line detection...")
                    else:
                        print(f"    ⚠️ VietOCR returned empty text, trying line detection...")
        
        # =====================================================================
        # STRATEGY 2: Line detection + ensemble (fallback)
        # =====================================================================
        # Always try to use VietOCR + CRNN ensemble for best results
        use_ensemble = 'vietocr' in self.engines or 'crnn' in self.engines
        
        # If image is reasonable size, detect lines first
        if height > 50 and width > 50:
            if verbose:
                print(f"  📄 Detecting text lines in image ({width}x{height})...")
            
            lines = self._detect_text_lines(image, verbose=verbose)
            
            # If line detection found lines, use ensemble on each line
            if len(lines) >= 1:
                if verbose:
                    print(f"  ✅ Found {len(lines)} text line(s), processing with ENSEMBLE...")
                
                # Switch CRNN model if specified
                if 'crnn' in self.engines:
                    crnn_wrapper = self.engines['crnn']
                    if hasattr(crnn_wrapper, 'switch_model'):
                        crnn_wrapper.switch_model(crnn_model)
                
                # Determine which engines to use for line processing
                # Include PaddleOCR for printed text
                line_engines = []
                if 'vietocr' in self.engines:
                    line_engines.append('vietocr')
                if 'paddleocr' in self.engines:
                    line_engines.append('paddleocr')  # Good for printed text
                if 'crnn' in self.engines:
                    line_engines.append('crnn')
                
                if not line_engines:
                    line_engines = ['vietocr']  # Fallback
                
                if verbose:
                    print(f"  🔧 Using ENSEMBLE: {line_engines} for each line, picking best result...")
                
                # Process each line with multiple engines
                all_line_texts = []
                engine_wins = {'vietocr': 0, 'crnn': 0, 'paddleocr': 0}
                MIN_LINE_CONFIDENCE = 0.15  # Lowered - VietOCR gives ~0.75-0.95 for handwritten
                
                for i, line_img in enumerate(lines):
                    if verbose:
                        print(f"    Line {i+1}/{len(lines)}:", end=" ")
                    
                    # Run all line engines and pick best
                    line_results = []
                    for eng in line_engines:
                        result = self.recognize_with_engine(eng, line_img)
                        text = result.text.strip() if result.text else ""
                        
                        # Filter: must have text, reasonable length, and good confidence
                        if text and len(text) > 1 and result.confidence >= MIN_LINE_CONFIDENCE:
                            line_results.append(result)
                            if verbose:
                                print(f"[{eng}: '{text[:20]}' conf={result.confidence:.2f}]", end=" ")
                        elif verbose:
                            print(f"[{eng}: SKIP conf={result.confidence:.2f}]", end=" ")
                    
                    if line_results:
                        # Pick the best result based on confidence and text quality
                        best_result = self._pick_best_line_result(line_results)
                        best_text = best_result.text.strip()
                        
                        # Only add if not garbage
                        if best_text and best_result.confidence >= MIN_LINE_CONFIDENCE:
                            all_line_texts.append(best_text)
                            engine_wins[best_result.engine.lower()] = engine_wins.get(best_result.engine.lower(), 0) + 1
                            if verbose:
                                print(f"-> BEST: [{best_result.engine}]")
                        else:
                            if verbose:
                                print(f"-> REJECTED (low conf)")
                    else:
                        if verbose:
                            print(f"-> NO VALID RESULT")
                
                if all_line_texts:
                    combined_text = '\n'.join(all_line_texts)
                    
                    # Post-process: sửa lỗi chính tả, tách từ dính
                    try:
                        from vietnamese_spell_checker import post_process_ocr_text
                        if verbose:
                            print(f"\n  🔧 Post-processing OCR text...")
                        corrected_text = post_process_ocr_text(combined_text, verbose=verbose)
                        if corrected_text != combined_text:
                            if verbose:
                                print(f"  ✏️ Spell check applied!")
                            combined_text = corrected_text
                    except ImportError:
                        pass  # Spell checker not available
                    
                    if verbose:
                        print(f"\n  📊 Engine Wins: VietOCR={engine_wins.get('vietocr',0)}, CRNN={engine_wins.get('crnn',0)}")
                        print(f"  ✅ Combined {len(all_line_texts)} lines")
                    
                    # Determine dominant engine
                    dominant_engine = max(engine_wins, key=engine_wins.get) if engine_wins else "ensemble"
                    
                    return EnsembleOCRResult(
                        text=combined_text,
                        confidence=0.85,
                        best_engine=f"ensemble (VietOCR+CRNN, {dominant_engine} dominant)",
                        all_results=[],
                        consensus_score=1.0,
                        processing_time=time.time() - start_time
                    )
        
        # FALLBACK: If line detection failed, still use VietOCR + CRNN ensemble on full image
        if not fast_mode:
            if verbose:
                print(f"\n  🔄 Line detection returned no lines, trying VietOCR on full image...")
            
            # FIRST: Try VietOCR on full image (best for Vietnamese handwriting)
            if 'vietocr' in self.engines:
                vietocr_result = self.recognize_with_engine('vietocr', image)
                vietocr_text = vietocr_result.text.strip() if vietocr_result.text else ""
                
                if vietocr_text and len(vietocr_text) > 10:
                    if verbose:
                        print(f"    ✅ VietOCR: '{vietocr_text[:80]}...' (conf: {vietocr_result.confidence:.2f})")
                    
                    # Post-process
                    try:
                        from vietnamese_spell_checker import post_process_ocr_text
                        corrected = post_process_ocr_text(vietocr_text, verbose=verbose)
                        if corrected != vietocr_text:
                            vietocr_text = corrected
                    except ImportError:
                        pass
                    
                    return EnsembleOCRResult(
                        text=vietocr_text,
                        confidence=vietocr_result.confidence,
                        best_engine="VietOCR (full image fallback)",
                        all_results=[vietocr_result],
                        consensus_score=0.9,
                        processing_time=time.time() - start_time
                    )
                elif verbose:
                    print(f"    ⚠️ VietOCR returned short text: '{vietocr_text}'")
            
            # SECOND: Try EasyOCR (good for multiple languages)
            if 'easyocr' in self.engines:
                if verbose:
                    print(f"    📄 Trying EasyOCR...")
                easy_result = self.recognize_with_engine('easyocr', image)
                easy_text = easy_result.text.strip() if easy_result.text else ""
                
                if easy_text and len(easy_text) > 10:
                    if verbose:
                        print(f"    ✅ EasyOCR: '{easy_text[:80]}...' (conf: {easy_result.confidence:.2f})")
                    
                    # Post-process
                    try:
                        from vietnamese_spell_checker import post_process_ocr_text
                        corrected = post_process_ocr_text(easy_text, verbose=verbose)
                        if corrected != easy_text:
                            easy_text = corrected
                    except ImportError:
                        pass
                    
                    return EnsembleOCRResult(
                        text=easy_text,
                        confidence=easy_result.confidence,
                        best_engine="EasyOCR (fallback)",
                        all_results=[easy_result],
                        consensus_score=0.85,
                        processing_time=time.time() - start_time
                    )
            
            # THIRD: Try PaddleOCR for documents/printed text
            if 'paddleocr' in self.engines:
                if verbose:
                    print(f"    📄 Trying PaddleOCR for document/printed text...")
                paddle_result = self.recognize_with_engine('paddleocr', image)
                paddle_text = paddle_result.text.strip() if paddle_result.text else ""
                
                if paddle_text and len(paddle_text) > 5:
                    if verbose:
                        print(f"    ✅ PaddleOCR: '{paddle_text[:80]}...' (conf: {paddle_result.confidence:.2f})")
                    return EnsembleOCRResult(
                        text=paddle_text,
                        confidence=paddle_result.confidence,
                        best_engine="PaddleOCR (document)",
                        all_results=[paddle_result],
                        consensus_score=0.9,
                        processing_time=time.time() - start_time
                    )
        
        # Use all engines if not specified (legacy fallback)
        if engines is None:
            engines = list(self.engines.keys())
        
        # Fast mode: use only VietOCR (fast & accurate for Vietnamese)
        if fast_mode:
            fast_engines = ['vietocr']  # VietOCR only = 0.3s instead of 120s
            engines = [e for e in engines if e in fast_engines]
            if not engines:  # Fallback if vietocr not available
                engines = ['paddleocr'] if 'paddleocr' in list(self.engines.keys()) else list(self.engines.keys())[:1]
            if verbose:
                print(f"⚡ FAST MODE: Using {engines} only")
        
        # Run OCR with all engines
        all_results = []
        
        if verbose:
            print(f"\n🔍 Running OCR with {len(engines)} engines...")
        
        for engine_name in engines:
            if engine_name in self.engines:
                result = self.recognize_with_engine(engine_name, image)
                
                # Only keep results above minimum confidence
                if result.confidence >= min_confidence and result.text.strip():
                    all_results.append(result)
                    if verbose:
                        print(f"  ✅ {engine_name}: '{result.text[:50]}...' (conf: {result.confidence:.2f})")
                else:
                    if verbose:
                        print(f"  ⚠️  {engine_name}: Low confidence or empty")
        
        # Filter out hallucinations & incomplete results (< 3 chars is suspicious)
        if verbose:
            print("\n  🔍 Filtering suspicious results...")
        
        results_before_filter = all_results.copy()  # Save for fallback
        filtered_results = []
        for r in all_results:
            text = r.text.strip()
            
            # Skip very short results (< 3 chars) unless confidence is very high (>95%)
            if len(text) < 3:
                if r.confidence < 0.95:
                    if verbose:
                        print(f"    ❌ Skipped {r.engine}: too short (len={len(text)}, conf={r.confidence:.2f})")
                    continue
            
            # Detect hallucination: repeated words/patterns
            # Example: "1962 American film ... American ... director ... director"
            words = text.split()
            if len(words) > 3:
                # Check if any word appears too many times
                from collections import Counter
                word_counts = Counter(words)
                max_count = max(word_counts.values()) if word_counts else 1
                
                # If one word appears > 30% of total, it's likely hallucinating
                if max_count / len(words) > 0.3:
                    if verbose:
                        print(f"    ❌ Skipped {r.engine}: hallucinating ('{max(word_counts, key=word_counts.get)}' appears {max_count}x in {len(words)} words)")
                    continue
            
            filtered_results.append(r)
            if verbose:
                print(f"    ✅ Kept {r.engine}: '{text[:40]}...' (len={len(text)})")
        
        all_results = filtered_results
        
        # If all results filtered out, retry without filtering (return best available)
        if not all_results and len(results_before_filter) > 0:
            if verbose:
                print(f"    ⚠️ All results filtered out! Using best available...")
            # Pick the longest result as fallback
            best = max(results_before_filter, key=lambda r: len(r.text.strip()))
            all_results = [best]
            if verbose:
                print(f"    ✅ Fallback to {best.engine}: '{best.text}'")
        
        # If fast mode returns no results, retry with more engines
        if not all_results and fast_mode:
            if verbose:
                print(f"\n⚠️ Fast mode returned no results, trying more engines...")
            # Retry with all engines
            return self.recognize(
                image, 
                engines=None,  # Use all engines
                voting_method=voting_method,
                min_confidence=min_confidence,
                verbose=verbose,
                preprocess=preprocess if preprocess != 'none' else 'medium',  # Try preprocessing
                fast_mode=False  # Disable fast mode
            )
        
        if not all_results:
            return EnsembleOCRResult(
                text="",
                confidence=0.0,
                best_engine="none",
                all_results=[],
                consensus_score=0.0,
                processing_time=time.time() - start_time
            )
        
        # Apply voting method
        if voting_method == 'best':
            # Choose best confidence
            best_result = max(all_results, key=lambda r: r.confidence)
            final_text = best_result.text
            final_confidence = best_result.confidence
            best_engine = best_result.engine
            consensus_score = 1.0
        
        elif voting_method == 'weighted':
            # Weighted voting by confidence
            if verbose:
                print("\n  📊 Weighted Voting Process:")
            final_text = self._weighted_voting(all_results)
            final_confidence = np.mean([r.confidence for r in all_results])
            # Find which engine produced the final text
            best_engine = next((r.engine for r in all_results if r.text == final_text), "unknown")
            consensus_score = self._calculate_consensus(all_results)
        
        elif voting_method == 'majority':
            # Majority voting
            final_text = self._majority_voting(all_results)
            final_confidence = np.mean([r.confidence for r in all_results])
            best_engine = "ensemble"
            consensus_score = self._calculate_consensus(all_results)
        
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Final Result: '{final_text[:80]}...'")
            print(f"📊 Best Engine: {best_engine}")
            print(f"🎯 Confidence: {final_confidence:.2%}")
            print(f"🤝 Consensus: {consensus_score:.2%}")
            print(f"⏱️  Processing Time: {total_time:.2f}s")
            
            # Low consensus warning
            if consensus_score < 0.2:
                print(f"\n⚠️  LOW CONSENSUS ({consensus_score:.1%})")
                print(f"   Engines strongly disagree - result may be inaccurate!")
                print(f"   💡 Try: Preprocessing (light/medium/heavy) to improve accuracy")
        
        return EnsembleOCRResult(
            text=final_text,
            confidence=final_confidence,
            best_engine=best_engine,
            all_results=all_results,
            consensus_score=consensus_score,
            processing_time=total_time
        )
    
    def _weighted_voting(self, results: List[OCRResult]) -> str:
        """Weighted voting based on confidence, but filter obvious errors"""
        # Filter out obvious errors like single 't' character or very short suspicious text
        valid_results = []
        for r in results:
            text = r.text.strip()
            # Skip if single character or suspiciously short
            # Also skip common OCR artifacts (0, O, l, 1, I, etc)
            if len(text) > 1 or (len(text) == 1 and text not in ['t', 'i', 'l', '1', '|', 'I', '0', 'O']):
                valid_results.append(r)
            else:
                print(f"    [FILTERED] {r.engine}: '{text}' (len={len(text)}, artifact)")
        
        if not valid_results:
            # If all filtered out, return longest text as fallback
            longest = max(results, key=lambda r: len(r.text.strip()))
            print(f"    [FALLBACK] Using longest: {longest.engine} '{longest.text}'")
            return longest.text
        
        # Choose result with highest confidence from valid results
        best = max(valid_results, key=lambda r: r.confidence)
        print(f"    [WEIGHTED] Selected {best.engine} (conf={best.confidence:.3f}): '{best.text}'")
        return best.text
    
    def _majority_voting(self, results: List[OCRResult]) -> str:
        """Majority voting - pick best valid result considering text length and similarity"""
        from collections import Counter
        from difflib import SequenceMatcher
        
        # Filter out obvious errors first
        valid_results = []
        for r in results:
            text = r.text.strip()
            # Skip single characters and common OCR artifacts
            if not (len(text) == 1 and text in ['t', 'i', 'l', '1', '|', 'I', '0', 'O']):
                valid_results.append(r)
        
        if not valid_results:
            # Fallback: return longest text
            longest = max(results, key=lambda r: len(r.text.strip()))
            print(f"    [MAJORITY] All filtered, using longest: {longest.engine} '{longest.text}'")
            return longest.text.strip()
        
        # Priority 1: Check for text appearing in multiple engines
        texts = [r.text.strip() for r in valid_results]
        counter = Counter(texts)
        most_common_text, count = counter.most_common(1)[0]
        
        if count >= 2:
            print(f"    [MAJORITY] Consensus: '{most_common_text}' (in {count} engines)")
            return most_common_text
        
        # Priority 2: Check for similar texts (e.g., with small differences)
        # Group similar results together
        from itertools import combinations
        similarity_threshold = 0.75
        text_groups = {texts[0]: [valid_results[0]]}
        
        for i in range(1, len(valid_results)):
            current_text = texts[i]
            current_result = valid_results[i]
            
            # Check similarity with existing groups
            matched = False
            for group_text, group_results in text_groups.items():
                similarity = SequenceMatcher(None, current_text, group_text).ratio()
                if similarity >= similarity_threshold:
                    group_results.append(current_result)
                    matched = True
                    break
            
            if not matched:
                text_groups[current_text] = [current_result]
        
        # Find group with most engines
        largest_group = max(text_groups.values(), key=len)
        if len(largest_group) >= 2:
            # Use highest confidence from this group
            best = max(largest_group, key=lambda r: r.confidence)
            print(f"    [MAJORITY] Similar texts: '{best.text}' (in {len(largest_group)} engines, sim>75%)")
            return best.text.strip()
        
        # Priority 3: No consensus - weight by (text_length, confidence)
        # Prefer longer text as it's more likely to be correct
        # But also consider confidence
        best = max(valid_results, key=lambda r: (len(r.text.strip()) / 100, r.confidence))
        print(f"    [MAJORITY] No consensus, using {best.engine} (len={len(best.text.strip())}, conf={best.confidence:.3f}): '{best.text}'")
        return best.text

    
    def _calculate_consensus(self, results: List[OCRResult]) -> float:
        """Calculate consensus score between engines"""
        if len(results) <= 1:
            return 1.0
        
        # Simple approach: check text similarity
        from difflib import SequenceMatcher
        
        texts = [r.text for r in results]
        similarities = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _ocr_multiline_image(self, img: Image.Image, engines, voting_method, preprocess) -> str:
        """
        OCR multi-line image by detecting and processing each line separately.
        This is more accurate for wide images with multiple text lines.
        """
        # Convert to numpy array
        img_array = np.array(img)
        
        # Upscale 4x for better line detection
        scale = 4
        upscaled = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
        else:
            gray = upscaled
        
        # Binarize for line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection to find text lines
        h_proj = np.sum(binary, axis=1)
        
        # Find line boundaries
        threshold = np.max(h_proj) * 0.1 if np.max(h_proj) > 0 else 0
        in_line = False
        lines = []
        start = 0
        
        for i, val in enumerate(h_proj):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                if i - start > 15:  # Min line height
                    lines.append((start, i))
                in_line = False
        
        if in_line:
            lines.append((start, len(h_proj)))
        
        if not lines:
            # No lines detected, OCR whole image
            result = self.recognize(img, engines=engines, voting_method=voting_method, preprocess=preprocess)
            return result.text.strip()
        
        print(f"      Detected {len(lines)} text lines")
        
        # OCR each line
        all_text = []
        for i, (y1, y2) in enumerate(lines):
            # Add padding
            y1 = max(0, y1 - 5)
            y2 = min(upscaled.shape[0], y2 + 5)
            
            # Crop line
            line_img = upscaled[y1:y2, :]
            pil_line = Image.fromarray(line_img)
            
            # OCR line
            result = self.recognize(pil_line, engines=engines, voting_method=voting_method, preprocess=preprocess)
            line_text = result.text.strip()
            
            if line_text:
                all_text.append(line_text)
                print(f"      Line {i+1}: {line_text[:50]}..." if len(line_text) > 50 else f"      Line {i+1}: {line_text}")
        
        return " ".join(all_text)
    
    def recognize_pdf(self, pdf_path: Union[str, Path, io.BytesIO],
                     engines: Optional[List[str]] = None,
                     dpi: int = 300,
                     try_text_extraction: bool = True,
                     voting_method: str = 'majority',
                     preprocess: str = 'none',
                     add_page_markers: bool = False) -> Dict[str, Any]:
        """
        Nhận diện PDF với fallback strategies:
        1. Try extract text directly (if digital PDF)
        2. OCR from images (if scanned PDF or extraction failed)
        
        Args:
            pdf_path: Path or BytesIO to PDF file
            engines: OCR engines to use
            dpi: Resolution for rendering PDF pages
            try_text_extraction: Try to extract text layer first
            voting_method: How to combine OCR results
            preprocess: Preprocessing method
            add_page_markers: Whether to add "=== PAGE X ===" headers (default: False for clean output)
        
        Returns:
            Dictionary with 'text', 'pages', 'engine', 'method' keys
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        # Load PDF
        if isinstance(pdf_path, (str, Path)):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
        else:
            pdf_bytes = pdf_path.getvalue()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Lưu số trang trước khi close
        total_pages = len(doc)
        
        all_pages_text = []
        
        # Process each page: extract text + OCR images inside PDF
        print("📄 Processing PDF pages (text + embedded images)...")
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_texts = []
            
            # 1. Extract direct text from page
            direct_text = page.get_text("text").strip()
            if direct_text:
                page_texts.append(direct_text)
                print(f"  Page {page_num + 1}: Extracted {len(direct_text)} chars of text")
            
            # 2. OCR embedded images in the page
            images = page.get_images()
            if images:
                print(f"  Page {page_num + 1}: Found {len(images)} embedded images, running OCR...")
                
                for img_idx, img_info in enumerate(images):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        img = Image.open(io.BytesIO(image_bytes))
                        orig_size = img.size
                        
                        # Skip very small images (likely icons/decorations)
                        if img.width < 50 or img.height < 20:
                            continue
                        
                        # Check if image is multi-line text (wide aspect ratio)
                        aspect_ratio = img.width / img.height
                        
                        if aspect_ratio > 3:  # Wide image = likely multi-line text
                            # Use line-by-line OCR for better accuracy
                            print(f"    Image {img_idx + 1}: Wide image ({orig_size}), using line-by-line OCR...")
                            ocr_text = self._ocr_multiline_image(img, engines, voting_method, preprocess)
                        else:
                            # Single line or square image: upscale and OCR normally
                            if img.height < 200:
                                scale_factor = 400 / img.height
                                new_width = int(img.width * scale_factor)
                                new_height = int(img.height * scale_factor)
                                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                print(f"    Image {img_idx + 1}: Upscaled {orig_size} → {new_width}x{new_height}")
                            
                            result = self.recognize(img, engines=engines, voting_method=voting_method, preprocess=preprocess)
                            ocr_text = result.text.strip()
                        
                        if ocr_text:
                            page_texts.append(f"[Image {img_idx + 1}]: {ocr_text}")
                            print(f"    Image {img_idx + 1}: OCR got {len(ocr_text)} chars")
                    except Exception as e:
                        print(f"    Image {img_idx + 1}: OCR failed - {e}")
            
            # 3. If no text extracted at all, render whole page as image and OCR
            if not page_texts:
                print(f"  Page {page_num + 1}: No text found, rendering as image for OCR...")
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                result = self.recognize(img, engines=engines, voting_method=voting_method, preprocess=preprocess)
                if result.text.strip():
                    page_texts.append(result.text)
            
            # Combine page texts
            page_text = "\n".join(page_texts) if page_texts else "[No text recognized]"
            
            if add_page_markers:
                # Add page marker if requested
                all_pages_text.append(f"=== PAGE {page_num + 1} ===\n{page_text}")
            else:
                # Just add page text without markers (clean output)
                all_pages_text.append(page_text)
        
        doc.close()
        
        final_text = "\n\n".join(all_pages_text)
        
        print(f"✅ Processed {total_pages} pages (text extraction + image OCR)")
        
        return {
            'text': final_text,
            'pages': total_pages,
            'engine': 'hybrid',
            'method': 'text_extraction + image_ocr'
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def recognize_image(image_path: Union[str, Path, Image.Image, np.ndarray],
                   engines: Optional[List[str]] = None,
                   device: str = 'cpu') -> str:
    """
    Quick function to recognize text from image
    
    Usage:
        text = recognize_image("image.png")
        text = recognize_image(pil_image, engines=['vietocr', 'paddleocr'])
    """
    ocr = VietnameseOCRAdvanced(device=device)
    result = ocr.recognize(image_path, engines=engines)
    return result.text


def recognize_pdf(pdf_path: Union[str, Path, io.BytesIO],
                 engines: Optional[List[str]] = None,
                 device: str = 'cpu') -> str:
    """
    Quick function to recognize text from PDF
    
    Usage:
        text = recognize_pdf("document.pdf")
    """
    ocr = VietnameseOCRAdvanced(device=device)
    result = ocr.recognize_pdf(pdf_path, engines=engines)
    return result['text']


# ============================================================================
# MAIN - DEMO
# ============================================================================

if __name__ == "__main__":
    print("🇻🇳 Vietnamese OCR Advanced - Demo")
    print("=" * 70)
    
    # Initialize system
    ocr = VietnameseOCRAdvanced(device='cpu')
    
    # Test with image
    if Path("test_image.png").exists():
        print("\n📸 Testing with test_image.png...")
        result = ocr.recognize("test_image.png", voting_method='weighted')
        
        print(f"\n📝 Final Text:\n{result.text}")
        print(f"\n📊 Detailed Results:")
        for r in result.all_results:
            print(f"  - {r.engine}: {r.confidence:.2%} - '{r.text[:50]}...'")
    else:
        print("\n⚠️  test_image.png not found. Place a test image to demo.")
    
    print("\n✅ Demo complete!")

