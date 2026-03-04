"""handocr package: thin wrapper around the existing `src` OCR code.

Provides a small public API and helper functions so the project can be used
as a module or via `python -m handocr.cli`.
"""
from typing import Optional
from PIL import Image
import io

from src.ocr_service import OCRService, OCRResult


def predict_image(path: Optional[str] = None, *, image_bytes: Optional[bytes] = None, char_level: bool = True) -> OCRResult:
    """Predict text from an image file path or raw bytes.

    Returns an `OCRResult`.
    """
    if path is None and image_bytes is None:
        raise ValueError("Either path or image_bytes must be provided")

    if image_bytes is None:
        with open(path, "rb") as f:
            image_bytes = f.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    service = OCRService(lazy=True)
    # use fused recognition with char-level fusion for best results
    res = service.recognize_image_fused(image, use_easyocr=True, use_tesseract=True, use_crnn=True, char_level=char_level)
    return res


def export_searchable_pdf(input_path: str, output_path: str, dpi: int = 150):
    """Create a searchable PDF from an input PDF/image using available OCR engines.
    Returns (success: bool, message: str)
    """
    with open(input_path, "rb") as f:
        data = f.read()
    service = OCRService(lazy=True)
    return service.export_searchable_pdf(data, output_path, dpi=dpi)


__all__ = ["OCRService", "OCRResult", "predict_image", "export_searchable_pdf"]
