"""
PDF utilities: convert PDF bytes to images and create searchable PDF with text layer.
Requires PyMuPDF (fitz) and reportlab.
"""
from typing import List, Tuple
from PIL import Image
import io

def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> Tuple[List[Image.Image], str]:
    """Convert PDF bytes to list of PIL Images. Returns (images, error_message).
    If fitz is not available, returns ([], error_str).
    """
    try:
        import fitz
    except Exception as e:
        return [], f"PyMuPDF (fitz) not installed: {e}"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        doc.close()
        return images, ""
    except Exception as e:
        return [], str(e)


def create_searchable_pdf(output_path: str, images: List[Image.Image], words_per_page: List[List[dict]], dpi: int = 150):
    """Create a PDF with each page image and a text layer overlay using ReportLab.
    `words_per_page` is a list (per page) of dicts: {'text': str, 'bbox': (x1,y1,x2,y2), 'confidence': float}
    Coordinates are expected in pixels at the same `dpi` used to render images.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import portrait
    except Exception as e:
        raise RuntimeError(f"reportlab not installed: {e}")

    from reportlab.lib.units import inch

    c = canvas.Canvas(output_path)

    for img, words in zip(images, words_per_page):
        w_px, h_px = img.size
        # page size in points
        page_w = w_px * 72.0 / dpi
        page_h = h_px * 72.0 / dpi
        c.setPageSize((page_w * inch, page_h * inch))

        # draw image full page
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        # reportlab draws image at lower-left corner
        c.drawImage(ImageReader(img_io), 0, 0, width=page_w * inch, height=page_h * inch)

        # overlay text (invisible text) at positions
        for w in words:
            text = w.get('text', '')
            if not text:
                continue
            x1, y1, x2, y2 = w.get('bbox')
            # convert pixel coords to points
            x_pt = x1 * 72.0 / dpi
            # reportlab origin bottom-left; PIL origin top-left
            y_pt = (h_px - y2) * 72.0 / dpi
            font_size = max(6, (y2 - y1) * 72.0 / dpi)
            try:
                c.setFont('Helvetica', font_size)
            except Exception:
                pass
            c.setFillColorRGB(1,1,1)
            c.drawString(x_pt * inch, y_pt * inch, text)

        c.showPage()

    c.save()


# helper for reportlab image
try:
    from reportlab.lib.utils import ImageReader
except Exception:
    ImageReader = None
