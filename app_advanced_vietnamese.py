"""
Vietnamese Handwriting Recognition Web App - ADVANCED VERSION
Ứng dụng web nhận dạng chữ viết tay tiếng Việt - PHIÊN BẢN NÂNG CAO

Tích hợp:
- VietOCR (Transformer, chuyên tiếng Việt)
- PaddleOCR (PP-OCRv4, mạnh cho châu Á)
- TrOCR (Microsoft Transformer)
- CRNN (Custom trained)
- Tesseract 5.0 + Vietnamese
- EasyOCR
- Ensemble voting system
"""

import streamlit as st
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import UI modules
from ui.styles import apply_custom_styles
from ui.sidebar import render_sidebar
from ui.tabs.image_ocr import render_image_ocr_tab
from ui.tabs.pdf_ocr import render_pdf_ocr_tab
from ui.tabs.camera_ocr import render_camera_ocr_tab
from ui.tabs.batch_ocr import render_batch_ocr_tab

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Vietnamese OCR - Chữ viết tay & Chữ in",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
apply_custom_styles()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🇻🇳 Vietnamese OCR Advanced System")
    st.markdown("""
    **Hệ thống nhận dạng chữ viết tay & chữ in tiếng Việt tiên tiến**
    
    📝 **Chữ viết tay** (Handwriting) - Powered by VietOCR, TrOCR, CRNN
    
    📖 **Chữ in** (Printed text) - Powered by PaddleOCR, EasyOCR, Tesseract
    
    📊 **Bảng biểu** (Tables) - Automatic detection + HTML table visualization with borders
    
    ✨ Tự động phát hiện loại chữ và chọn engine phù hợp nhất!
    """)
    st.divider()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    if settings is None:
        return
    
    st.info(f"🎯 Using: {', '.join(settings['selected_engines'])} | Voting: {settings['voting_method']}")
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Image",
        "📄 PDF Processing",
        "📸 Camera",
        "📋 Batch Processing"
    ])
    
    # Render tabs
    with tab1:
        render_image_ocr_tab(settings)
    
    with tab2:
        render_pdf_ocr_tab(settings)
    
    with tab3:
        render_camera_ocr_tab(settings)
    
    with tab4:
        render_batch_ocr_tab(settings)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 2rem;">
        <p>🇻🇳 <strong>Vietnamese OCR Advanced System</strong> v2.2.0</p>
        <p>📝 <strong>Chữ viết tay</strong>: VietOCR, TrOCR, CRNN | 📖 <strong>Chữ in</strong>: PaddleOCR, EasyOCR, Tesseract</p>
        <p>📊 <strong>Bảng biểu</strong>: Automatic detection + HTML visualization with table structure/borders</p>
        <p>© 2024 | Made with ❤️ for Vietnamese OCR</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
