import streamlit as st
from vietnamese_ocr_advanced import VietnameseOCRAdvanced

@st.cache_resource
def load_ocr_system(device: str = 'cpu'):
    """Load OCR system with caching"""
    with st.spinner("🚀 Initializing Advanced OCR System..."):
        return VietnameseOCRAdvanced(device=device, enable_all=True)

def render_sidebar():
    """Render the sidebar and return settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device = st.selectbox(
            "🖥️ Device",
            ['cpu', 'cuda'],
            help="Use CUDA for GPU acceleration (if available)"
        )
        
        st.divider()
        
        # Load OCR system
        if 'ocr_system' not in st.session_state or st.session_state.ocr_system is None:
            st.session_state.ocr_system = load_ocr_system(device)
        
        ocr_system = st.session_state.ocr_system
        
        # Engine selection - ONLY SHOW AVAILABLE ENGINES
        st.markdown("### 🤖 OCR Engines")
        
        # Get available engines first
        temp_available = list(ocr_system.engines.keys()) if ocr_system else []
        
        # Show status
        if temp_available:
            st.caption(f"✅ Available: {', '.join(temp_available)}")
        
        # Only show checkboxes for available engines
        selected_engines = []
        
        if 'vietocr' in temp_available:
            if st.checkbox("VietOCR (Transformer, Best for Vietnamese)", value=True, key='cb_vietocr'):
                selected_engines.append('vietocr')
        else:
            st.checkbox("VietOCR (Not installed)", value=False, disabled=True, key='cb_vietocr_disabled',
                       help="Install: pip install vietocr")
        
        if 'paddleocr' in temp_available:
            if st.checkbox("PaddleOCR (PP-OCRv4, Excellent for Asian)", value=True, key='cb_paddleocr'):
                selected_engines.append('paddleocr')
        else:
            st.checkbox("PaddleOCR (Not installed)", value=False, disabled=True, key='cb_paddleocr_disabled',
                       help="Install: pip install paddlepaddle paddleocr")
        
        if 'trocr' in temp_available:
            if st.checkbox("TrOCR (Microsoft Transformer)", value=True, key='cb_trocr',
                          help="Slow but accurate"):
                selected_engines.append('trocr')
        else:
            st.checkbox("TrOCR (Not loaded)", value=False, disabled=True, key='cb_trocr_disabled')
        
        if 'protonx' in temp_available:
            if st.checkbox("ProtonX (New Engine)", value=True, key='cb_protonx'):
                selected_engines.append('protonx')
        else:
            st.checkbox("ProtonX (Not installed)", value=False, disabled=True, key='cb_protonx_disabled',
                       help="Install: pip install --upgrade protonx")
        
        if 'crnn' in temp_available:
            if st.checkbox("CRNN (Custom trained)", value=True, key='cb_crnn'):
                selected_engines.append('crnn')
        else:
            st.checkbox("CRNN (Not loaded)", value=False, disabled=True, key='cb_crnn_disabled')
        
        if 'tesseract' in temp_available:
            if st.checkbox("Tesseract (Printed text)", value=True, key='cb_tesseract'):
                selected_engines.append('tesseract')
        else:
            st.checkbox("Tesseract (Not installed)", value=False, disabled=True, key='cb_tesseract_disabled',
                       help="Install: See INSTALL_NOW.txt")
        
        if 'easyocr' in temp_available:
            if st.checkbox("EasyOCR (General purpose)", value=True, key='cb_easyocr'):
                selected_engines.append('easyocr')
        else:
            st.checkbox("EasyOCR (Not loaded)", value=False, disabled=True, key='cb_easyocr_disabled')
        
        # Show summary
        if not selected_engines and temp_available:
            st.warning("⚠️ Please select at least one available engine")
            selected_engines = temp_available  # Use all available as default
        
        st.divider()
        
        # Voting method
        voting_method = st.selectbox(
            "🗳️ Voting Method",
            ['weighted', 'best', 'majority'],
            help="How to combine results from multiple engines"
        )
        
        st.divider()
        
        # Speed vs Accuracy
        st.markdown("### ⚡ Speed Settings")
        
        fast_mode = st.checkbox(
            "⚡ Fast Mode (VietOCR only - NHANH NHẤT)",
            value=True,
            help="Skip slow engines (CRNN, EasyOCR) - 3s instead of 120s"
        )
        
        st.divider()
        
        # Preprocessing options
        st.markdown("### 🖼️ Preprocessing")
        
        preprocess_enabled = st.checkbox("Enable Advanced Preprocessing", value=True)
        aggressive_mode = st.checkbox("Aggressive Enhancement", value=False,
                                     help="Use for difficult/low-quality images")
        image_type = st.selectbox(
            "Image Type",
            ['auto', 'handwritten', 'printed'],
            help="Auto-detect or manually specify"
        )
        
        st.divider()
        
        # Statistics
        if 'processing_history' in st.session_state and st.session_state.processing_history:
            st.markdown("### 📊 Statistics")
            st.metric("Total Processed", len(st.session_state.processing_history))
            
            if st.button("Clear History"):
                st.session_state.processing_history = []
                st.rerun()
        
        st.divider()
        
        # Info
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Version:** 2.0.0 Advanced
        
        **Engines:**
        - 🇻🇳 VietOCR: Transformer-based
        - 🚀 PaddleOCR: PP-OCRv4
        - 🤖 TrOCR: Microsoft
        - 🧠 CRNN: Custom trained
        - 📄 Tesseract: v5.0
        - 🌐 EasyOCR: Multi-lang
        
        **Features:**
        - Ensemble voting
        - Vietnamese diacritics optimization
        - Advanced preprocessing
        - PDF support
        - Batch processing
        """)

    # Validate selection
    available = list(ocr_system.engines.keys())
    available_selected = [eng for eng in selected_engines if eng in available]
    
    if not available_selected and selected_engines:
        st.error(f"❌ None of selected engines are available!")
        return None
        
    if len(available_selected) < len(selected_engines):
        selected_engines = available_selected
        
    return {
        'ocr_system': ocr_system,
        'selected_engines': selected_engines,
        'voting_method': voting_method,
        'preprocess_enabled': preprocess_enabled,
        'aggressive_mode': aggressive_mode,
        'image_type': image_type,
        'fast_mode': fast_mode
    }
