import streamlit as st
from PIL import Image
from datetime import datetime
from image_preprocessing import ImagePreprocessor, get_preprocessing_options

def display_result(result, show_details: bool = True):
    """Display OCR result with nice formatting"""
    
    # Low consensus warning
    if result.consensus_score < 0.2:
        st.warning(f"""
        ⚠️ **LOW CONSENSUS ({result.consensus_score:.0%})** - Engines disagree strongly!
        
        The result may be inaccurate. Try:
        - **Preprocessing**: Use Light/Medium/Heavy to enhance image
        - **Retake photo**: Better lighting, less blur
        - **Check**: Does the output look reasonable?
        """)
    
    # Main result
    st.markdown(f"""
    <div class="result-box">
        <h3>📝 Recognized Text:</h3>
        <p style="font-size: 1.2rem; margin-top: 1rem;">{result.text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Best Engine", result.best_engine.upper())
    
    with col2:
        st.metric("🎯 Confidence", f"{result.confidence:.1%}")
    
    with col3:
        st.metric("🤝 Consensus", f"{result.consensus_score:.1%}")
    
    with col4:
        st.metric("⏱️ Time", f"{result.processing_time:.2f}s")
    
    # Detailed results
    if show_details and len(result.all_results) > 1:
        with st.expander("🔍 Detailed Engine Results"):
            st.markdown("### All Engine Outputs")
            
            for r in result.all_results:
                st.markdown(f"""
                <div class="success-box">
                    <span class="engine-badge">{r.engine}</span>
                    <strong>Confidence:</strong> {r.confidence:.1%} | 
                    <strong>Time:</strong> {r.processing_time:.2f}s
                    <br/>
                    <strong>Text:</strong> {r.text[:200]}{'...' if len(r.text) > 200 else ''}
                </div>
                """, unsafe_allow_html=True)

def save_to_history(image, result):
    """Save processing to history"""
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
        
    st.session_state.processing_history.append({
        'timestamp': datetime.now(),
        'text': result.text,
        'engine': result.best_engine,
        'confidence': result.confidence
    })

def render_image_ocr_tab(settings):
    """Render the Image OCR tab"""
    st.header("📤 Upload Image or PDF for OCR")
    
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'],
        help="Upload handwritten or printed Vietnamese text (Image or PDF)"
    )
    
    if uploaded_file is not None:
        try:
            # Handle PDF files
            if uploaded_file.name.lower().endswith('.pdf'):
                import io
                try:
                    import fitz  # PyMuPDF
                except ImportError:
                    st.error("❌ PyMuPDF not installed for PDF support")
                    return
                
                # Read PDF
                pdf_bytes = io.BytesIO(uploaded_file.read())
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                # Let user select which page
                total_pages = len(doc)
                if total_pages > 1:
                    page_num = st.slider("Select page", 1, total_pages, 1) - 1
                else:
                    page_num = 0
                
                # Convert page to image
                page = doc.load_page(page_num)
                zoom = 300 / 72  # 300 DPI
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                
                st.info(f"📄 PDF: {uploaded_file.name} | Page {page_num + 1}/{total_pages}")
                doc.close()
            else:
                # Handle image files
                image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return
        
        # Preprocessing settings
        preprocessing_opts = get_preprocessing_options()
        
        col_settings1, col_settings2 = st.columns(2)
        
        with col_settings1:
            preprocessing_key = st.selectbox(
                "🖼️ Image Preprocessing",
                options=list(preprocessing_opts.keys()),
                format_func=lambda x: preprocessing_opts[x]['name'],
                key="image_tab_preprocessing"
            )
        
        with col_settings2:
            spell_check = st.checkbox(
                "✏️ Spell Check (sửa lỗi chính tả)",
                value=True,
                help="Tự động sửa lỗi dính chữ (xinhãy → xin hãy), sai dấu (ranhé → ra nhé)"
            )
        
        # Info about ensemble
        st.info("🔄 **Ensemble Mode**: VietOCR + CRNN chạy song song, kết quả tốt nhất cho mỗi dòng được chọn tự động!")
        
        # Display original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, width=800)
        
        with col2:
            st.subheader("🔧 Preprocessed")
            
            if preprocessing_key != 'none':
                with st.spinner(f"Preprocessing ({preprocessing_key})..."):
                    import cv2
                    cv_image = ImagePreprocessor.to_cv2(image)
                    preprocessed_cv = ImagePreprocessor.auto_preprocess(cv_image, quality=preprocessing_key)
                    preprocessed = ImagePreprocessor.to_pil(preprocessed_cv)
                    st.image(preprocessed, width=800)
                    
                    # Use preprocessed for OCR
                    ocr_input = preprocessed
            else:
                st.image(image, width=800)
                ocr_input = image
        
        st.divider()
        
        # OCR button
        if st.button("🔍 Recognize Text", type="primary", use_container_width=True, key="image_ocr_btn"):
            # Prepare spinner message - now always ensemble
            spinner_msg = "🔄 Ensemble OCR (VietOCR + CRNN song song)"
            
            with st.spinner(spinner_msg + "..."):
                result = settings['ocr_system'].recognize(
                    ocr_input,
                    engines=settings['selected_engines'],
                    voting_method=settings['voting_method'],
                    preprocess='none',  # Already preprocessed above
                    fast_mode=False  # Use ensemble mode (VietOCR + CRNN)
                )
            
            if result.text:
                final_text = result.text
                
                # Apply spell check if enabled
                if spell_check:
                    try:
                        from vietnamese_spell_checker import post_process_ocr_text
                        corrected = post_process_ocr_text(result.text, verbose=False)
                        if corrected != result.text:
                            st.success("✏️ **Spell Check Applied!** Đã sửa lỗi dính chữ và sai dấu.")
                            
                            # Show before/after
                            with st.expander("📝 Xem chi tiết sửa đổi"):
                                col_before, col_after = st.columns(2)
                                with col_before:
                                    st.markdown("**Trước:**")
                                    st.text(result.text)
                                with col_after:
                                    st.markdown("**Sau:**")
                                    st.text(corrected)
                            
                            final_text = corrected
                    except ImportError:
                        st.warning("⚠️ Spell checker module not available")
                
                # Update result text
                result.text = final_text
                
                # Display result
                display_result(result, show_details=True)
                
                # Save to history
                save_to_history(image, result)
                
                # Download options
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        "📥 Download Text (.txt)",
                        final_text,
                        file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="image_download_txt_btn"
                    )
                
                with col_dl2:
                    # Create detailed report
                    report = f"""Vietnamese OCR Advanced - Recognition Report
{'='*60}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Best Engine: {result.best_engine}
Confidence: {result.confidence:.2%}
Consensus: {result.consensus_score:.2%}
Processing Time: {result.processing_time:.2f}s

{'='*60}
RECOGNIZED TEXT:
{'='*60}
{result.text}

{'='*60}
ENGINE DETAILS:
{'='*60}
"""
                    for r in result.all_results:
                        report += f"\n{r.engine.upper()}:\n"
                        report += f"  Confidence: {r.confidence:.2%}\n"
                        report += f"  Time: {r.processing_time:.2f}s\n"
                        report += f"  Text: {r.text[:100]}...\n"
                    
                    st.download_button(
                        "📄 Download Full Report",
                        report,
                        file_name=f"ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="image_download_report_btn"
                    )
            else:
                st.error("❌ No text recognized. Try adjusting preprocessing settings.")
