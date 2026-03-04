import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime
from image_preprocessing import ImagePreprocessor, get_preprocessing_options
from ui.tabs.image_ocr import display_result, save_to_history
from table_detector import TableDetector, TableOCRExtractor

def render_camera_ocr_tab(settings):
    """Render the Camera OCR tab"""
    st.header("📸 Camera Capture")
    
    st.info("📷 Use your device camera to capture Vietnamese text")
    
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        try:
            # Load image
            image = Image.open(camera_image)
            
            # Preprocessing settings
            preprocessing_opts = get_preprocessing_options()
            
            col_settings1, col_settings2 = st.columns(2)
            
            with col_settings1:
                preprocessing_key = st.selectbox(
                    "🖼️ Image Preprocessing",
                    options=list(preprocessing_opts.keys()),
                    index=0,  # Default to 'none' - let OCR handle it
                    format_func=lambda x: preprocessing_opts[x]['name'],
                    key="camera_tab_preprocessing"
                )
            
            with col_settings2:
                spell_check = st.checkbox(
                    "✏️ Spell Check (sửa lỗi chính tả)",
                    value=True,
                    help="Tự động sửa lỗi dính chữ và sai dấu",
                    key="camera_spell_check"
                )
            
            # Table detection option
            col_settings3, col_settings4 = st.columns(2)
            with col_settings3:
                detect_tables = st.checkbox(
                    "📊 Detect Tables (Phát hiện bảng)",
                    value=False,
                    help="Phát hiện và trích xuất bảng biểu trong ảnh",
                    key="camera_detect_tables"
                )
            
            with col_settings4:
                if detect_tables:
                    table_ocr = st.checkbox(
                        "📝 OCR Text in Tables",
                        value=True,
                        help="Trích xuất chữ từ các bảng phát hiện được",
                        key="camera_table_ocr"
                    )
            
            # Info about ensemble
            st.info("🔄 **Ensemble Mode**: VietOCR + PaddleOCR + CRNN chạy song song, kết quả tốt nhất được chọn tự động!")
            st.warning("📸 **Tip**: Với ảnh tài liệu chụp camera, để **None** preprocessing thường cho kết quả tốt nhất!")
            
            # Display preview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Captured Image")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
            
            with col2:
                st.subheader("🔧 Preview")
                
                # Only apply preprocessing if user selected one
                if preprocessing_key != 'none':
                    try:
                        import cv2
                        cv_image = ImagePreprocessor.to_cv2(image)
                        preprocessed_cv = ImagePreprocessor.auto_preprocess(cv_image, quality=preprocessing_key)
                        preprocessed = ImagePreprocessor.to_pil(preprocessed_cv)
                        st.image(preprocessed, use_container_width=True)
                        st.caption(f"Preprocessing: {preprocessing_opts.get(preprocessing_key, {}).get('name', preprocessing_key)}")
                        ocr_input = preprocessed
                    except Exception as e:
                        st.warning(f"⚠️ Preprocessing failed: {e}")
                        st.image(image, use_container_width=True)
                        ocr_input = image
                else:
                    st.image(image, use_container_width=True)
                    st.caption("Original image (no preprocessing)")
                    ocr_input = image
            
            st.divider()
            
            # OCR button
            if st.button("🔍 Recognize Text", type="primary", use_container_width=True, key="camera_ocr_btn"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                try:
                    # Step 1: Table detection (if enabled)
                    if detect_tables:
                        status_text.text("📊 Step 1/2: Detecting tables...")
                        progress_bar.progress(0.2)
                        
                        detector = TableDetector()
                        detection_result = detector.detect_tables(image)
                        
                        if detection_result.has_tables:
                            st.success(f"✅ Phát hiện được {detection_result.table_count} bảng biểu")
                            
                            # Visualize detected tables
                            annotated = detector.visualize_detections(detection_result)
                            col_vis1, col_vis2 = st.columns(2)
                            with col_vis1:
                                st.subheader("📊 Table Detection")
                                st.image(annotated, use_container_width=True)
                            
                            # Extract text from tables
                            if table_ocr:
                                with col_vis2:
                                    st.subheader("📝 Table Content")
                                    extractor = TableOCRExtractor(settings['ocr_system'])
                                    extracted_tables = extractor.extract_text_from_tables(
                                        detection_result,
                                        engines=settings['selected_engines']
                                    )
                                    
                                    for table_data in extracted_tables:
                                        st.markdown(f"**Bảng {table_data['index'] + 1}:**")
                                        st.text_area(
                                            f"Content",
                                            table_data['text'],
                                            height=150,
                                            disabled=True,
                                            key=f"table_{table_data['index']}"
                                        )
                        else:
                            st.info("ℹ️ Không phát hiện bảng trong ảnh")
                    
                    status_text.text("🚀 Step 2/2: Recognizing text...")
                    progress_bar.progress(0.5)
                    
                    # Run OCR with ensemble - use ORIGINAL image, not preprocessed
                    # OCR engines have their own preprocessing
                    result = settings['ocr_system'].recognize(
                        image,  # Use original image - OCR will handle preprocessing
                        engines=settings['selected_engines'],
                        voting_method=settings['voting_method'],
                        preprocess=preprocessing_key,  # Let OCR system handle it
                        fast_mode=False  # Use ensemble mode
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    progress_bar.empty()
                    
                    if result.text and result.text.strip() and len(result.text.strip()) > 1:
                        final_text = result.text
                        
                        # Apply spell check if enabled
                        if spell_check:
                            try:
                                from vietnamese_spell_checker import post_process_ocr_text
                                corrected = post_process_ocr_text(result.text, verbose=False)
                                if corrected != result.text:
                                    st.success("✏️ **Spell Check Applied!** Đã sửa lỗi dính chữ và sai dấu.")
                                    final_text = corrected
                            except ImportError:
                                pass
                        
                        result.text = final_text
                        
                        st.divider()
                        display_result(result, show_details=True)
                        save_to_history(image, result)
                        
                        # Download button
                        st.download_button(
                            "📥 Download Text",
                            final_text,
                            file_name=f"camera_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="camera_download_btn"
                        )
                    else:
                        st.error("❌ No text recognized")
                        
                        # Debug info
                        with st.expander("🔍 Debug Information", expanded=True):
                            st.markdown("### 📊 OCR Analysis")
                            
                            col_d1, col_d2, col_d3 = st.columns(3)
                            with col_d1:
                                st.metric("Result Length", f"{len(result.text)} chars")
                            with col_d2:
                                st.metric("Confidence", f"{result.confidence:.0%}")
                            with col_d3:
                                st.metric("Best Engine", result.best_engine)
                            
                            st.markdown("### 🤖 Individual Engine Results")
                            
                            if result.all_results:
                                for r in result.all_results:
                                    with st.container():
                                        st.markdown(f"**{r.engine.upper()}:**")
                                        st.code(f"Text: '{r.text}'\nLength: {len(r.text)} chars\nConfidence: {r.confidence:.2%}")
                            else:
                                st.warning("No results from any engine!")
                            
                            st.markdown("### ⚙️ Settings Used")
                            st.write(f"- Engines: {', '.join(settings['selected_engines'])}")
                            st.write(f"- Voting method: {settings['voting_method']}")
                            st.write(f"- Preprocessing: {preprocessing_key}")
                            
                            st.markdown("### 📸 Image Info")
                            st.write(f"- Size: {image.size}")
                            st.write(f"- Mode: {image.mode}")
                            
                            # Show image stats
                            img_array = np.array(image.convert('L'))
                            st.write(f"- Mean pixel: {img_array.mean():.1f}")
                            st.write(f"- Min: {img_array.min()}, Max: {img_array.max()}")
                            
                            if img_array.mean() > 250:
                                st.error("⚠️ Image mostly WHITE - might be empty!")
                            elif img_array.mean() < 5:
                                st.error("⚠️ Image mostly BLACK - might be empty!")
                            else:
                                st.success(f"✅ Image has content")
                        
                        st.warning("💡 Try: Better lighting, different preprocessing, or retake photo")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ OCR Error: {e}")
                    import traceback
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())
                    
        except Exception as e:
            st.error(f"❌ Error loading camera image: {e}")
            st.info("💡 Try taking the picture again")
            import traceback
            with st.expander("🐛 Error Details"):
                st.code(traceback.format_exc())
