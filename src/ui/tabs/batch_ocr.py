import streamlit as st
from PIL import Image
from datetime import datetime
from vietnamese_preprocessing import VietnameseImagePreprocessor

def render_batch_ocr_tab(settings):
    """Render the Batch OCR tab"""
    st.header("📋 Batch Processing")
    
    st.info("💡 Upload multiple images for batch OCR processing")
    
    batch_files = st.file_uploader(
        "Choose multiple images",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True
    )
    
    if batch_files:
        st.success(f"✅ Loaded {len(batch_files)} files")
        
        if st.button("🚀 Process All", type="primary", width='stretch'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, file in enumerate(batch_files):
                status_text.text(f"Processing {file.name}...")
                
                image = Image.open(file)
                
                # Preprocess if enabled
                if settings['preprocess_enabled']:
                    preprocessor = VietnameseImagePreprocessor()
                    preprocessed = preprocessor.process_for_ocr(
                        image, image_type=settings['image_type'], aggressive=settings['aggressive_mode']
                    )
                    ocr_input = Image.fromarray(preprocessed)
                else:
                    ocr_input = image
                
                result = settings['ocr_system'].recognize(
                    ocr_input,
                    engines=settings['selected_engines'],
                    voting_method=settings['voting_method']
                )
                
                results.append({
                    'file': file.name,
                    'text': result.text,
                    'confidence': result.confidence,
                    'engine': result.best_engine
                })
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display results
            st.markdown("### 📊 Batch Results")
            
            for r in results:
                with st.expander(f"📄 {r['file']}"):
                    st.markdown(f"**Engine:** {r['engine']} | **Confidence:** {r['confidence']:.1%}")
                    st.text_area("Text", r['text'], height=100, key=r['file'])
            
            # Download all
            combined_text = "\n\n".join([
                f"=== {r['file']} ===\n{r['text']}"
                for r in results
            ])
            
            st.download_button(
                "📥 Download All Results",
                combined_text,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
