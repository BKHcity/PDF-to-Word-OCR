import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the application"""
    st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stTabs [data-baseweb="tab-list"] button { font-size: 1.2rem; }
        .result-box {
            padding: 1.5rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-left: 5px solid #00D9FF;
            margin: 1rem 0;
        }
        .success-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            margin: 0.5rem 0;
        }
        .engine-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            background-color: #667eea;
            color: white;
            font-weight: bold;
            margin: 0.2rem;
        }
    </style>
    """, unsafe_allow_html=True)
