import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
from io import StringIO, BytesIO
import base64
from datetime import datetime

sys.path.append('.')

st.set_page_config(
    page_title="Data Anonymization Tool",
    page_icon=":lock:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'classification_df' not in st.session_state:
    st.session_state.classification_df = None
if 'anonymized_df' not in st.session_state:
    st.session_state.anonymized_df = None
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None
if 'button_colors' not in st.session_state:
    st.session_state.button_colors = {}
if 'has_watermark' not in st.session_state:
    st.session_state.has_watermark = False
if 'watermark_data' not in st.session_state:
    st.session_state.watermark_data = None

steps = ["Data Setup", "Risk Classification", "Anonymization", "Quality Assessment", "Results"]

if 'current_step' not in st.session_state:
    st.session_state.current_step = steps[0]

# Adminator-inspired CSS Theme
st.markdown("""
<style>
    :root {
        --primary: #4361ee;
        --primary-dark: #3a56d4;
        --secondary: #6c757d;
        --success: #28a745;
        --info: #17a2b8;
        --warning: #ffc107;
        --danger: #dc3545;
        --dark: #343a40;
        --light: #f8f9fa;
        --sidebar-bg: #1a1a2e;
        --sidebar-hover: #16213e;
        --card-bg: #2d3748;
        --body-bg: #0f1419;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --border-color: #4a5568;
    }
    
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, #7209b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--body-bg) 0%, #1a202c 50%, var(--body-bg) 100%);
    }
    
    .stButton>button {
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(67, 97, 238, 0.4);
        background: linear-gradient(135deg, var(--primary-dark) 0%, #2a3fa3 100%);
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d3748 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border-color: var(--primary);
    }
    
    .step-progress {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
        padding: 0;
        border-radius: 16px;
        background: rgba(45, 55, 72, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
    }
    
    .step-item {
        flex: 1;
        text-align: center;
        padding: 15px 10px;
        margin: 0;
        border-radius: 0;
        cursor: pointer;
        transition: all 0.3s ease;
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 500;
        position: relative;
        overflow: hidden;
    }
    
    .step-item:not(:last-child)::after {
        content: '';
        position: absolute;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 1px;
        height: 60%;
        background: var(--border-color);
    }
    
    .step-item.active {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
    }
    
    .step-item.completed {
        background: rgba(67, 97, 238, 0.1);
        color: var(--primary);
        border: 1px solid rgba(67, 97, 238, 0.3);
    }
    
    .navigation-buttons {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
        padding: 20px 0;
        border-top: 1px solid var(--border-color);
    }
    
    .watermark-warning {
        background: linear-gradient(135deg, #ff6b35, #f7931e, #ff6b35);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #ff6b35;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        font-weight: 600;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3); }
        50% { transform: scale(1.02); box-shadow: 0 12px 35px rgba(255, 107, 53, 0.5); }
        100% { transform: scale(1); box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3); }
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--border-color);
    }
    
    .stExpander {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .file-uploader {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(45, 55, 72, 0.5);
        transition: all 0.3s ease;
    }
    
    .file-uploader:hover {
        border-color: var(--primary);
        background: rgba(67, 97, 238, 0.1);
    }
    
    .stat-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d3748 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border-color: var(--primary);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, #7209b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .progress-bar {
        height: 8px;
        background: var(--border-color);
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--card-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
</style>
""", unsafe_allow_html=True)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types compatible with NumPy 2.0"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class WatermarkManager:
    def __init__(self):
        self.watermark_key = "__anonymization_metadata__"
        self.watermark_comment_prefix = "# ANONYMIZATION_WATERMARK:"
    
    def create_watermark(self, anonymization_params, quality_metrics=None):
        """Create watermark metadata"""
        watermark = {
            "anonymized": True,
            "anonymization_timestamp": datetime.now().isoformat(),
            "anonymization_params": anonymization_params,
            "quality_metrics": quality_metrics or {},
            "version": "1.0"
        }
        return watermark
    
    def detect_watermark(self, uploaded_file):
        """Detect watermark in uploaded file"""
        try:
            # Read the file content as string
            content = uploaded_file.getvalue().decode('utf-8')
            
            # Check if the first line contains our watermark
            lines = content.split('\n')
            if lines and lines[0].startswith(self.watermark_comment_prefix):
                # Extract watermark data
                watermark_line = lines[0]
                watermark_json = watermark_line.replace(self.watermark_comment_prefix, "").strip()
                
                try:
                    watermark_data = json.loads(watermark_json)
                    # Return clean content (without watermark line)
                    clean_content = '\n'.join(lines[1:])
                    return watermark_data, clean_content
                except json.JSONDecodeError:
                    st.warning("Found watermark but couldn't parse it. Proceeding without watermark detection.")
                    return None, content
            return None, content
            
        except Exception as e:
            st.warning(f"Watermark detection error: {e}. Proceeding without watermark detection.")
            # Reset file pointer and return original content
            uploaded_file.seek(0)
            content = uploaded_file.getvalue().decode('utf-8')
            return None, content
    
    def create_watermarked_csv(self, df, watermark_data):
        """Create a CSV string with watermark as comment"""
        try:
            # Convert numpy types to Python native types for JSON serialization
            processed_data = self._preprocess_for_json(watermark_data)
            
            # Serialize with custom encoder
            watermark_json = json.dumps(processed_data, cls=NumpyEncoder, indent=2)
            
            output = StringIO()
            
            # Write watermark as comment
            output.write(f"{self.watermark_comment_prefix} {watermark_json}\n")
            
            # Write the dataframe
            df.to_csv(output, index=False)
            
            return output.getvalue()
        except Exception as e:
            st.error(f"Watermark creation error: {e}")
            # Fallback: return regular CSV without watermark
            return df.to_csv(index=False)
    
    def _preprocess_for_json(self, data):
        """Recursively convert numpy types in nested structures for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._preprocess_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._preprocess_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'dtype'):  # Handle numpy scalars safely
            if np.issubdtype(data.dtype, np.integer):
                return int(data)
            elif np.issubdtype(data.dtype, np.floating):
                return float(data)
            elif np.issubdtype(data.dtype, np.bool_):
                return bool(data)
            else:
                return str(data)
        elif isinstance(data, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data

class DataAnonymizationApp:
    def __init__(self):
        self.watermark_manager = WatermarkManager()
        
    def toggle_theme(self):
        st.session_state.theme = 'dark'
    
    def reset_all_data(self):
        st.session_state.df = None
        st.session_state.classification_df = None
        st.session_state.anonymized_df = None
        st.session_state.quality_metrics = None
        st.session_state.button_colors = {}
        st.session_state.has_watermark = False
        st.session_state.watermark_data = None
        st.session_state.current_step = steps[0]
        st.success("All data has been reset. You can start a new session.")
    
    def apply_theme(self):
        theme_class = "dark-mode"
        st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            pass
        
        with col2:
            if st.button("Reset All", key="reset_btn", use_container_width=True):
                self.reset_all_data()
                st.rerun()
        
        with col3:
            pass
    
    def get_current_step_index(self):
        return steps.index(st.session_state.current_step)
    
    def navigate_to_step(self, step_name):
        st.session_state.current_step = step_name
        st.rerun()
    
    def show_step_progress(self):
        current_index = self.get_current_step_index()
        
        st.markdown('<div class="step-progress">', unsafe_allow_html=True)
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if st.button(f"{step}", key=f"step_{i}", use_container_width=True):
                    self.navigate_to_step(step)
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_navigation_buttons(self):
        current_index = self.get_current_step_index()
        
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_index > 0:
                if st.button("Previous Step", use_container_width=True, key="prev_btn"):
                    prev_step = steps[current_index - 1]
                    self.navigate_to_step(prev_step)
        
        with col2:
            progress = (current_index + 1) / len(steps) * 100
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; border-radius: 12px; 
                        background: rgba(45, 55, 72, 0.8); border: 1px solid var(--border-color);">
                <div style="font-weight: 600; margin-bottom: 8px;">Step {current_index + 1} of {len(steps)}</div>
                <div style="font-size: 1.1rem; color: var(--primary); font-weight: 700;">{steps[current_index]}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if current_index < len(steps) - 1:
                if st.button("Next Step", use_container_width=True, key="next_btn"):
                    next_step = steps[current_index + 1]
                    self.navigate_to_step(next_step)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        self.apply_theme()
        
        st.markdown('<div class="main-header">Data Anonymization Dashboard</div>', unsafe_allow_html=True)
        
        subtitle_color = "#a0c8ff"
        st.markdown(f'''
        <p style="text-align: center; color: {subtitle_color}; font-size: 1.1rem; margin-bottom: 2rem;">
            Protect sensitive data while preserving utility • Enterprise-grade privacy solutions
        </p>
        ''', unsafe_allow_html=True)
        
        self.show_step_progress()
        
        if st.session_state.current_step == "Data Setup":
            self.data_setup_tab()
        elif st.session_state.current_step == "Risk Classification":
            self.classification_tab()
        elif st.session_state.current_step == "Anonymization":
            self.anonymization_tab()
        elif st.session_state.current_step == "Quality Assessment":
            self.quality_tab()
        elif st.session_state.current_step == "Results":
            self.results_tab()
        
        self.show_navigation_buttons()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def data_setup_tab(self):
        st.header("Data Setup & Upload")
        
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag and drop your CSV file here", type="csv", key="file_uploader")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                # First, check for watermark
                watermark_data, clean_content = self.watermark_manager.detect_watermark(uploaded_file)
                
                if watermark_data and watermark_data.get('anonymized'):
                    st.markdown('<div class="watermark-warning">WARNING: This file appears to be already anonymized!</div>', unsafe_allow_html=True)
                    
                    with st.expander("View Anonymization Details"):
                        st.json(watermark_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Proceed Anyway", type="secondary", use_container_width=True):
                            # Load data without watermark line
                            if clean_content:
                                df = pd.read_csv(StringIO(clean_content))
                            else:
                                # Fallback: read normally and skip first line
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file, skiprows=1)
                            st.session_state.df = df
                            st.session_state.has_watermark = True
                            st.session_state.watermark_data = watermark_data
                            st.success("File loaded with watermark acknowledgment")
                            st.rerun()
                    
                    with col2:
                        if st.button("Upload Different File", type="primary", use_container_width=True):
                            st.session_state.df = None
                            st.rerun()
                    
                    return
                
                # If no watermark detected, load normally
                if clean_content:
                    df = pd.read_csv(StringIO(clean_content))
                else:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file)
                    
                st.session_state.df = df
                st.session_state.has_watermark = False
                st.session_state.watermark_data = None
                st.success(f"Data loaded successfully! Shape: {df.shape}")
                
                # Show dataset overview
                self.show_dataset_overview(df)
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            if st.session_state.df is not None:
                watermark_status = " (Previously Anonymized)" if st.session_state.get('has_watermark') else ""
                st.info(f"Data already loaded{watermark_status}. You can proceed to the next step.")
                st.dataframe(st.session_state.df.head(5), use_container_width=True)
            else:
                st.info("Please upload a CSV file to get started")
    
    def show_dataset_overview(self, df):
        """Show dataset overview with stat cards"""
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-number">{df.shape[0]:,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-number">{df.shape[1]}</div>
                <div class="stat-label">Total Columns</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-number">{memory_usage:.1f}</div>
                <div class="stat-label">Memory (MB)</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            null_count = df.isnull().sum().sum()
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-number">{null_count}</div>
                <div class="stat-label">Null Values</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
    
    def classification_tab(self):
        st.header("Risk Classification Analysis")
        
        if st.session_state.df is None:
            st.warning("Please load data first in the Data Setup step.")
            return
        
        if st.session_state.classification_df is not None:
            st.info("Risk classification already completed. You can view results below or rerun classification.")
            st.dataframe(st.session_state.classification_df, use_container_width=True)
            self.show_tier_distribution(st.session_state.classification_df)
        
        if st.button("Run Risk Classification", type="primary", use_container_width=True, key="run_classification"):
            with st.spinner("Analyzing data and classifying risks..."):
                try:
                    from first import DataTierClassifier
                    classifier = DataTierClassifier()
                    classification_df = classifier.analyze_dataset(st.session_state.df)
                    
                    if 'Risk Tier' in classification_df.columns:
                        classification_df = classification_df.rename(
                            columns={'Risk Tier': 'Risk Tier Classification'}
                        )
                    
                    st.session_state.classification_df = classification_df
                    st.success("Risk classification completed!")
                    
                    st.subheader("Classification Results")
                    st.dataframe(classification_df, use_container_width=True)
                    self.show_tier_distribution(classification_df)
                    
                except Exception as e:
                    st.error(f"Error during classification: {e}")
    
    def show_tier_distribution(self, classification_df):
        st.subheader("Tier Distribution")
        tier_column = 'Risk Tier Classification' if 'Risk Tier Classification' in classification_df.columns else 'Risk Tier'
        tier_counts = classification_df[tier_column].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#ff6b6b', '#ffa726', '#66bb6a', '#4a90e2', '#8b5cf6']
        text_color = 'white'
        bg_color = '#0f1419'
        
        wedges, texts, autotexts = ax.pie(
            tier_counts.values, 
            labels=tier_counts.index, 
            autopct='%1.1f%%',
            colors=colors[:len(tier_counts)],
            startangle=90,
            explode=[0.05] * len(tier_counts)
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('Risk Tier Distribution', color=text_color, fontsize=16, fontweight='bold', pad=20)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        for text in texts:
            text.set_color(text_color)
            text.set_fontsize(11)
        
        st.pyplot(fig)

    def anonymization_tab(self):
        st.header("Data Anonymization")
        
        if st.session_state.df is None:
            st.warning("Please load data first in the Data Setup step.")
            return
        
        if st.session_state.classification_df is None:
            st.warning("Please run risk classification in the Risk Classification step first.")
            return
        
        # Show watermark warning if data is already anonymized
        if st.session_state.get('has_watermark'):
            st.markdown('<div class="watermark-warning">WARNING: This data appears to be already anonymized! Re-anonymizing may reduce data quality and should not be performed.</div>', unsafe_allow_html=True)
            
            if st.session_state.get('watermark_data'):
                with st.expander("Original Anonymization Details"):
                    st.json(st.session_state.watermark_data)
            
            # Disable anonymization for watermarked data
            st.error("Anonymization is disabled for already anonymized data. Please upload original data to perform anonymization.")
            return
        
        if st.session_state.anonymized_df is not None:
            st.info("Anonymization already completed. You can view results below or rerun anonymization.")
            st.dataframe(st.session_state.anonymized_df.head(10), use_container_width=True)
        
        st.subheader("Anonymization Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            k = st.number_input("k-anonymity", min_value=2, max_value=100, value=5, key="k_param")
            st.info(f"Privacy level: {k}")
        
        with col2:
            epsilon = st.number_input("DP Epsilon", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="epsilon_param")
            st.info(f"Noise scale: {epsilon}")
        
        with col3:
            pca_threshold = st.number_input("PCA Threshold", min_value=0.1, max_value=1.0, value=0.85, step=0.05, key="pca_param")
            st.info(f"Variance: {pca_threshold}")
        
        with col4:
            use_pca = st.checkbox("Use PCA", value=True, key="use_pca")
            st.info("Dimensionality reduction")
        
        if st.button("Run Anonymization", type="primary", use_container_width=True, key="run_anonymization"):
            with st.spinner("Applying anonymization techniques..."):
                try:
                    from ultimate_anonymizer import DataAnonymizer
                    
                    classification_df = st.session_state.classification_df
                    
                    anonymizer = DataAnonymizer(
                        k=k, 
                        epsilon=epsilon, 
                        pca_variance_threshold=pca_threshold
                    )
                    
                    anonymized_df, tier_mapping, techniques = anonymizer.anonymize_dataset(
                        st.session_state.df, classification_df, use_pca=use_pca
                    )
                    
                    st.session_state.anonymized_df = anonymized_df
                    st.success("Anonymization completed!")
                    
                    st.subheader("Anonymization Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f'''
                        <div class="stat-card">
                            <div class="stat-number">{st.session_state.df.shape[0]:,}</div>
                            <div class="stat-label">Original Rows</div>
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">{st.session_state.df.shape[1]} columns</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'''
                        <div class="stat-card">
                            <div class="stat-number">{anonymized_df.shape[0]:,}</div>
                            <div class="stat-label">Anonymized Rows</div>
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">{anonymized_df.shape[1]} columns</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    st.subheader("Anonymized Data Preview")
                    st.dataframe(anonymized_df.head(10), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error during anonymization: {str(e)}")

    def quality_tab(self):
        st.header("Quality Assessment")
        
        if st.session_state.df is None or st.session_state.anonymized_df is None:
            st.warning("Please complete anonymization in the Anonymization step first.")
            return
        
        if st.session_state.quality_metrics is not None:
            st.info("Quality assessment already completed. You can view results below or rerun assessment.")
            self.show_quality_metrics(st.session_state.quality_metrics)
        
        if st.button("Run Quality Assessment", type="primary", use_container_width=True, key="run_quality"):
            with st.spinner("Analyzing data quality and privacy metrics..."):
                try:
                    from PCA_Anonymization_Report import AnonymizationQualityReport
                    
                    original_file = "temp_original.csv"
                    anonymized_file = "temp_anonymized.csv"
                    
                    st.session_state.df.to_csv(original_file, index=False)
                    st.session_state.anonymized_df.to_csv(anonymized_file, index=False)
                    
                    report_generator = AnonymizationQualityReport()
                    quality_metrics = report_generator.generate_report(
                        original_file, anonymized_file, "quality_report"
                    )
                    
                    if os.path.exists(original_file):
                        os.remove(original_file)
                    if os.path.exists(anonymized_file):
                        os.remove(anonymized_file)
                    
                    st.session_state.quality_metrics = quality_metrics
                    st.success("Quality assessment completed!")
                    
                    self.show_quality_metrics(quality_metrics)
                        
                except Exception as e:
                    st.error(f"Error during quality assessment: {e}")
    
    def show_quality_metrics(self, quality_metrics):
        st.subheader("Quality Metrics")
        
        if quality_metrics:
            stats = quality_metrics['statistical_metrics']
            privacy = quality_metrics['privacy_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Statistical Preservation")
                
                correlation_pres = stats.get('correlation_preservation', 0)
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-number">{correlation_pres:.3f}</div>
                    <div class="stat-label">Correlation Preservation</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {correlation_pres * 100}%"></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                mean_mse = stats.get('mean_mse', 0)
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-number">{mean_mse:.3f}</div>
                    <div class="stat-label">Average MSE</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Privacy Protection")
                
                uniqueness_red = privacy.get('uniqueness_reduction', 0)
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-number">{uniqueness_red:.1%}</div>
                    <div class="stat-label">Uniqueness Reduction</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {uniqueness_red * 100}%"></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                variance_pres = privacy.get('variance_preservation', 0)
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-number">{variance_pres:.1%}</div>
                    <div class="stat-label">Variance Preservation</div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.subheader("Overall Assessment")
        if stats.get('correlation_preservation', 0) > 0.8 and privacy.get('uniqueness_reduction', 0) > 0.3:
            st.success("EXCELLENT - Strong privacy with high utility preservation!")
        elif stats.get('correlation_preservation', 0) > 0.6:
            st.info("GOOD - Balanced privacy-utility tradeoff")
        else:
            st.warning("FAIR - Consider adjusting anonymization parameters")

    def results_tab(self):
        st.header("Results & Export")
        
        if st.session_state.anonymized_df is None:
            st.warning("Please complete anonymization first.")
            return
        
        st.subheader("Anonymized Data Preview")
        st.dataframe(st.session_state.anonymized_df.head(10), use_container_width=True)
        
        st.subheader("Download Options")
        
        # Prepare watermark data
        anonymization_params = {
            "k_anonymity": st.session_state.get('k_param', 5),
            "epsilon": st.session_state.get('epsilon_param', 1.0),
            "pca_threshold": st.session_state.get('pca_param', 0.85),
            "use_pca": st.session_state.get('use_pca', True)
        }
        
        quality_metrics = st.session_state.quality_metrics or {}
        
        watermark_data = self.watermark_manager.create_watermark(
            anonymization_params, 
            quality_metrics
        )
        
        # Create download options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### With Watermark")
            st.info("Recommended - includes anonymization metadata")
            
            watermarked_csv = self.watermark_manager.create_watermarked_csv(
                st.session_state.anonymized_df, 
                watermark_data
            )
            b64_watermarked = base64.b64encode(watermarked_csv.encode()).decode()
            
            download_style = """
            background: linear-gradient(135deg, #4361ee, #3a56d4, #4361ee);
            color: white;
            padding: 15px 30px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            border-radius: 12px;
            font-weight: 600;
            margin: 10px 0;
            width: 100%;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
            """
            
            href_watermarked = f'<a href="data:file/csv;base64,{b64_watermarked}" download="anonymized_data_with_watermark.csv" style="{download_style}">Download with Watermark</a>'
            st.markdown(href_watermarked, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Plain CSV")
            st.info("Standard CSV without metadata")
            
            csv_plain = st.session_state.anonymized_df.to_csv(index=False)
            b64_plain = base64.b64encode(csv_plain.encode()).decode()
            
            download_style_plain = """
            background: linear-gradient(135deg, #6c757d, #5a6268, #6c757d);
            color: white;
            padding: 15px 30px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            border-radius: 12px;
            font-weight: 600;
            margin: 10px 0;
            width: 100%;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
            """
            
            href_plain = f'<a href="data:file/csv;base64,{b64_plain}" download="anonymized_data.csv" style="{download_style_plain}">Download Plain CSV</a>'
            st.markdown(href_plain, unsafe_allow_html=True)
        
        # Show watermark preview
        with st.expander("Preview Watermark Contents"):
            st.json(watermark_data)
            st.info("This metadata will be embedded in the downloaded file and detected if re-uploaded.")
        
        if st.session_state.quality_metrics is not None:
            st.subheader("Quality Assessment Results")
            self.show_quality_metrics(st.session_state.quality_metrics)

if __name__ == "__main__":
    app = DataAnonymizationApp()
    app.run()