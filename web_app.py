import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from io import StringIO
import base64

sys.path.append('.')

st.set_page_config(
    page_title="Data Anonymization Tool",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

steps = ["Data Setup", "Risk Classification", "Anonymization", "Quality Assessment", "Results"]

if 'current_step' not in st.session_state:
    st.session_state.current_step = steps[0]

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .stButton>button {
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.5s ease;
        background-size: 200% 200% !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .reset-button {
        background: linear-gradient(45deg, #e53e3e, #c53030, #e53e3e) !important;
    }
    .progress-active {
        background: linear-gradient(45deg, #4a90e2, #8b5cf6, #4a90e2) !important;
    }
    .dark-mode {
        background: linear-gradient(135deg, #0a0a2a 0%, #1a1a4a 50%, #0a0a2a 100%);
        color: white;
    }
    .dark-mode .main-header {
        color: #ffffff;
        text-shadow: 0 0 10px #4a90e2;
    }
    .dark-mode .stButton>button {
        background: linear-gradient(45deg, #4a90e2, #8b5cf6, #4a90e2) !important;
        color: white;
    }
    .dark-mode .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .dark-mode .stSidebar {
        background: rgba(10, 10, 42, 0.9) !important;
    }
    .dark-mode .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .step-progress {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
        padding: 0px;
        border-radius: 0px;
        background: transparent;
    }
    .step-item {
        flex: 1;
        text-align: center;
        padding: 10px;
        margin: 0 2px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .step-item.active {
        background: linear-gradient(45deg, #4a90e2, #8b5cf6);
        color: white;
        font-weight: bold;
        border: 1px solid #4a90e2;
    }
    .step-item.completed {
        background: rgba(74, 144, 226, 0.2);
        color: white;
        border: 1px solid rgba(74, 144, 226, 0.3);
    }
    .navigation-buttons {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
        padding: 10px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class DataAnonymizationApp:
    def __init__(self):
        pass
        
    def toggle_theme(self):
        st.session_state.theme = 'dark'
    
    def reset_all_data(self):
        st.session_state.df = None
        st.session_state.classification_df = None
        st.session_state.anonymized_df = None
        st.session_state.quality_metrics = None
        st.session_state.button_colors = {}
        st.session_state.current_step = steps[0]
        st.success("All data has been reset. You can start a new session.")
    
    def apply_theme(self):
        theme_class = "dark-mode"
        st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            pass
        
        with col2:
            if st.button("🔄 Reset All", key="reset_btn", use_container_width=True):
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
        
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if st.button(step, key=f"step_{i}", use_container_width=True):
                    self.navigate_to_step(step)
    
    def is_step_completed(self, step_index):
        if step_index == 0:
            return st.session_state.df is not None
        elif step_index == 1:
            return st.session_state.classification_df is not None
        elif step_index == 2:
            return st.session_state.anonymized_df is not None
        elif step_index == 3:
            return st.session_state.quality_metrics is not None
        elif step_index == 4:
            return st.session_state.anonymized_df is not None
        return False
    
    def show_navigation_buttons(self):
        current_index = self.get_current_step_index()
        
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_index > 0:
                if st.button("← Previous Step", use_container_width=True, key="prev_btn"):
                    prev_step = steps[current_index - 1]
                    self.navigate_to_step(prev_step)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; border-radius: 10px; 
                        background: linear-gradient(45deg, #4a90e220, #8b5cf640);
                        border: 1px solid #4a90e230;">
                <strong>Step {current_index + 1} of {len(steps)}: {steps[current_index]}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if current_index < len(steps) - 1:
                if st.button("Next Step →", use_container_width=True, key="next_btn"):
                    next_step = steps[current_index + 1]
                    self.navigate_to_step(next_step)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        self.apply_theme()
        
        st.markdown('<div class="main-header">Data Anonymization Tool</div>', unsafe_allow_html=True)
        
        subtitle_color = "#a0c8ff"
        st.markdown(f'<p style="text-align: center; color: {subtitle_color}; font-size: 1.2rem;">Protect sensitive data while preserving utility</p>', unsafe_allow_html=True)
        
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
        st.header("📊 Step 1: Data Setup")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"Data loaded successfully! Shape: {df.shape}")
                
                st.subheader("Dataset Overview")
                col1, col2, col3 = st.columns(3)
                
                card_bg = "rgba(255,255,255,0.1)"
                card_color = "#ffffff"
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="background: {card_bg}; color: {card_color};">
                        <div style="font-size: 1.2rem; font-weight: bold;">{df.shape[0]}</div>
                        <div>Rows</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: {card_bg}; color: {card_color};">
                        <div style="font-size: 1.2rem; font-weight: bold;">{df.shape[1]}</div>
                        <div>Columns</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
                    st.markdown(f"""
                    <div class="metric-card" style="background: {card_bg}; color: {card_color};">
                        <div style="font-size: 1.2rem; font-weight: bold;">{memory_usage:.2f} MB</div>
                        <div>Memory Usage</div>
                    </div>
                    """, unsafe_allow_html=True)
                
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
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            if st.session_state.df is not None:
                st.info("Data already loaded. You can proceed to the next step.")
                st.dataframe(st.session_state.df.head(5), use_container_width=True)
            else:
                st.info("Please upload a CSV file to get started")
    
    def classification_tab(self):
        st.header("🔍 Step 2: Risk Classification")
        
        if st.session_state.df is None:
            st.warning("Please load data first in the Data Setup step.")
            return
        
        if st.session_state.classification_df is not None:
            st.info("Risk classification already completed. You can view results below or rerun classification.")
            st.dataframe(st.session_state.classification_df, use_container_width=True)
            self.show_tier_distribution(st.session_state.classification_df)
        
        if st.button("Run Risk Classification", type="primary", use_container_width=True, key="run_classification"):
            with st.spinner("Running risk classification..."):
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
        bg_color = '#0a0a2a'
        
        wedges, texts, autotexts = ax.pie(
            tier_counts.values, 
            labels=tier_counts.index, 
            autopct='%1.1f%%',
            colors=colors[:len(tier_counts)],
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Risk Tier Distribution', color=text_color, fontsize=14)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        for text in texts:
            text.set_color(text_color)
        
        st.pyplot(fig)

    def anonymization_tab(self):
        st.header("🛡️ Step 3: Anonymization")
        
        if st.session_state.df is None:
            st.warning("Please load data first in the Data Setup step.")
            return
        
        if st.session_state.classification_df is None:
            st.warning("Please run risk classification in the Risk Classification step first.")
            return
        
        if st.session_state.anonymized_df is not None:
            st.info("Anonymization already completed. You can view results below or rerun anonymization.")
            st.dataframe(st.session_state.anonymized_df.head(10), use_container_width=True)
        
        st.subheader("Anonymization Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            k = st.number_input("k-anonymity", min_value=2, max_value=100, value=5, key="k_param")
        
        with col2:
            epsilon = st.number_input("DP Epsilon", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="epsilon_param")
        
        with col3:
            pca_threshold = st.number_input("PCA Threshold", min_value=0.1, max_value=1.0, value=0.85, step=0.05, key="pca_param")
        
        with col4:
            use_pca = st.checkbox("Use PCA", value=True, key="use_pca")
        
        if st.button("Run Anonymization", type="primary", use_container_width=True, key="run_anonymization"):
            with st.spinner("Running anonymization..."):
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
                    
                    card_bg = "rgba(255,255,255,0.1)"
                    card_color = "#ffffff"
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card" style="background: {card_bg}; color: {card_color};">
                            <div>Original Dataset</div>
                            <div style="font-size: 1.2rem; font-weight: bold;">{st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card" style="background: {card_bg}; color: {card_color};">
                            <div>Anonymized Dataset</div>
                            <div style="font-size: 1.2rem; font-weight: bold;">{anonymized_df.shape[0]} rows × {anonymized_df.shape[1]} columns</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.subheader("Anonymized Data Preview")
                    st.dataframe(anonymized_df.head(10), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error during anonymization: {str(e)}")

    def quality_tab(self):
        st.header("📈 Step 4: Quality Assessment")
        
        if st.session_state.df is None or st.session_state.anonymized_df is None:
            st.warning("Please complete anonymization in the Anonymization step first.")
            return
        
        if st.session_state.quality_metrics is not None:
            st.info("Quality assessment already completed. You can view results below or rerun assessment.")
            self.show_quality_metrics(st.session_state.quality_metrics)
        
        if st.button("Run Quality Assessment", type="primary", use_container_width=True, key="run_quality"):
            with st.spinner("Running quality assessment..."):
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
                st.metric("Mean Correlation", f"{stats.get('mean_correlation', 0):.3f}")
                st.metric("Correlation Preservation", f"{stats.get('correlation_preservation', 0):.3f}")
                st.metric("Average MSE", f"{stats.get('mean_mse', 0):.3f}")
            
            with col2:
                st.markdown("#### Privacy Protection")
                st.metric("Uniqueness Reduction", f"{privacy.get('uniqueness_reduction', 0):.1%}")
                st.metric("Variance Preservation", f"{privacy.get('variance_preservation', 0):.1%}")
        
        st.subheader("Overall Assessment")
        if stats.get('correlation_preservation', 0) > 0.8 and privacy.get('uniqueness_reduction', 0) > 0.3:
            st.success("EXCELLENT - Strong privacy with high utility preservation!")
        elif stats.get('correlation_preservation', 0) > 0.6:
            st.info("GOOD - Balanced privacy-utility tradeoff")
        else:
            st.warning("FAIR - Consider adjusting anonymization parameters")

    def results_tab(self):
        st.header("📥 Step 5: Results & Download")
        
        if st.session_state.anonymized_df is None:
            st.warning("Please complete anonymization first.")
            return
        
        st.subheader("Anonymized Data Preview")
        st.dataframe(st.session_state.anonymized_df.head(10), use_container_width=True)
        
        st.subheader("Download Anonymized Data")
        csv = st.session_state.anonymized_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        button_style = """
        background: linear-gradient(45deg, #4a90e2, #8b5cf6, #4a90e2);
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        border-radius: 10px;
        font-weight: bold;
        margin: 10px 0;
        """
        
        href = f'<a href="data:file/csv;base64,{b64}" download="anonymized_data.csv" style="{button_style}">Download Anonymized CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        if st.session_state.quality_metrics is not None:
            st.subheader("Quality Assessment Results")
            self.show_quality_metrics(st.session_state.quality_metrics)

if __name__ == "__main__":
    app = DataAnonymizationApp()
    app.run()