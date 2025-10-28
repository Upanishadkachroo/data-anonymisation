import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class AnonymizationQualityReport:
    """Generates comprehensive quality report for anonymized data"""
    
    def __init__(self):
        self.original_df = None
        self.anonymized_df = None
        self.scaler = StandardScaler()
        self.pca_original = PCA()
        self.pca_anonymized = PCA()
        
    def load_datasets(self, original_file, anonymized_file):
        """Load original and anonymized datasets"""
        print("Loading datasets...")
        self.original_df = pd.read_csv(original_file)
        self.anonymized_df = pd.read_csv(anonymized_file)
        
        # Validate datasets have same structure
        if self.original_df.shape != self.anonymized_df.shape:
            print("Warning: Dataset shapes don't match. Using common columns.")
            common_cols = list(set(self.original_df.columns) & set(self.anonymized_df.columns))
            self.original_df = self.original_df[common_cols]
            self.anonymized_df = self.anonymized_df[common_cols]
        
        print(f"Original data shape: {self.original_df.shape}")
        print(f"Anonymized data shape: {self.anonymized_df.shape}")
        
    def preprocess_data(self):
        """Preprocess data for analysis - handle missing values and select numeric columns"""
        # Select numeric columns only for PCA
        numeric_cols_original = self.original_df.select_dtypes(include=[np.number]).columns
        numeric_cols_anonymized = self.anonymized_df.select_dtypes(include=[np.number]).columns
        
        # Use common numeric columns
        common_numeric_cols = list(set(numeric_cols_original) & set(numeric_cols_anonymized))
        
        if len(common_numeric_cols) == 0:
            raise ValueError("No common numeric columns found for analysis")
        
        print(f"Using {len(common_numeric_cols)} numeric columns for analysis")
        
        # Extract numeric data and handle missing values
        original_numeric = self.original_df[common_numeric_cols].fillna(self.original_df[common_numeric_cols].mean())
        anonymized_numeric = self.anonymized_df[common_numeric_cols].fillna(self.anonymized_df[common_numeric_cols].mean())
        
        return original_numeric, anonymized_numeric, common_numeric_cols
    
    def compute_pca_metrics(self, original_data, anonymized_data):
        """Compute PCA-based quality metrics"""
        print("Computing PCA metrics...")
        
        # Standardize the data
        original_scaled = self.scaler.fit_transform(original_data)
        anonymized_scaled = self.scaler.transform(anonymized_data)
        
        # Fit PCA on original data
        self.pca_original.fit(original_scaled)
        original_components = self.pca_original.transform(original_scaled)
        
        # Fit PCA on anonymized data
        self.pca_anonymized.fit(anonymized_scaled)
        anonymized_components = self.pca_anonymized.transform(anonymized_scaled)
        
        # Calculate variance explained metrics
        original_variance = np.cumsum(self.pca_original.explained_variance_ratio_)
        anonymized_variance = np.cumsum(self.pca_anonymized.explained_variance_ratio_)
        
        # Component correlation
        component_corr = []
        for i in range(min(5, len(original_components.T))):
            corr, _ = pearsonr(original_components[:, i], anonymized_components[:, i])
            component_corr.append(corr)
        
        return {
            'original_variance': original_variance,
            'anonymized_variance': anonymized_variance,
            'original_components': original_components,
            'anonymized_components': anonymized_components,
            'component_correlation': component_corr,
            'original_explained_variance': self.pca_original.explained_variance_ratio_,
            'anonymized_explained_variance': self.pca_anonymized.explained_variance_ratio_
        }
    
    def compute_statistical_metrics(self, original_data, anonymized_data):
        """Compute statistical similarity metrics"""
        print("Computing statistical metrics...")
        
        metrics = {}
        
        # Mean and STD preservation
        metrics['mean_correlation'] = pearsonr(original_data.mean(), anonymized_data.mean())[0]
        metrics['std_correlation'] = pearsonr(original_data.std(), anonymized_data.std())[0]
        
        # Correlation matrix preservation
        orig_corr = original_data.corr().values
        anon_corr = anonymized_data.corr().values
        mask = ~np.eye(orig_corr.shape[0], dtype=bool)  # Exclude diagonal
        metrics['correlation_preservation'] = pearsonr(orig_corr[mask], anon_corr[mask])[0]
        
        # Reconstruction error (MSE)
        mse_values = []
        for col in original_data.columns:
            mse = mean_squared_error(original_data[col], anonymized_data[col])
            mse_values.append(mse)
        metrics['mean_mse'] = np.mean(mse_values)
        metrics['max_mse'] = np.max(mse_values)
        
        # Distribution similarity (KL divergence approximation)
        from scipy.stats import ks_2samp
        ks_stats = []
        for col in original_data.columns:
            stat, _ = ks_2samp(original_data[col], anonymized_data[col])
            ks_stats.append(stat)
        metrics['mean_ks_statistic'] = np.mean(ks_stats)
        
        return metrics
    
    def compute_privacy_metrics(self, original_data, anonymized_data):
        """Compute privacy protection metrics"""
        print("Computing privacy metrics...")
        
        metrics = {}
        
        # Uniqueness reduction (approximate k-anonymity)
        original_uniqueness = []
        anonymized_uniqueness = []
        
        for col in original_data.columns:
            orig_unique = len(original_data[col].unique()) / len(original_data)
            anon_unique = len(anonymized_data[col].unique()) / len(anonymized_data)
            original_uniqueness.append(orig_unique)
            anonymized_uniqueness.append(anon_unique)
        
        metrics['uniqueness_reduction'] = 1 - (np.mean(anonymized_uniqueness) / np.mean(original_uniqueness))
        
        # Information loss (simplified)
        total_variance_orig = np.sum(np.var(original_data, axis=0))
        total_variance_anon = np.sum(np.var(anonymized_data, axis=0))
        metrics['variance_preservation'] = total_variance_anon / total_variance_orig
        
        return metrics
    
    def create_visualizations(self, pca_results, original_data, anonymized_data, output_prefix):
        """Generate quality assessment visualizations"""
        print("Generating visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anonymization Quality Assessment Report', fontsize=16, fontweight='bold')
        
        # 1. Scree Plot - Variance Explained
        n_components = len(pca_results['original_explained_variance'])
        components = range(1, n_components + 1)
        
        axes[0, 0].plot(components, pca_results['original_variance'][:n_components], 
                       'b-', label='Original Data', linewidth=2, marker='o')
        axes[0, 0].plot(components, pca_results['anonymized_variance'][:n_components], 
                       'r-', label='Anonymized Data', linewidth=2, marker='s')
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Cumulative Variance Explained')
        axes[0, 0].set_title('Scree Plot: Variance Preservation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Component Correlation
        comp_corr = pca_results['component_correlation']
        axes[0, 1].bar(range(1, len(comp_corr) + 1), comp_corr, color='skyblue', alpha=0.7)
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Correlation Coefficient')
        axes[0, 1].set_title('Component-wise Correlation\n(Original vs Anonymized)')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. First Two Components Scatter
        orig_components = pca_results['original_components']
        anon_components = pca_results['anonymized_components']
        
        scatter1 = axes[0, 2].scatter(orig_components[:, 0], orig_components[:, 1], 
                                     alpha=0.6, c='blue', label='Original', s=30)
        scatter2 = axes[0, 2].scatter(anon_components[:, 0], anon_components[:, 1], 
                                     alpha=0.6, c='red', label='Anonymized', s=30)
        axes[0, 2].set_xlabel('First Principal Component')
        axes[0, 2].set_ylabel('Second Principal Component')
        axes[0, 2].set_title('PCA Projection: Original vs Anonymized')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Reconstruction Error by Column
        mse_by_column = []
        columns = original_data.columns
        for col in columns:
            mse = mean_squared_error(original_data[col], anonymized_data[col])
            mse_by_column.append(mse)
        
        axes[1, 0].bar(range(len(columns)), mse_by_column, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Columns')
        axes[1, 0].set_ylabel('Mean Squared Error')
        axes[1, 0].set_title('Reconstruction Error by Column')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Distribution Comparison (first 3 numeric columns)
        num_cols_to_plot = min(3, len(columns))
        for i in range(num_cols_to_plot):
            col = columns[i]
            axes[1, 1].hist(original_data[col], alpha=0.5, label=f'Original {col}', bins=20)
            axes[1, 1].hist(anonymized_data[col], alpha=0.5, label=f'Anonymized {col}', bins=20)
        axes[1, 1].set_xlabel('Values')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Correlation Matrix Heatmap Difference
        orig_corr = original_data.corr()
        anon_corr = anonymized_data.corr()
        corr_diff = np.abs(orig_corr - anon_corr)
        
        im = axes[1, 2].imshow(corr_diff, cmap='hot', interpolation='nearest')
        axes[1, 2].set_title('Correlation Matrix Difference\n(Hotter = More Change)')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_quality_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved as {output_prefix}_quality_report.png")
    
    def generate_report(self, original_file, anonymized_file, output_prefix="anonymization_quality"):
        """Generate comprehensive quality assessment report"""
        print("=== Anonymization Quality Assessment Report ===\n")
        
        # Load data
        self.load_datasets(original_file, anonymized_file)
        
        # Preprocess
        original_numeric, anonymized_numeric, numeric_cols = self.preprocess_data()
        
        # Compute metrics
        pca_results = self.compute_pca_metrics(original_numeric, anonymized_numeric)
        statistical_metrics = self.compute_statistical_metrics(original_numeric, anonymized_numeric)
        privacy_metrics = self.compute_privacy_metrics(original_numeric, anonymized_numeric)
        
        # Generate visualizations
        self.create_visualizations(pca_results, original_numeric, anonymized_numeric, output_prefix)
        
        # Generate text report
        self.generate_text_report(pca_results, statistical_metrics, privacy_metrics, output_prefix)
        
        return {
            'pca_metrics': pca_results,
            'statistical_metrics': statistical_metrics,
            'privacy_metrics': privacy_metrics
        }
    
    def generate_text_report(self, pca_results, statistical_metrics, privacy_metrics, output_prefix):
        """Generate detailed text report with proper encoding"""
        
        report = []
        report.append("=" * 70)
        report.append("ANONYMIZATION QUALITY ASSESSMENT REPORT")
        report.append("=" * 70)
        report.append("")
        
        # PCA Metrics Section
        report.append("PCA-BASED QUALITY METRICS")
        report.append("-" * 40)
        
        # Variance explained comparison - using ASCII arrow instead of Unicode
        for n_components in [2, 5, 10]:
            if n_components <= len(pca_results['original_variance']):
                orig_var = pca_results['original_variance'][n_components-1] * 100
                anon_var = pca_results['anonymized_variance'][n_components-1] * 100
                retention = (anon_var / orig_var) * 100
                report.append(f"Top {n_components} PCs: {orig_var:.1f}% -> {anon_var:.1f}% "
                             f"(Retention: {retention:.1f}%)")
        
        # Component correlation
        avg_component_corr = np.mean(pca_results['component_correlation'])
        report.append(f"Average Component Correlation: {avg_component_corr:.3f}")
        report.append("")
        
        # Statistical Metrics Section
        report.append("STATISTICAL PRESERVATION METRICS")
        report.append("-" * 40)
        report.append(f"Mean Correlation: {statistical_metrics['mean_correlation']:.3f}")
        report.append(f"Standard Deviation Correlation: {statistical_metrics['std_correlation']:.3f}")
        report.append(f"Correlation Matrix Preservation: {statistical_metrics['correlation_preservation']:.3f}")
        report.append(f"Average Reconstruction Error (MSE): {statistical_metrics['mean_mse']:.3f}")
        report.append(f"Maximum Reconstruction Error (MSE): {statistical_metrics['max_mse']:.3f}")
        report.append(f"Distribution Similarity (KS Statistic): {statistical_metrics['mean_ks_statistic']:.3f}")
        report.append("")
        
        # Privacy Metrics Section
        report.append("PRIVACY PROTECTION METRICS")
        report.append("-" * 40)
        report.append(f"Uniqueness Reduction: {privacy_metrics['uniqueness_reduction']:.1%}")
        report.append(f"Variance Preservation: {privacy_metrics['variance_preservation']:.1%}")
        report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        # Calculate overall quality score (0-100)
        quality_score = (
            avg_component_corr * 25 +
            statistical_metrics['correlation_preservation'] * 25 +
            (1 - statistical_metrics['mean_ks_statistic']) * 25 +
            privacy_metrics['variance_preservation'] * 25
        )
        
        privacy_score = (
            privacy_metrics['uniqueness_reduction'] * 50 +
            (1 - statistical_metrics['correlation_preservation']) * 50
        ) * 100
        
        report.append(f"Data Utility Score: {quality_score:.1f}/100")
        report.append(f"Privacy Protection Score: {privacy_score:.1f}/100")
        
        if quality_score >= 80 and privacy_score >= 70:
            assessment = "EXCELLENT - Strong privacy with high utility"
        elif quality_score >= 60 and privacy_score >= 50:
            assessment = "GOOD - Balanced privacy-utility tradeoff"
        elif quality_score >= 40:
            assessment = "FAIR - Acceptable with room for improvement"
        else:
            assessment = "POOR - Consider re-evaluating anonymization parameters"
            
        report.append(f"Overall Assessment: {assessment}")
        report.append("")
        report.append("=" * 70)
        
        # Save text report with proper encoding
        report_text = "\n".join(report)
        try:
            with open(f"{output_prefix}_report.txt", "w", encoding='utf-8') as f:
                f.write(report_text)
        except UnicodeEncodeError:
            # Fallback for systems that don't support UTF-8
            with open(f"{output_prefix}_report.txt", "w", encoding='cp1252') as f:
                f.write(report_text)
        
        # Print to console with safe encoding
        try:
            print(report_text)
        except UnicodeEncodeError:
            # Create ASCII-safe version for console
            safe_report = report_text.replace('→', '->').replace('–', '-')
            print(safe_report)
        
        print(f"\nDetailed report saved as {output_prefix}_report.txt")

def main():
    """Main function to generate anonymization quality report"""
    import sys
    import os
    
    if len(sys.argv) != 4:
        print("Usage: python quality_report.py <original_data.csv> <anonymized_data.csv> <output_prefix>")
        print("\nExample:")
        print("python quality_report.py original.csv anonymized.csv my_report")
        sys.exit(1)
    
    original_file = sys.argv[1]
    anonymized_file = sys.argv[2]
    output_prefix = sys.argv[3]
    
    try:
        # Generate quality report
        report_generator = AnonymizationQualityReport()
        metrics = report_generator.generate_report(original_file, anonymized_file, output_prefix)
        
        print(f"\nQuality assessment completed!")
        print(f"Report files generated:")
        print(f"  - {output_prefix}_quality_report.png (Visualizations)")
        print(f"  - {output_prefix}_report.txt (Detailed metrics)")
        
        # Print key findings
        print(f"\nKey Findings:")
        print(f"• Dataset analyzed: {report_generator.original_df.shape[0]} rows, {report_generator.original_df.shape[1]} columns")
        print(f"• Numeric columns used for analysis: {len(report_generator.preprocess_data()[2])}")
        
    except Exception as e:
        print(f"Error generating quality report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# python quality_report.py original_data.csv anonymized_data.csv report_name