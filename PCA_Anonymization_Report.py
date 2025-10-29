import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, ks_2samp
import warnings
warnings.filterwarnings('ignore')

class AnonymizationQualityReport:
    """Generates comprehensive quality report for anonymized data"""
    
    def __init__(self):
        self.original_df = None
        self.anonymized_df = None
        self.scaler = RobustScaler()  # More robust to outliers
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
        original_numeric = self.original_df[common_numeric_cols].copy()
        anonymized_numeric = self.anonymized_df[common_numeric_cols].copy()
        
        # Fill missing values with median (more robust)
        for col in common_numeric_cols:
            original_numeric[col] = original_numeric[col].fillna(original_numeric[col].median())
            anonymized_numeric[col] = anonymized_numeric[col].fillna(anonymized_numeric[col].median())
        
        return original_numeric, anonymized_numeric, common_numeric_cols
    
    def safe_correlation(self, x, y):
        """Safe correlation calculation that handles edge cases"""
        try:
            # Remove any remaining NaN or infinite values
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 0.0
            
            # Check if data is constant
            if np.std(x_clean) == 0 or np.std(y_clean) == 0:
                return 0.0
                
            corr, _ = pearsonr(x_clean, y_clean)
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def compute_pca_metrics(self, original_data, anonymized_data):
        """Compute PCA-based quality metrics with robust error handling"""
        print("Computing PCA metrics...")
        
        try:
            # Standardize the data using RobustScaler
            original_scaled = self.scaler.fit_transform(original_data)
            anonymized_scaled = self.scaler.transform(anonymized_data)
            
            # Handle any remaining infinite values
            original_scaled = np.nan_to_num(original_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            anonymized_scaled = np.nan_to_num(anonymized_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Fit PCA on original data
            self.pca_original.fit(original_scaled)
            original_components = self.pca_original.transform(original_scaled)
            
            # Fit PCA on anonymized data
            self.pca_anonymized.fit(anonymized_scaled)
            anonymized_components = self.pca_anonymized.transform(anonymized_scaled)
            
            # Calculate variance explained metrics
            original_variance = np.cumsum(self.pca_original.explained_variance_ratio_)
            anonymized_variance = np.cumsum(self.pca_anonymized.explained_variance_ratio_)
            
            # Component correlation (safe)
            component_corr = []
            n_components = min(5, len(original_components.T), len(anonymized_components.T))
            for i in range(n_components):
                corr = self.safe_correlation(original_components[:, i], anonymized_components[:, i])
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
        except Exception as e:
            print(f"Warning: PCA computation failed: {e}")
            # Return safe default values
            n_features = original_data.shape[1]
            default_variance = np.ones(min(10, n_features)) * 0.1
            return {
                'original_variance': default_variance,
                'anonymized_variance': default_variance,
                'original_components': np.zeros((len(original_data), 2)),
                'anonymized_components': np.zeros((len(anonymized_data), 2)),
                'component_correlation': [0.0, 0.0],
                'original_explained_variance': default_variance,
                'anonymized_explained_variance': default_variance
            }
    
    def compute_statistical_metrics(self, original_data, anonymized_data):
        """Compute statistical similarity metrics with robust calculations"""
        print("Computing statistical metrics...")
        
        metrics = {}
        
        try:
            # Mean and STD preservation
            metrics['mean_correlation'] = self.safe_correlation(original_data.mean(), anonymized_data.mean())
            metrics['std_correlation'] = self.safe_correlation(original_data.std(), anonymized_data.std())
            
            # Correlation matrix preservation
            orig_corr = original_data.corr().values
            anon_corr = anonymized_data.corr().values
            
            # Flatten matrices and remove diagonal
            mask = ~np.eye(orig_corr.shape[0], dtype=bool)
            orig_flat = orig_corr[mask]
            anon_flat = anon_corr[mask]
            
            metrics['correlation_preservation'] = self.safe_correlation(orig_flat, anon_flat)
            
            # Normalized Reconstruction error (using relative error)
            mse_values = []
            for col in original_data.columns:
                # Use relative MSE to handle scale differences
                mse = mean_squared_error(original_data[col], anonymized_data[col])
                # Normalize by variance of original data
                var_orig = original_data[col].var()
                if var_orig > 0:
                    normalized_mse = mse / var_orig
                else:
                    normalized_mse = mse
                mse_values.append(normalized_mse)
            
            metrics['mean_mse'] = np.mean(mse_values) if mse_values else 0
            metrics['max_mse'] = np.max(mse_values) if mse_values else 0
            
            # Distribution similarity (KS statistic)
            ks_stats = []
            for col in original_data.columns:
                try:
                    stat, _ = ks_2samp(original_data[col], anonymized_data[col])
                    ks_stats.append(stat)
                except:
                    ks_stats.append(1.0)  # Worst case if comparison fails
            
            metrics['mean_ks_statistic'] = np.mean(ks_stats) if ks_stats else 1.0
            
        except Exception as e:
            print(f"Warning: Statistical metrics computation failed: {e}")
            # Set default values
            metrics['mean_correlation'] = 0.0
            metrics['std_correlation'] = 0.0
            metrics['correlation_preservation'] = 0.0
            metrics['mean_mse'] = 1.0
            metrics['max_mse'] = 1.0
            metrics['mean_ks_statistic'] = 1.0
        
        return metrics
    
    def compute_privacy_metrics(self, original_data, anonymized_data):
        """Compute privacy protection metrics with safe calculations"""
        print("Computing privacy metrics...")
        
        metrics = {}
        
        try:
            # Uniqueness reduction (safe calculation)
            original_uniqueness = []
            anonymized_uniqueness = []
            
            for col in original_data.columns:
                try:
                    orig_unique = len(original_data[col].unique()) / len(original_data)
                    anon_unique = len(anonymized_data[col].unique()) / len(anonymized_data)
                    original_uniqueness.append(orig_unique)
                    anonymized_uniqueness.append(anon_unique)
                except:
                    original_uniqueness.append(1.0)
                    anonymized_uniqueness.append(1.0)
            
            orig_avg = np.mean(original_uniqueness)
            anon_avg = np.mean(anonymized_uniqueness)
            
            if orig_avg > 0:
                reduction = 1 - (anon_avg / orig_avg)
                # Clamp between 0 and 1
                metrics['uniqueness_reduction'] = max(0, min(1, reduction))
            else:
                metrics['uniqueness_reduction'] = 0.0
            
            # Information loss (variance preservation)
            try:
                total_variance_orig = np.sum(np.var(original_data, axis=0))
                total_variance_anon = np.sum(np.var(anonymized_data, axis=0))
                
                if total_variance_orig > 0:
                    preservation = total_variance_anon / total_variance_orig
                    # Clamp to reasonable range
                    metrics['variance_preservation'] = max(0, min(2, preservation))
                else:
                    metrics['variance_preservation'] = 1.0
            except:
                metrics['variance_preservation'] = 1.0
                
        except Exception as e:
            print(f"Warning: Privacy metrics computation failed: {e}")
            metrics['uniqueness_reduction'] = 0.0
            metrics['variance_preservation'] = 1.0
        
        return metrics
    
    def create_visualizations(self, pca_results, original_data, anonymized_data, output_prefix):
        """Generate quality assessment visualizations"""
        print("Generating visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anonymization Quality Assessment Report', fontsize=16, fontweight='bold')
        
        try:
            # 1. Scree Plot - Variance Explained
            n_components = min(10, len(pca_results['original_explained_variance']))
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
            comp_corr = pca_results['component_correlation'][:5]  # First 5 components
            axes[0, 1].bar(range(1, len(comp_corr) + 1), comp_corr, color='skyblue', alpha=0.7)
            axes[0, 1].set_xlabel('Principal Component')
            axes[0, 1].set_ylabel('Correlation Coefficient')
            axes[0, 1].set_title('Component-wise Correlation\n(Original vs Anonymized)')
            axes[0, 1].set_ylim(-0.1, 1.1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. First Two Components Scatter
            orig_components = pca_results['original_components']
            anon_components = pca_results['anonymized_components']
            
            # Sample points if too many for clear visualization
            n_samples = min(1000, len(orig_components))
            indices = np.random.choice(len(orig_components), n_samples, replace=False)
            
            axes[0, 2].scatter(orig_components[indices, 0], orig_components[indices, 1], 
                             alpha=0.6, c='blue', label='Original', s=30)
            axes[0, 2].scatter(anon_components[indices, 0], anon_components[indices, 1], 
                             alpha=0.6, c='red', label='Anonymized', s=30)
            axes[0, 2].set_xlabel('First Principal Component')
            axes[0, 2].set_ylabel('Second Principal Component')
            axes[0, 2].set_title('PCA Projection: Original vs Anonymized')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Normalized Reconstruction Error by Column
            mse_by_column = []
            columns = original_data.columns
            for col in columns:
                mse = mean_squared_error(original_data[col], anonymized_data[col])
                var_orig = original_data[col].var()
                if var_orig > 0:
                    normalized_mse = mse / var_orig
                else:
                    normalized_mse = mse
                mse_by_column.append(normalized_mse)
            
            # Plot only first 10 columns for clarity
            plot_cols = min(10, len(columns))
            axes[1, 0].bar(range(plot_cols), mse_by_column[:plot_cols], color='orange', alpha=0.7)
            axes[1, 0].set_xlabel('Columns')
            axes[1, 0].set_ylabel('Normalized MSE')
            axes[1, 0].set_title('Normalized Reconstruction Error\n(First 10 Columns)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Distribution Comparison (first 2 numeric columns)
            num_cols_to_plot = min(2, len(columns))
            for i in range(num_cols_to_plot):
                col = columns[i]
                axes[1, 1].hist(original_data[col], alpha=0.5, label=f'Original {col}', bins=20, density=True)
                axes[1, 1].hist(anonymized_data[col], alpha=0.5, label=f'Anonymized {col}', bins=20, density=True)
            axes[1, 1].set_xlabel('Values')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Distribution Comparison\n(First 2 Columns)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Correlation Matrix Heatmap Difference
            orig_corr = original_data.corr()
            anon_corr = anonymized_data.corr()
            corr_diff = np.abs(orig_corr - anon_corr)
            
            im = axes[1, 2].imshow(corr_diff, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            axes[1, 2].set_title('Correlation Matrix Difference\n(Hotter = More Change)')
            plt.colorbar(im, ax=axes[1, 2])
            
        except Exception as e:
            print(f"Warning: Visualization generation had issues: {e}")
            # Add error message to plot
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Visualization Error', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12, color='red')
        
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
        """Generate detailed text report with proper encoding and safe formatting"""
        
        report = []
        report.append("=" * 70)
        report.append("ANONYMIZATION QUALITY ASSESSMENT REPORT")
        report.append("=" * 70)
        report.append("")
        
        # PCA Metrics Section
        report.append("PCA-BASED QUALITY METRICS")
        report.append("-" * 40)
        
        try:
            # Variance explained comparison
            n_components_to_show = min(3, len(pca_results['original_variance']))
            for n in [2, 5, 10]:
                if n <= len(pca_results['original_variance']):
                    orig_var = pca_results['original_variance'][n-1] * 100
                    anon_var = pca_results['anonymized_variance'][n-1] * 100
                    if orig_var > 0:
                        retention = (anon_var / orig_var) * 100
                    else:
                        retention = 0
                    report.append(f"Top {n} PCs: {orig_var:5.1f}% (orig) -> {anon_var:5.1f}% (anon) | Retention: {retention:5.1f}%")
            
            # Component correlation
            avg_component_corr = np.mean(pca_results['component_correlation'])
            report.append(f"Average Component Correlation: {avg_component_corr:6.3f}")
        except:
            report.append("PCA metrics unavailable due to computation issues")
        
        report.append("")
        
        # Statistical Metrics Section
        report.append("STATISTICAL PRESERVATION METRICS")
        report.append("-" * 40)
        report.append(f"Mean Correlation:           {statistical_metrics['mean_correlation']:6.3f}")
        report.append(f"Std Deviation Correlation:  {statistical_metrics['std_correlation']:6.3f}")
        report.append(f"Correlation Matrix Pres.:   {statistical_metrics['correlation_preservation']:6.3f}")
        report.append(f"Avg Reconstruction Error:   {statistical_metrics['mean_mse']:6.3f}")
        report.append(f"Distribution Similarity:    {statistical_metrics['mean_ks_statistic']:6.3f}")
        report.append("")
        
        # Privacy Metrics Section
        report.append("PRIVACY PROTECTION METRICS")
        report.append("-" * 40)
        report.append(f"Uniqueness Reduction:       {privacy_metrics['uniqueness_reduction']:6.1%}")
        report.append(f"Variance Preservation:      {privacy_metrics['variance_preservation']:6.1%}")
        report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        try:
            # Calculate overall quality score (0-100) with safe calculations
            quality_components = []
            
            # Component correlation contribution (max 25 points)
            comp_corr = np.mean(pca_results['component_correlation'])
            quality_components.append(max(0, comp_corr * 25))
            
            # Correlation preservation (max 25 points)
            corr_pres = statistical_metrics['correlation_preservation']
            quality_components.append(max(0, corr_pres * 25))
            
            # Distribution similarity (max 25 points)
            ks_stat = statistical_metrics['mean_ks_statistic']
            quality_components.append(max(0, (1 - ks_stat) * 25))
            
            # Variance preservation (max 25 points)
            var_pres = privacy_metrics['variance_preservation']
            # Penalize both too low and too high variance preservation
            if var_pres > 2:  # If variance increased too much
                var_score = 25 * (2 / var_pres)
            else:
                var_score = min(25, var_pres * 12.5)  # Normal scaling
            quality_components.append(var_score)
            
            quality_score = sum(quality_components)
            quality_score = max(0, min(100, quality_score))  # Clamp to 0-100
            
            # Privacy score calculation
            privacy_components = []
            
            # Uniqueness reduction (max 50 points)
            uniqueness_red = privacy_metrics['uniqueness_reduction']
            privacy_components.append(uniqueness_red * 50)
            
            # Correlation reduction (max 50 points) - lower correlation = better privacy
            corr_pres = statistical_metrics['correlation_preservation']
            privacy_components.append((1 - abs(corr_pres)) * 50)
            
            privacy_score = sum(privacy_components)
            privacy_score = max(0, min(100, privacy_score))
            
            report.append(f"Data Utility Score:         {quality_score:6.1f}/100")
            report.append(f"Privacy Protection Score:   {privacy_score:6.1f}/100")
            
            # Overall assessment
            if quality_score >= 80 and privacy_score >= 70:
                assessment = "EXCELLENT - Strong privacy with high utility"
            elif quality_score >= 70 and privacy_score >= 60:
                assessment = "GOOD - Balanced privacy-utility tradeoff"
            elif quality_score >= 50 and privacy_score >= 50:
                assessment = "FAIR - Acceptable with room for improvement"
            elif quality_score >= 30:
                assessment = "POOR - Significant utility loss or weak privacy"
            else:
                assessment = "VERY POOR - Consider re-evaluating anonymization approach"
                
            report.append(f"Overall Assessment: {assessment}")
            
        except Exception as e:
            report.append("Overall assessment unavailable due to calculation issues")
            report.append(f"Error: {e}")
        
        report.append("")
        report.append("=" * 70)
        
        # Save text report
        report_text = "\n".join(report)
        try:
            with open(f"{output_prefix}_report.txt", "w", encoding='utf-8') as f:
                f.write(report_text)
        except:
            with open(f"{output_prefix}_report.txt", "w") as f:
                f.write(report_text)
        
        # Print to console
        print(report_text)
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
        
    except Exception as e:
        print(f"Error generating quality report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

    
# python quality_report.py original_data.csv anonymized_data.csv report_name