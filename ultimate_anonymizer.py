import pandas as pd
import numpy as np
import hashlib
import secrets
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

class DataAnonymizer:
    """Applies anonymization techniques based on risk tier classifications"""
    
    def __init__(self, k=5, l=2, epsilon=1.0, hash_salt=None, pca_variance_threshold=0.85):
        self.k = k  # k-anonymity parameter
        self.l = l  # l-diversity parameter
        self.epsilon = epsilon  # Differential privacy parameter
        self.hash_salt = hash_salt or secrets.token_hex(16)
        self.pca_variance_threshold = pca_variance_threshold
        self.pca_model = None
        self.scaler = None
        
    def apply_pca_anonymization(self, df, tier_columns):
        """Apply PCA-based anonymization to numeric columns"""
        print("Applying PCA-based anonymization...")
        
        # Select numeric columns from Tier 2 and Tier 3 for PCA
        numeric_columns = []
        for col in tier_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Only use columns with sufficient variance
                if df[col].std() > 0:
                    numeric_columns.append(col)
        
        if len(numeric_columns) < 2:
            print("  Not enough numeric columns for PCA")
            return df
            
        print(f"  Applying PCA to {len(numeric_columns)} numeric columns: {numeric_columns}")
        
        # Extract numeric data
        numeric_data = df[numeric_columns].copy()
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Standardize the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        # Apply PCA
        self.pca_model = PCA(n_components=self.pca_variance_threshold)
        pca_components = self.pca_model.fit_transform(scaled_data)
        
        print(f"  Original dimensions: {scaled_data.shape[1]}")
        print(f"  PCA components: {pca_components.shape[1]}")
        print(f"  Explained variance: {np.sum(self.pca_model.explained_variance_ratio_):.3f}")
        
        # Add differential privacy noise to PCA components
        noise_scale = 1.0 / self.epsilon
        dp_noise = np.random.laplace(0, noise_scale, pca_components.shape)
        noisy_components = pca_components + dp_noise
        
        # Transform back to original space
        reconstructed_data = self.pca_model.inverse_transform(noisy_components)
        
        # Scale back to original range
        reconstructed_original = self.scaler.inverse_transform(reconstructed_data)
        
        # Create a copy of the dataframe and replace the numeric columns
        pca_df = df.copy()
        for i, col in enumerate(numeric_columns):
            pca_df[col] = reconstructed_original[:, i]
        
        return pca_df
    
    def hash_with_salt(self, value):
        """One-way hashing with salt for direct identifiers"""
        if pd.isna(value):
            return value
        return hashlib.sha256(f"{value}{self.hash_salt}".encode()).hexdigest()[:16]
    
    def generalize_numeric(self, series, bins=5):
        """Generalize numeric values through binning"""
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                min_val, max_val = non_null.min(), non_null.max()
                if min_val == max_val:
                    return series
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(bins)]
                return pd.cut(series, bins=bin_edges, labels=labels, include_lowest=True)
        return series
    
    def generalize_categorical(self, series, max_categories=10):
        """Generalize categorical values through grouping"""
        if series.dtype == 'object':
            value_counts = series.value_counts()
            if len(value_counts) <= max_categories:
                return series
            
            top_categories = value_counts.head(max_categories - 1).index
            return series.apply(lambda x: x if x in top_categories else 'Other')
        return series
    
    def microaggregate(self, series, k=5):
        """Microaggregation for numeric data"""
        if not pd.api.types.is_numeric_dtype(series):
            return series
        
        non_null = series.dropna()
        if len(non_null) < k:
            return series
        
        # Simple microaggregation: sort and group into clusters of size k
        sorted_vals = non_null.sort_values().values
        clusters = [sorted_vals[i:i+k] for i in range(0, len(sorted_vals), k)]
        
        # Replace with cluster means
        result = series.copy()
        start_idx = 0
        for cluster in clusters:
            cluster_mean = np.mean(cluster)
            end_idx = start_idx + len(cluster)
            result.iloc[non_null.index[start_idx:end_idx]] = cluster_mean
            start_idx = end_idx
            
        return result
    
    def add_dp_noise(self, series, epsilon=1.0):
        """Add differential privacy noise to numeric data"""
        if not pd.api.types.is_numeric_dtype(series):
            return series
        
        non_null = series.dropna()
        if len(non_null) == 0:
            return series
        
        # Calculate sensitivity (assuming bounded domain)
        sensitivity = non_null.max() - non_null.min()
        if sensitivity == 0:
            return series
            
        scale = sensitivity / epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, scale, len(non_null))
        result = series.copy()
        result[non_null.index] = non_null + noise
        
        return result
    
    def tokenize_text(self, series, keep_ratio=0.3):
        """Tokenize and partially redact text data"""
        def tokenize_value(value):
            if pd.isna(value) or not isinstance(value, str):
                return value
            words = str(value).split()
            if len(words) <= 1:
                return "***"  # Full redaction for very short text
            keep_words = max(1, int(len(words) * keep_ratio))
            kept = words[:keep_words]
            redacted = ["***"] * (len(words) - keep_words)
            return " ".join(kept + redacted)
        
        return series.apply(tokenize_value)
    
    def suppress_rare_values(self, series, min_frequency=0.05):
        """Suppress rare values that appear less than min_frequency"""
        if series.dtype == 'object':
            value_counts = series.value_counts(normalize=True)
            rare_values = value_counts[value_counts < min_frequency].index
            return series.apply(lambda x: '***' if x in rare_values else x)
        return series
    
    def apply_tier1_anonymization(self, series, column_name):
        """Apply Tier 1 anonymization - remove or hash direct identifiers"""
        print(f"  Applying Tier 1 anonymization to: {column_name}")
        
        # For direct identifiers, use hashing
        if any(keyword in column_name.lower() for keyword in 
               ['name', 'email', 'phone', 'ssn', 'id', 'address', 'credit', 'card']):
            return series.apply(self.hash_with_salt)
        else:
            # For high-risk sensitive data, use strong anonymization
            if pd.api.types.is_numeric_dtype(series):
                return self.add_dp_noise(series, self.epsilon)
            else:
                return self.tokenize_text(series, keep_ratio=0.2)
    
    def apply_tier2_anonymization(self, series, column_name):
        """Apply Tier 2 anonymization - generalization and k-anonymity"""
        print(f"  Applying Tier 2 anonymization to: {column_name}")
        
        if pd.api.types.is_numeric_dtype(series):
            # For numeric quasi-identifiers, use generalization or microaggregation
            if any(keyword in column_name.lower() for keyword in ['age', 'salary', 'income', 'birth']):
                return self.generalize_numeric(series, bins=5)
            else:
                return self.microaggregate(series, k=self.k)
        else:
            # For categorical quasi-identifiers, use generalization
            return self.generalize_categorical(series, max_categories=10)
    
    def apply_tier3_anonymization(self, series, column_name):
        """Apply Tier 3 anonymization - minimal or no changes"""
        print(f"  Applying Tier 3 anonymization to: {column_name}")
        
        # For non-identifying data, apply light anonymization or none
        if pd.api.types.is_numeric_dtype(series):
            # Add very light noise for numeric
            return self.add_dp_noise(series, epsilon=10.0)  # Weak privacy
        elif series.dtype == 'object':
            # For text, minimal tokenization
            return self.tokenize_text(series, keep_ratio=0.8)
        else:
            return series
    
    def anonymize_dataset(self, df, classification_df, use_pca=True):
        """Apply anonymization based on tier classification from CSV"""
        print("Starting anonymization process...")
        
        # First apply per-column anonymization
        anonymized_df = df.copy()
        tier_mapping = {}
        applied_techniques = {}
        
        # Create a mapping from column name to tier classification
        classification_map = {}
        for _, row in classification_df.iterrows():
            col_name = row['Column Name']
            tier = row['Risk Tier Classification']
            classification_map[col_name] = tier
        
        # Separate columns by tier for potential PCA
        tier2_columns = []
        tier3_columns = []
        
        for column_name in df.columns:
            if column_name in classification_map:
                tier = classification_map[column_name]
                tier_mapping[column_name] = tier
                
                original_series = anonymized_df[column_name]
                
                if 'Tier 1' in tier:
                    anonymized_series = self.apply_tier1_anonymization(original_series, column_name)
                    applied_techniques[column_name] = 'Hashing/DP Noise/Heavy Tokenization'
                elif 'Tier 2' in tier:
                    anonymized_series = self.apply_tier2_anonymization(original_series, column_name)
                    applied_techniques[column_name] = 'Generalization/Microaggregation'
                    tier2_columns.append(column_name)
                elif 'Tier 3' in tier:
                    anonymized_series = self.apply_tier3_anonymization(original_series, column_name)
                    applied_techniques[column_name] = 'Light Noise/Minimal Tokenization'
                    tier3_columns.append(column_name)
                else:
                    # Default to Tier 3 if classification is not recognized
                    anonymized_series = self.apply_tier3_anonymization(original_series, column_name)
                    applied_techniques[column_name] = 'Default (Tier 3)'
                    tier3_columns.append(column_name)
                
                anonymized_df[column_name] = anonymized_series
            else:
                # If column not in classification, apply Tier 3 by default
                print(f"  Warning: {column_name} not found in classification. Applying Tier 3 by default.")
                anonymized_df[column_name] = self.apply_tier3_anonymization(
                    anonymized_df[column_name], column_name
                )
                applied_techniques[column_name] = 'Default (Tier 3) - Not in classification'
                tier3_columns.append(column_name)
        
        # Apply PCA-based anonymization if requested and we have suitable columns
        if use_pca and (len(tier2_columns) + len(tier3_columns) >= 2):
            print("\nApplying PCA-based utility-first anonymization...")
            pca_columns = tier2_columns + tier3_columns
            anonymized_df = self.apply_pca_anonymization(anonymized_df, pca_columns)
            applied_techniques['PCA'] = f'Applied to {len(pca_columns)} Tier 2/3 columns'
        
        return anonymized_df, tier_mapping, applied_techniques

def validate_classification_file(classification_df, data_columns):
    """Validate that classification file matches data columns"""
    classification_columns = set(classification_df['Column Name'])
    data_columns_set = set(data_columns)
    
    missing_in_classification = data_columns_set - classification_columns
    extra_in_classification = classification_columns - data_columns_set
    
    if missing_in_classification:
        print(f"Warning: These data columns are missing from classification: {missing_in_classification}")
    
    if extra_in_classification:
        print(f"Warning: These classification columns are not in data: {extra_in_classification}")
    
    return len(missing_in_classification) == 0

def main():
    """Main execution function"""
    if len(sys.argv) < 4:
        print("Usage: python anonymizer.py <actual_data.csv> <classification.csv> <output_anonymized.csv>")
        print("Optional parameters: [k=5] [l=2] [epsilon=1.0] [pca_threshold=0.85] [use_pca=true]")
        print("\nExample:")
        print("python anonymizer.py raw_data.csv classification.csv anonymized_data.csv 5 2 1.0 0.85 true")
        sys.exit(1)
    
    input_data_file = sys.argv[1]
    classification_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Extract optional parameters
    k = 5
    l = 2
    epsilon = 1.0
    pca_threshold = 0.85
    use_pca = True
    
    if len(sys.argv) > 4:
        k = int(sys.argv[4])
    if len(sys.argv) > 5:
        l = int(sys.argv[5])
    if len(sys.argv) > 6:
        epsilon = float(sys.argv[6])
    if len(sys.argv) > 7:
        pca_threshold = float(sys.argv[7])
    if len(sys.argv) > 8:
        use_pca = sys.argv[8].lower() == 'true'
    
    try:
        # Read the actual data file
        print(f"Reading actual data from: {input_data_file}")
        df = pd.read_csv(input_data_file)
        
        # Read the classification file
        print(f"Reading classification from: {classification_file}")
        classification_df = pd.read_csv(classification_file)
        
        # Validate that classification matches data columns
        if not validate_classification_file(classification_df, df.columns):
            print("Proceeding with available classifications...")
        
        print(f"\nDataset Information:")
        print(f"Original data shape: {df.shape}")
        print(f"Columns in data: {list(df.columns)}")
        print(f"Columns in classification: {list(classification_df['Column Name'])}")
        
        # Initialize anonymizer
        anonymizer = DataAnonymizer(k=k, l=l, epsilon=epsilon, pca_variance_threshold=pca_threshold)
        
        print(f"\nAnonymization Parameters:")
        print(f"k-anonymity: {k}")
        print(f"l-diversity: {l}")
        print(f"DP epsilon: {epsilon}")
        print(f"PCA variance threshold: {pca_threshold}")
        print(f"Use PCA: {use_pca}")
        print(f"Hash salt: {anonymizer.hash_salt}")
        
        # Apply anonymization
        anonymized_df, tier_mapping, techniques = anonymizer.anonymize_dataset(
            df, classification_df, use_pca=use_pca
        )
        
        # Save anonymized data
        print(f"\nSaving anonymized data to: {output_file}")
        anonymized_df.to_csv(output_file, index=False)
        
        # Generate summary report
        print(f"\n=== ANONYMIZATION SUMMARY ===")
        print(f"Original dataset: {df.shape}")
        print(f"Anonymized dataset: {anonymized_df.shape}")
        
        print(f"\nTier Distribution:")
        tier_counts = {}
        for col, tier in tier_mapping.items():
            tier_type = tier.split(':')[0]  # Extract "Tier 1", "Tier 2", etc.
            tier_counts[tier_type] = tier_counts.get(tier_type, 0) + 1
        
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count} columns")
        
        print(f"\nApplied Techniques:")
        for col, technique in techniques.items():
            if col != 'PCA':  # PCA is handled separately
                tier = tier_mapping.get(col, 'Not classified')
                print(f"  {col}: {technique}")
        
        if 'PCA' in techniques and use_pca:
            print(f"  PCA: {techniques['PCA']}")
            if anonymizer.pca_model is not None:
                print(f"    - Explained variance: {np.sum(anonymizer.pca_model.explained_variance_ratio_):.3f}")
                print(f"    - Components: {anonymizer.pca_model.n_components_}")
        
        print(f"\nSample of Original vs Anonymized Data:")
        sample_cols = min(3, len(df.columns))
        for i, col in enumerate(df.columns[:sample_cols]):
            if col in df and col in anonymized_df:
                orig_sample = df[col].dropna().head(2).tolist() if len(df[col].dropna()) > 0 else ['No data']
                anon_sample = anonymized_df[col].dropna().head(2).tolist() if len(anonymized_df[col].dropna()) > 0 else ['No data']
                print(f"  {col}:")
                print(f"    Original: {orig_sample}")
                print(f"    Anonymized: {anon_sample}")
        
        print(f"\nAnonymization completed successfully!")
        print(f"Output saved to: {output_file}")
        print(f"Remember to securely store the hash salt: {anonymizer.hash_salt}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



# # With PCA enabled (default)
# python anonymizer.py actual_data.csv classification.csv anonymized_output.csv

# # With custom PCA parameters
# python anonymizer.py actual_data.csv classification.csv anonymized_output.csv 5 2 1.0 0.90 true

# # Without PCA
# python anonymizer.py actual_data.csv classification.csv anonymized_output.csv 5 2 1.0 0.85 false

# python anonymizer.py actual_data.csv classification.csv anonymized_output.csv 5 2 1.0 0.90 true
#                                                        ↑   ↑   ↑    ↑     ↑
#                                                        k   l   ε    pca   use_pca