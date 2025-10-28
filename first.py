import pandas as pd
import numpy as np
import scipy.stats as stats
import gzip
import re
import math
import sys
import os
import warnings
# Suppress warnings, especially related to setting values on a copy or
# statistical calculations on small samples.
warnings.filterwarnings('ignore')

class DataTierClassifier:
    """
    Analyzes columns of a DataFrame using a combination of statistical features,
    regex patterns, and keyword heuristics to classify them into three risk tiers.
    """
    def __init__(self):
        # --- Keyword Lists for Heuristics ---
        self.identifier_keywords = [
            'id', 'ssn', 'passport', 'driver', 'license', 'account', 'number',
            'employee_id', 'customer_id', 'patient_id', 'student_id', 'uuid',
            'guid', 'social_security', 'tax_id', 'national_id', 'credit_card',
            'card_number', 'iban', 'bank_account', 'phone', 'mobile', 'telephone',
            'email', 'address', 'full_name', 'name', 'first_name', 'last_name'
        ]
        
        self.high_risk_sensitive_keywords = [
            'salary', 'income', 'revenue', 'financial', 'credit_score', 'debt',
            'balance', 'transaction', 'payment', 'medical_record', 'diagnosis',
            'treatment', 'disease', 'health', 'medical', 'psychiatric', 'therapy',
            'hiv', 'aids', 'cancer', 'disability', 'password', 'pin', 'secret',
            'security_answer', 'biometric', 'fingerprint', 'retinal', 'dna',
            'genetic', 'sexual_orientation', 'religion', 'political', 'union'
        ]
        
        self.low_risk_sensitive_keywords = [
            'age', 'gender', 'marital_status', 'education', 'occupation',
            'employment', 'department', 'title', 'hobby', 'interest',
            'preference', 'purchase_history', 'browsing_history', 'cookie'
        ]
        
        self.quasi_keywords = [
            'zip', 'postal', 'birth', 'date', 'location', 'city', 'state',
            'country', 'region', 'area', 'coordinate', 'latitude', 'longitude',
            'ip_address', 'mac_address', 'device_id', 'browser', 'operating_system'
        ]
    
    def compute_numeric_features(self, column):
        """Compute numeric distributional features: cardinality, frequency, entropy, Gini."""
        clean_data = column.dropna()
        n = len(clean_data)
        
        if n == 0:
            return {
                'distinct_ratio': 0, 'top1_freq': 0, 'top5_freq': 0, 'entropy': 0,
                'norm_entropy': 0, 'gini': 0, 'is_numeric': False
            }
        
        # Basic counts
        d = len(clean_data.unique())
        distinct_ratio = d / n
        
        # Frequency analysis
        value_counts = clean_data.value_counts()
        top1_freq = value_counts.iloc[0] / n if len(value_counts) > 0 else 0
        top5_freq = value_counts.iloc[:5].sum() / n if len(value_counts) >= 5 else 1.0
        
        # Entropy
        probabilities = value_counts / n
        # Use a small value (1e-10) to avoid log2(0)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(d) if d > 0 else 1
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Gini impurity
        gini = 1 - np.sum(probabilities ** 2)
        
        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(clean_data)
        
        return {
            'distinct_ratio': distinct_ratio,
            'top1_freq': top1_freq,
            'top5_freq': top5_freq,
            'entropy': entropy,
            'norm_entropy': norm_entropy,
            'gini': gini,
            'is_numeric': is_numeric
        }
    
    def compute_skewness_kurtosis(self, column):
        """Compute skewness, kurtosis, and outlier ratio for numeric columns."""
        clean_data = column.dropna()
        
        # Check if data is numeric and sufficient for statistical calculation
        if len(clean_data) < 2 or not pd.api.types.is_numeric_dtype(clean_data):
            return {'skewness': 0, 'kurtosis': 0, 'outlier_ratio': 0}
        
        try:
            # Use scipy's statistics functions
            skewness = stats.skew(clean_data, nan_policy='omit')
            kurtosis = stats.kurtosis(clean_data, nan_policy='omit')
            
            # Outlier detection using 1.5 * IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            outlier_ratio = len(outliers) / len(clean_data)
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'outlier_ratio': outlier_ratio
            }
        except:
            return {'skewness': 0, 'kurtosis': 0, 'outlier_ratio': 0}
    
    def compute_string_features(self, column):
        """Compute string, pattern-based, and compression features."""
        clean_data = column.dropna().astype(str)
        n = len(clean_data)
        
        default_features = {
            'avg_length': 0, 'std_length': 0, 'numeric_ratio': 0, 'special_char_ratio': 0,
            'email_ratio': 0, 'phone_ratio': 0, 'ssn_ratio': 0, 'credit_card_ratio': 0,
            'token_count': 0, 'monotonic_ratio': 0, 'pattern_ratio': 0, 'compression_ratio': 1
        }
        
        if n == 0:
            return default_features
        
        # Length statistics
        lengths = clean_data.str.len()
        avg_length = lengths.mean()
        std_length = lengths.std() if n > 1 else 0
        
        # Numeric and special character ratios
        numeric_ratio = clean_data.str.match(r'^\d+$').mean()
        special_char_ratio = clean_data.str.contains(r'[^a-zA-Z0-9\s]').mean()
        
        # PII Pattern matching
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        # Simplified phone regex to catch common formats
        phone_pattern = r'^\+?1?\-?\.?\(?[0-9]{3}\)?\-?\.?[0-9]{3}\-?\.?[0-9]{4}$'
        ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'
        credit_card_pattern = r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$'
        
        email_ratio = clean_data.str.match(email_pattern).mean()
        phone_ratio = clean_data.str.match(phone_pattern).mean()
        ssn_ratio = clean_data.str.match(ssn_pattern).mean()
        credit_card_ratio = clean_data.str.match(credit_card_pattern).mean()
        
        # Token count
        token_count = clean_data.str.split(r'[\s@_]').str.len().mean()
        
        # Monotonic pattern detection (for sequential IDs/timestamps)
        monotonic_ratio = 0
        try:
            numeric_vals = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(numeric_vals) > 1:
                diff = np.diff(numeric_vals)
                # Check for strictly non-decreasing or strictly non-increasing
                if np.all(diff >= 0) or np.all(diff <= 0):
                    monotonic_ratio = len(numeric_vals) / n
        except:
            monotonic_ratio = 0
        
        # Simple pattern ratio (e.g., alphanumeric strings)
        pattern_ratio = clean_data.str.match(r'^[A-Za-z0-9]{8,}$').mean()
        
        # Compression ratio
        compression_ratio = 1
        try:
            # Join all cleaned strings for compression test
            text_data = ' '.join(clean_data).encode('utf-8')
            if len(text_data) > 0:
                compressed = gzip.compress(text_data)
                compression_ratio = len(compressed) / len(text_data)
        except:
            compression_ratio = 1
        
        return {
            'avg_length': avg_length, 'std_length': std_length, 'numeric_ratio': numeric_ratio, 
            'special_char_ratio': special_char_ratio, 'email_ratio': email_ratio, 
            'phone_ratio': phone_ratio, 'ssn_ratio': ssn_ratio, 'credit_card_ratio': credit_card_ratio, 
            'token_count': token_count, 'monotonic_ratio': monotonic_ratio, 
            'pattern_ratio': pattern_ratio, 'compression_ratio': compression_ratio
        }
    
    def keyword_match_score(self, column_name, keyword_list):
        """Calculates a score based on keyword presence in the column name."""
        col_lower = column_name.lower().replace('_', ' ').replace('-', ' ')
        max_score = 0.0
        for keyword in keyword_list:
            if keyword.lower() in col_lower:
                # Exact match or surrounded by boundaries gets a perfect score
                if col_lower == keyword.lower() or re.search(r'\b' + re.escape(keyword) + r'\b', col_lower):
                    return 1.0
                else:
                    # Partial match gets a decent score
                    max_score = max(max_score, 0.7)
        return max_score
    
    def compute_mutual_information(self, df, column):
        """
        Compute an approximation of mutual information (MI) with other columns.
        Uses average absolute correlation for numeric, and normalized value
        count variance for categorical/string.
        """
        clean_col = df[column].dropna()
        if len(clean_col) < 2:
            return 0.0
        
        if pd.api.types.is_numeric_dtype(clean_col):
            correlations = []
            for other_col in df.columns:
                if other_col != column and pd.api.types.is_numeric_dtype(df[other_col]):
                    try:
                        corr = abs(df[[column, other_col]].corr().iloc[0,1])
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except:
                        continue
            # Return the average correlation with other numeric fields
            return np.mean(correlations) if correlations else 0.0
        else:
            # Proxy for MI in categorical data: variance of normalized value counts.
            # High variance means some values predict the column heavily.
            value_counts = df[column].value_counts(normalize=True)
            return np.std(value_counts) if len(value_counts) > 0 else 0.0
    
    def calculate_comprehensive_scores(self, features, column_name):
        """Calculate the composite scores for the five base categories."""
        
        # --- Extract normalized features for scoring ---
        f1 = features['numeric']['distinct_ratio']
        f2 = features['numeric']['norm_entropy']
        f3 = 1 - features['numeric']['top1_freq']
        f4 = features['string']['numeric_ratio']
        
        # PII pattern score (max of email/phone/ssn/cc ratio)
        f5 = max(features['string']['email_ratio'], 
                 features['string']['phone_ratio'], 
                 features['string']['ssn_ratio'],
                 features['string']['credit_card_ratio'])
        
        # Keyword matches
        identifier_keyword = self.keyword_match_score(column_name, self.identifier_keywords)
        high_risk_sensitive_keyword = self.keyword_match_score(column_name, self.high_risk_sensitive_keywords)
        low_risk_sensitive_keyword = self.keyword_match_score(column_name, self.low_risk_sensitive_keywords)
        quasi_keyword = self.keyword_match_score(column_name, self.quasi_keywords)
        
        # Pattern and compression score (low compression implies high pattern/structure)
        pattern_score = features['string']['pattern_ratio'] * (1 - features['string']['compression_ratio'])
        
        # --- Calculate individual category scores (based on your formulas) ---
        
        # 1. Identifier Score (High distinctness, High entropy, Keyword, PII pattern)
        identifier_score = (0.4 * f1 + 0.3 * f2 + 0.15 * f3 + 
                            0.1 * f4 + 0.05 * f5 + 0.1 * identifier_keyword)
        
        # 2. High-Risk Sensitive Score (Keyword is dominant, plus MI and PII pattern)
        high_risk_sensitive_score = (0.3 * high_risk_sensitive_keyword + 
                                     0.25 * features['mi'] + 0.2 * identifier_keyword + 
                                     0.15 * f2 + 0.1 * features['string']['special_char_ratio'])
        
        # 3. Low-Risk Sensitive Score (Keyword, MI, and Gini)
        low_risk_sensitive_score = (0.3 * low_risk_sensitive_keyword + 
                                     0.25 * features['mi'] + 0.2 * quasi_keyword + 
                                     0.15 * f2 + 0.1 * features['numeric']['gini'])
        
        # 4. Quasi-Identifier Score (Medium distinctness, MI, and Pattern)
        quasi_identifier_score = (0.25 * f1 + 0.15 * f2 + 0.15 * f3 + 
                                  0.15 * features['mi'] + 0.15 * pattern_score + 
                                  0.15 * quasi_keyword)
        
        # 5. Non-Identifying Score (Inverse relationship to ID/Quasi/Sensitive scores, high Top-1 frequency)
        # Note: This is an inverse measure, so it needs careful interpretation
        non_identifying_score = (0.4 * (1 - max(identifier_score, quasi_identifier_score)) + 
                                 0.3 * (1 - max(high_risk_sensitive_score, low_risk_sensitive_score)) +
                                 0.3 * features['numeric']['top1_freq'])
        
        return {
            'identifier_score': identifier_score,
            'high_risk_sensitive_score': high_risk_sensitive_score,
            'low_risk_sensitive_score': low_risk_sensitive_score,
            'quasi_identifier_score': quasi_identifier_score,
            'non_identifying_score': non_identifying_score
        }
    
    def classify_tier(self, scores):
        """Classify column into Tier 1, 2, or 3 based on composite scores and thresholds."""
        
        # Get the highest scoring category
        max_score = max(scores.values())
        
        # Tier 1 check: Highest score belongs to Identifier or High-Risk Sensitive, above a strong threshold
        if (scores.get('identifier_score', 0) == max_score and max_score >= 0.5) or \
           (scores.get('high_risk_sensitive_score', 0) == max_score and max_score >= 0.45):
            return 'Tier 1 (Highest Risk)'
        
        # Tier 2 check: Highest score belongs to Quasi-Identifier or Low-Risk Sensitive, above a medium threshold
        elif (scores.get('quasi_identifier_score', 0) == max_score and max_score >= 0.35) or \
             (scores.get('low_risk_sensitive_score', 0) == max_score and max_score >= 0.35):
            return 'Tier 2 (Medium/High Risk)'
        
        # Tier 3 check: All other cases, or if Non-Identifying Score is dominant.
        else:
            return 'Tier 3 (Low Risk)'
    
    def analyze_dataset(self, df):
        """Main function to analyze entire dataset and classify into tiers."""
        results = []
        
        for column in df.columns:
            # 1. Compute all features
            numeric_features = self.compute_numeric_features(df[column])
            skewness_features = self.compute_skewness_kurtosis(df[column])
            string_features = self.compute_string_features(df[column])
            # Pass the entire DataFrame to MI calculation
            mi_score = self.compute_mutual_information(df, column)
            
            # 2. Combine features
            all_features = {
                'numeric': numeric_features,
                'skewness': skewness_features,
                'string': string_features,
                'mi': mi_score
            }
            
            # 3. Calculate comprehensive scores
            scores = self.calculate_comprehensive_scores(all_features, column)
            
            # 4. Classify into tier
            tier = self.classify_tier(scores)
            
            results.append({
                'Column Name': column,
                'Risk Tier': tier
            })
        
        return pd.DataFrame(results)

def main():
    """
    Main execution function. Reads file path from command line arguments,
    analyzes the data, and saves results to a CSV file.
    """
    if len(sys.argv) != 2:
        print("Usage: python data_tier_classifier_v2.py <input_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Read the data file
        df = pd.read_csv(input_file)
        
        # Initialize classifier
        classifier = DataTierClassifier()
        
        # Analyze dataset
        results = classifier.analyze_dataset(df)
        
        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_risk_tier_classification.csv"
        
        # Save results to CSV
        results.to_csv(output_file, index=False)
        
        # Print confirmation and summary
        print(f"Analysis completed successfully!")
        print(f"Results saved to: {output_file}")
        print(f"\nSummary of classifications:")
        print(results[['Column Name', 'Risk Tier']].to_string(index=False))
        
        # Print tier distribution
        print(f"\nTier Distribution:")
        tier_counts = results['Risk Tier'].value_counts()
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count} columns")
        
    except FileNotFoundError:
        print(f"Error: File not found at '{input_file}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty at '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()