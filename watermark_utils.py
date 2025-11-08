import pandas as pd
import json
import base64
from datetime import datetime
import io
import numpy as np

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
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class WatermarkManager:
    def __init__(self):
        self.watermark_key = "__anonymization_metadata__"
        self.watermark_comment_prefix = "# ANONYMIZATION_WATERMARK:"
    
    def create_watermark(self, anonymization_params, quality_metrics=None, dataset_info=None):
        """Create comprehensive watermark metadata"""
        watermark = {
            "anonymized": True,
            "anonymization_timestamp": datetime.now().isoformat(),
            "anonymization_params": anonymization_params,
            "quality_metrics": quality_metrics or {},
            "dataset_info": dataset_info or {},
            "version": "1.0",
            "tool": "DataAnonymizationTool"
        }
        return watermark
    
    def detect_watermark(self, file_content):
        """Detect and extract watermark from file content"""
        try:
            # Handle both bytes and string input
            if isinstance(file_content, bytes):
                content_str = file_content.decode('utf-8')
            else:
                content_str = str(file_content)
            
            # Check for watermark in the first line
            lines = content_str.split('\n')
            if lines and lines[0].startswith(self.watermark_comment_prefix):
                try:
                    watermark_line = lines[0]
                    # Extract JSON part after the prefix
                    watermark_json = watermark_line[len(self.watermark_comment_prefix):].strip()
                    watermark_data = json.loads(watermark_json)
                    
                    # Return remaining content without watermark line
                    clean_content = '\n'.join(lines[1:])
                    return watermark_data, clean_content
                except json.JSONDecodeError as e:
                    print(f"Watermark JSON decode error: {e}")
                    return None, content_str
                except Exception as e:
                    print(f"Watermark extraction error: {e}")
                    return None, content_str
            
            return None, content_str
            
        except Exception as e:
            print(f"Watermark detection error: {e}")
            return None, file_content
    
    def create_watermarked_csv(self, df, watermark_data):
        """Create CSV with embedded watermark metadata"""
        try:
            # Preprocess data to ensure JSON serializability
            processed_watermark = self._preprocess_for_json(watermark_data)
            
            # Serialize with custom encoder
            watermark_json = json.dumps(processed_watermark, cls=NumpyEncoder, indent=2)
            
            # Create output with watermark
            output = io.StringIO()
            output.write(f"{self.watermark_comment_prefix} {watermark_json}\n")
            
            # Write dataframe to CSV
            df.to_csv(output, index=False)
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Watermark creation error: {e}")
            # Fallback: return regular CSV without watermark
            return df.to_csv(index=False)
    
    def _preprocess_for_json(self, data):
        """Recursively convert non-serializable types to JSON-compatible formats"""
        if isinstance(data, dict):
            return {key: self._preprocess_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._preprocess_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return [self._preprocess_for_json(item) for item in data]
        elif hasattr(data, 'dtype'):  # Handle numpy types safely
            if np.issubdtype(data.dtype, np.integer):
                return int(data)
            elif np.issubdtype(data.dtype, np.floating):
                return float(data)
            elif np.issubdtype(data.dtype, np.bool_):
                return bool(data)
            else:
                return str(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        elif isinstance(data, pd.Series):
            return data.tolist()
        elif data is pd.NaT:
            return None
        elif isinstance(data, float) and np.isnan(data):
            return None
        else:
            try:
                # Try to serialize, if it fails return string representation
                json.dumps(data)
                return data
            except:
                return str(data)
    
    def validate_watermark(self, watermark_data):
        """Validate watermark structure and content"""
        if not isinstance(watermark_data, dict):
            return False, "Watermark must be a dictionary"
        
        required_fields = ["anonymized", "anonymization_timestamp", "anonymization_params", "version"]
        
        for field in required_fields:
            if field not in watermark_data:
                return False, f"Missing required field: {field}"
        
        if not watermark_data.get("anonymized"):
            return False, "Watermark indicates data is not anonymized"
        
        return True, "Watermark is valid"
    
    def extract_dataset_info(self, df):
        """Extract basic dataset information for watermark"""
        return {
            "original_shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
            "null_count": int(df.isnull().sum().sum())
        }
    
    def create_download_link(self, watermarked_content, filename="anonymized_data.csv"):
        """Create a downloadable link for watermarked content"""
        try:
            if isinstance(watermarked_content, str):
                watermarked_content = watermarked_content.encode('utf-8')
            
            b64 = base64.b64encode(watermarked_content).decode()
            href = f'data:file/csv;base64,{b64}'
            return href
        except Exception as e:
            print(f"Download link creation error: {e}")
            return None
    
    def compare_watermarks(self, original_watermark, new_watermark):
        """Compare two watermarks for consistency analysis"""
        comparison = {
            "same_anonymization": original_watermark.get("anonymization_params") == new_watermark.get("anonymization_params"),
            "timestamp_difference": None,
            "parameter_changes": {}
        }
        
        # Calculate time difference
        try:
            orig_time = datetime.fromisoformat(original_watermark.get("anonymization_timestamp", ""))
            new_time = datetime.fromisoformat(new_watermark.get("anonymization_timestamp", ""))
            comparison["timestamp_difference"] = str(new_time - orig_time)
        except:
            comparison["timestamp_difference"] = "Unknown"
        
        # Compare parameters
        orig_params = original_watermark.get("anonymization_params", {})
        new_params = new_watermark.get("anonymization_params", {})
        
        for key in set(orig_params.keys()) | set(new_params.keys()):
            if orig_params.get(key) != new_params.get(key):
                comparison["parameter_changes"][key] = {
                    "original": orig_params.get(key),
                    "new": new_params.get(key)
                }
        
        return comparison

# Utility functions
def is_watermarked(file_content):
    """Quick check if content contains watermark"""
    manager = WatermarkManager()
    watermark_data, _ = manager.detect_watermark(file_content)
    return watermark_data is not None

def create_watermark_from_dataframe(df, anonymization_params, quality_metrics=None):
    """Convenience function to create watermark from dataframe"""
    manager = WatermarkManager()
    dataset_info = manager.extract_dataset_info(df)
    return manager.create_watermark(anonymization_params, quality_metrics, dataset_info)

def load_watermarked_dataframe(file_content):
    """Load dataframe from watermarked content"""
    manager = WatermarkManager()
    watermark_data, clean_content = manager.detect_watermark(file_content)
    
    if clean_content:
        df = pd.read_csv(io.StringIO(clean_content))
    else:
        df = pd.read_csv(io.StringIO(file_content))
    
    return df, watermark_data

def safe_json_serialize(obj):
    """Safely serialize an object to JSON with NumPy 2.0 compatibility"""
    return json.dumps(obj, cls=NumpyEncoder)