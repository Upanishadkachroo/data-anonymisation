<<<<<<< HEAD
# 🛡️ Data Anonymization Dashboard

A comprehensive **web application** for anonymizing sensitive data while preserving utility.  
Built with **Streamlit**, this enterprise-grade tool ensures robust privacy through multiple anonymization techniques.
=======
# Data Anonymization Dashboard

A comprehensive **web application** for anonymizing sensitive data while preserving utility.  
Built with **Streamlit**, this enterprise-grade tool ensures robust privacy through multiple anonymization techniques.

---
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

---

<<<<<<< HEAD
## 🚀 Features

### 🧠 Core Functionality
=======
### Core Functionality
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
- **Multi-step Anonymization Pipeline** – Guided process from upload to export  
- **Risk Classification** – Automatic detection of sensitive columns  
- **Multiple Techniques:**  
  - k-Anonymity  
  - Differential Privacy  
  - PCA-based Dimensionality Reduction  
- **Quality Assessment** – Privacy-utility tradeoff metrics  
- **Watermark Protection** – Prevents re-anonymization  

<<<<<<< HEAD
### 🔐 Privacy Techniques
=======
### Privacy Techniques
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
- **k-Anonymity:** Each record indistinguishable from k-1 others  
- **Differential Privacy:** Adds calibrated noise for mathematical privacy guarantees  
- **PCA Transformation:** Reduces dimensionality while maintaining data structure  
- **Generalization:** Broader categories replace specific values  

---

<<<<<<< HEAD
## 📊 Data Quality Metrics
=======
## Data Quality Metrics
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
| Category | Metrics |
|-----------|----------|
| Statistical Preservation | Correlation, Mean Squared Error |
| Privacy Metrics | Uniqueness reduction, Variance preservation |
| Utility Assessment | Overall quality scoring & recommendations |

---

<<<<<<< HEAD
## ⚡ Quick Start
=======
## Quick Start
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

### Prerequisites
- Python 3.8+  
- pip package manager  

### Installation
```bash
git clone <repository-url>
cd data-anonymization
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run web_app.py
```
Then open your browser and go to **http://localhost:8501**.

---

<<<<<<< HEAD
## 🧰 Required Dependencies
=======
## Required Dependencies
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

---

<<<<<<< HEAD
## 📘 Usage Guide
=======
## Usage Guide
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

### Step 1: Data Setup
- Upload your CSV file (only original, non-anonymized data)  
- Automatic watermark detection prevents re-anonymization  
- View dataset overview and statistics  

### Step 2: Risk Classification
- Automated sensitivity analysis  
- Assigns tiers from **Tier 1 (Very High)** to **Tier 5 (Very Low)**  
- Visualize distribution of risk categories  

### Step 3: Anonymization
- Configure privacy parameters:  
  - `k-value`: 2–100  
  - `Epsilon`: 0.1–10.0  
  - `PCA Threshold`: 0.1–1.0  
- Apply selected techniques  
- Real-time anonymized data preview  

### Step 4: Quality Assessment
- Statistical preservation and privacy protection metrics  
- Visual comparison (original vs anonymized)  
- Automatic tuning recommendations  

### Step 5: Results & Export
- Download:  
<<<<<<< HEAD
  - ✅ Watermarked CSV (metadata included)  
  - 🧾 Plain CSV (standard format)  
=======
  - Watermarked CSV (metadata included)  
  - Plain CSV (standard format)  
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
- Generate **Quality Report (JSON)**  
- Save and reuse configurations  

---

<<<<<<< HEAD
## 🔒 Privacy Protection Features
=======
## Privacy Protection Features
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

### Watermark System
- Detects previously anonymized files  
- Blocks re-processing  
- Embeds anonymization parameters and timestamps  
- Maintains full audit trail  

### Security
- Data never leaves your local environment  
- Session-based cleanup  
- Configurable privacy budgets  
- Layered privacy mechanisms  

---

<<<<<<< HEAD
## ⚙️ Configuration
=======
## Configuration
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

| Parameter | Range | Default | Description |
|------------|--------|----------|--------------|
| k-Anonymity | 2–100 | 5 | Privacy level |
| Epsilon | 0.1–10.0 | 1.0 | Privacy budget |
| PCA Threshold | 0.1–1.0 | 0.85 | Variance threshold |
| Use PCA | Boolean | True | Enable PCA transformation |

### Risk Classification Tiers
| Tier | Risk Level | Description |
|------|-------------|-------------|
| 1 | Very High | Direct identifiers (SSN, Email, Phone) |
| 2 | High | Quasi-identifiers (Age, Zipcode, Gender) |
| 3 | Medium | Sensitive attributes (Salary, Diagnosis) |
<<<<<<< HEAD
| 4 | Low | Non-sensitive but unique attributes |
| 5 | Very Low | Public or aggregated data |

---

## 📦 Output Formats

### ✅ Watermarked CSV
=======
<!-- | 4 | Low | Non-sensitive but unique attributes |
| 5 | Very Low | Public or aggregated data | -->

---

## Output Formats

### Watermarked CSV
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
```
# ANONYMIZATION_WATERMARK: {"anonymized": true, "timestamp": "...", "params": {...}}
column1,column2,column3
value1,value2,value3
...
```

<<<<<<< HEAD
### 📑 Quality Report
=======
### Quality Report
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
JSON format containing:
- Statistical preservation scores  
- Privacy protection levels  
- Parameter effectiveness analysis  

---

<<<<<<< HEAD
## 🎯 Use Cases
=======
## Use Cases
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

### Healthcare
- Patient record anonymization for research  
- HIPAA-compliant data sharing  

### Finance
- Transaction anonymization for GDPR/CCPA compliance  
- Fraud detection dataset preparation  

### Research
- Survey & academic data anonymization  
- Public dataset creation  

### Enterprise
- Employee data protection  
- Secure data analytics and BI preparation  

---

<<<<<<< HEAD
## 🧠 Advanced Usage
=======
## Advanced Usage
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

### Custom Anonymization Rules
```python
anonymization_params = {
    "k_anonymity": 10,
    "epsilon": 0.5,
    "pca_threshold": 0.9,
    "custom_rules": {
        "email": "hash",
        "age": "generalize_5_years",
        "salary": "add_noise_10percent"
    }
}
```

### Batch Processing
```bash
for file in datasets/*.csv; do
    streamlit run web_app.py -- --input "$file" --output "anonymized_${file}"
done
```

---

<<<<<<< HEAD
## 🛠️ Technical Architecture
=======
## Technical Architecture
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

**Components**
- Streamlit Web Interface  
- Modular Anonymization Engine  
- Statistical Quality Assessor  
- Watermark Metadata Manager  

**Data Flow**
1. Input → CSV upload & validation  
2. Analysis → Risk classification  
3. Processing → Configurable anonymization pipeline  
4. Assessment → Privacy & utility metrics  
5. Output → Watermarked CSV + Report  

---

<<<<<<< HEAD
## 📈 Performance
=======
## Performance
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

- Handles datasets up to **1GB**  
- Optimized for efficient memory use  
- Preserves >80% correlation with ε < 1.0  

---
<<<<<<< HEAD

## 🤝 Contributing
=======
<!--
## Contributing
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771

We welcome contributions! Please refer to the **Contributing Guidelines**.

### Development Setup
```bash
git clone <repository>
cd data-anonymization
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### Testing
```bash
pytest tests/
python -m pytest --cov=src tests/
```

---
<<<<<<< HEAD

=======
-->
>>>>>>> ee6a7b95120c3dddc5957c5455ae739a9a67d771
**© 2025 Data Anonymization Dashboard**  
_Protecting privacy while preserving insight._
