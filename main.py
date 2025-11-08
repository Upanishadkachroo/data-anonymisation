import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sys
import os
import threading
import time

# Import your existing modules
from first import DataTierClassifier
from PCA_Anonymization_Report import AnonymizationQualityReport
from ultimate_anonymizer import DataAnonymizer, validate_classification_file

class StarryBackground(tk.Canvas):
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.width = width
        self.height = height
        self.configure(bg='#0a0a2a')  # Dark blue background
        self.stars = []
        self.create_stars()
        
    def create_stars(self):
        # Create random stars
        for _ in range(100):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            size = np.random.uniform(0.5, 2)
            brightness = np.random.uniform(0.3, 1.0)
            
            star = self.create_oval(
                x, y, x+size, y+size, 
                fill=self.brightness_to_color(brightness),
                outline=""
            )
            self.stars.append(star)
            
    def brightness_to_color(self, brightness):
        # Convert brightness to a color from dark blue to white
        r = int(200 * brightness)
        g = int(200 * brightness)
        b = int(255 * brightness)
        return f'#{r:02x}{g:02x}{b:02x}'

class DataAnonymizationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Anonymization Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a2a')
        
        # Initialize variables
        self.df = None
        self.classification_df = None
        self.anonymized_df = None
        self.quality_metrics = None
        
        # Create the UI
        self.create_ui()
        
    def create_ui(self):
        # Create main frame with starry background
        self.main_frame = tk.Frame(self.root, bg='#0a0a2a')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with title
        header_frame = tk.Frame(self.main_frame, bg='#0a0a2a')
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = tk.Label(
            header_frame, 
            text="Data Anonymization Tool", 
            font=("Arial", 24, "bold"),
            fg="white",
            bg='#0a0a2a'
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Protect sensitive data while preserving utility",
            font=("Arial", 12),
            fg="#a0a0ff",
            bg='#0a0a2a'
        )
        subtitle_label.pack(pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Style the notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#0a0a2a')
        style.configure('TNotebook.Tab', background='#1a1a3a', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#2a2a5a')])
        
        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.classification_tab = ttk.Frame(self.notebook)
        self.anonymization_tab = ttk.Frame(self.notebook)
        self.quality_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Data Setup")
        self.notebook.add(self.classification_tab, text="Risk Classification")
        self.notebook.add(self.anonymization_tab, text="Anonymization")
        self.notebook.add(self.quality_tab, text="Quality Assessment")
        self.notebook.add(self.results_tab, text="Results")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_classification_tab()
        self.setup_anonymization_tab()
        self.setup_quality_tab()
        self.setup_results_tab()
        
    def setup_data_tab(self):
        # Data setup tab
        setup_frame = tk.Frame(self.setup_tab, bg='#1a1a3a')
        setup_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection section
        file_frame = tk.LabelFrame(setup_frame, text="Data File Selection", 
                                  font=("Arial", 12, "bold"),
                                  bg='#1a1a3a', fg='white', padx=10, pady=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Data file selection
        data_file_frame = tk.Frame(file_frame, bg='#1a1a3a')
        data_file_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(data_file_frame, text="Data File:", 
                bg='#1a1a3a', fg='white', font=("Arial", 10)).pack(side=tk.LEFT)
        
        self.data_file_var = tk.StringVar()
        tk.Entry(data_file_frame, textvariable=self.data_file_var, 
                width=50, bg='#2a2a4a', fg='white', insertbackground='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(data_file_frame, text="Browse", 
                 command=self.browse_data_file, bg='#3a3a5a', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Load data button
        tk.Button(file_frame, text="Load Data", 
                 command=self.load_data, bg='#4a4a7a', fg='white',
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        # Data preview section
        preview_frame = tk.LabelFrame(setup_frame, text="Data Preview", 
                                     font=("Arial", 12, "bold"),
                                     bg='#1a1a3a', fg='white', padx=10, pady=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the data preview
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame, 
            height=15,
            bg='#2a2a4a',
            fg='white',
            insertbackground='white'
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_classification_tab(self):
        # Classification tab
        classification_frame = tk.Frame(self.classification_tab, bg='#1a1a3a')
        classification_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Classification controls
        controls_frame = tk.Frame(classification_frame, bg='#1a1a3a')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(controls_frame, text="Run Risk Classification", 
                 command=self.run_classification, bg='#4a4a7a', fg='white',
                 font=("Arial", 10, "bold")).pack(pady=5)
        
        # Classification results
        results_frame = tk.LabelFrame(classification_frame, text="Risk Classification Results", 
                                     font=("Arial", 12, "bold"),
                                     bg='#1a1a3a', fg='white', padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a treeview for classification results
        columns = ('Column Name', 'Risk Tier')
        self.classification_tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        for col in columns:
            self.classification_tree.heading(col, text=col)
            self.classification_tree.column(col, width=200)
        
        # Add scrollbar to treeview
        tree_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.classification_tree.yview)
        self.classification_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.classification_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Style the treeview
        style = ttk.Style()
        style.configure("Treeview", 
                       background="#2a2a4a",
                       foreground="white",
                       fieldbackground="#2a2a4a")
        style.map('Treeview', background=[('selected', '#4a4a7a')])
        
    def setup_anonymization_tab(self):
        # Anonymization tab
        anonymization_frame = tk.Frame(self.anonymization_tab, bg='#1a1a3a')
        anonymization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Anonymization parameters
        params_frame = tk.LabelFrame(anonymization_frame, text="Anonymization Parameters", 
                                    font=("Arial", 12, "bold"),
                                    bg='#1a1a3a', fg='white', padx=10, pady=10)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # k-anonymity parameter
        k_frame = tk.Frame(params_frame, bg='#1a1a3a')
        k_frame.pack(fill=tk.X, pady=5)
        tk.Label(k_frame, text="k-anonymity:", bg='#1a1a3a', fg='white').pack(side=tk.LEFT)
        self.k_var = tk.StringVar(value="5")
        tk.Entry(k_frame, textvariable=self.k_var, width=10, bg='#2a2a4a', fg='white').pack(side=tk.LEFT, padx=5)
        
        # epsilon parameter
        epsilon_frame = tk.Frame(params_frame, bg='#1a1a3a')
        epsilon_frame.pack(fill=tk.X, pady=5)
        tk.Label(epsilon_frame, text="DP Epsilon:", bg='#1a1a3a', fg='white').pack(side=tk.LEFT)
        self.epsilon_var = tk.StringVar(value="1.0")
        tk.Entry(epsilon_frame, textvariable=self.epsilon_var, width=10, bg='#2a2a4a', fg='white').pack(side=tk.LEFT, padx=5)
        
        # PCA threshold
        pca_frame = tk.Frame(params_frame, bg='#1a1a3a')
        pca_frame.pack(fill=tk.X, pady=5)
        tk.Label(pca_frame, text="PCA Variance Threshold:", bg='#1a1a3a', fg='white').pack(side=tk.LEFT)
        self.pca_var = tk.StringVar(value="0.85")
        tk.Entry(pca_frame, textvariable=self.pca_var, width=10, bg='#2a2a4a', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Use PCA checkbox
        self.use_pca_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="Use PCA", variable=self.use_pca_var,
                      bg='#1a1a3a', fg='white', selectcolor='#2a2a4a').pack(pady=5)
        
        # Anonymization controls
        controls_frame = tk.Frame(anonymization_frame, bg='#1a1a3a')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(controls_frame, text="Run Anonymization", 
                 command=self.run_anonymization, bg='#4a4a7a', fg='white',
                 font=("Arial", 10, "bold")).pack(pady=5)
        
        # Output file selection
        output_frame = tk.Frame(controls_frame, bg='#1a1a3a')
        output_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(output_frame, text="Output File:", bg='#1a1a3a', fg='white').pack(side=tk.LEFT)
        self.output_file_var = tk.StringVar()
        tk.Entry(output_frame, textvariable=self.output_file_var, 
                width=40, bg='#2a2a4a', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Button(output_frame, text="Browse", 
                 command=self.browse_output_file, bg='#3a3a5a', fg='white').pack(side=tk.LEFT, padx=5)
        
    def setup_quality_tab(self):
        # Quality assessment tab
        quality_frame = tk.Frame(self.quality_tab, bg='#1a1a3a')
        quality_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Quality assessment controls
        controls_frame = tk.Frame(quality_frame, bg='#1a1a3a')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(controls_frame, text="Run Quality Assessment", 
                 command=self.run_quality_assessment, bg='#4a4a7a', fg='white',
                 font=("Arial", 10, "bold")).pack(pady=5)
        
        # Quality metrics display
        metrics_frame = tk.LabelFrame(quality_frame, text="Quality Metrics", 
                                     font=("Arial", 12, "bold"),
                                     bg='#1a1a3a', fg='white', padx=10, pady=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.metrics_text = scrolledtext.ScrolledText(
            metrics_frame, 
            height=20,
            bg='#2a2a4a',
            fg='white',
            insertbackground='white'
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_results_tab(self):
        # Results tab
        results_frame = tk.Frame(self.results_tab, bg='#1a1a3a')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary section
        summary_frame = tk.LabelFrame(results_frame, text="Anonymization Summary", 
                                     font=("Arial", 12, "bold"),
                                     bg='#1a1a3a', fg='white', padx=10, pady=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, 
            height=10,
            bg='#2a2a4a',
            fg='white',
            insertbackground='white'
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Data comparison section
        comparison_frame = tk.LabelFrame(results_frame, text="Data Comparison", 
                                        font=("Arial", 12, "bold"),
                                        bg='#1a1a3a', fg='white', padx=10, pady=10)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a treeview for data comparison
        columns = ('Column', 'Original Sample', 'Anonymized Sample')
        self.comparison_tree = ttk.Treeview(comparison_frame, columns=columns, show='headings')
        
        for col in columns:
            self.comparison_tree.heading(col, text=col)
        
        # Add scrollbar to treeview
        tree_scroll = ttk.Scrollbar(comparison_frame, orient=tk.VERTICAL, command=self.comparison_tree.yview)
        self.comparison_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.comparison_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Style the treeview
        style = ttk.Style()
        style.configure("Treeview", 
                       background="#2a2a4a",
                       foreground="white",
                       fieldbackground="#2a2a4a")
        style.map('Treeview', background=[('selected', '#4a4a7a')])
        
    def browse_data_file(self):
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.data_file_var.set(filename)
            
    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save Anonymized Data As",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.output_file_var.set(filename)
            
    def load_data(self):
        if not self.data_file_var.get():
            messagebox.showerror("Error", "Please select a data file first")
            return
            
        try:
            self.df = pd.read_csv(self.data_file_var.get())
            
            # Update preview
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, f"Dataset Shape: {self.df.shape}\n\n")
            self.preview_text.insert(tk.END, f"Columns: {list(self.df.columns)}\n\n")
            self.preview_text.insert(tk.END, "First 10 rows:\n")
            self.preview_text.insert(tk.END, self.df.head(10).to_string())
            
            messagebox.showinfo("Success", f"Data loaded successfully!\nShape: {self.df.shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def run_classification(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        # Run classification in a separate thread to avoid UI freezing
        def classify():
            try:
                classifier = DataTierClassifier()
                self.classification_df = classifier.analyze_dataset(self.df)
                
                # Update UI in main thread
                self.root.after(0, self.update_classification_results)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Classification failed: {str(e)}"))
                
        threading.Thread(target=classify).start()
        
    def update_classification_results(self):
        # Clear existing items
        for item in self.classification_tree.get_children():
            self.classification_tree.delete(item)
            
        # Add new items
        for _, row in self.classification_df.iterrows():
            self.classification_tree.insert('', tk.END, values=(row['Column Name'], row['Risk Tier']))
            
        messagebox.showinfo("Success", "Risk classification completed!")
        
    def run_anonymization(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        if self.classification_df is None:
            messagebox.showerror("Error", "Please run risk classification first")
            return
            
        if not self.output_file_var.get():
            messagebox.showerror("Error", "Please specify an output file")
            return
            
        # Run anonymization in a separate thread
        def anonymize():
            try:
                # Get parameters
                k = int(self.k_var.get())
                epsilon = float(self.epsilon_var.get())
                pca_threshold = float(self.pca_var.get())
                use_pca = self.use_pca_var.get()
                
                # Initialize anonymizer
                anonymizer = DataAnonymizer(
                    k=k, 
                    epsilon=epsilon, 
                    pca_variance_threshold=pca_threshold
                )
                
                # Apply anonymization
                self.anonymized_df, tier_mapping, techniques = anonymizer.anonymize_dataset(
                    self.df, self.classification_df, use_pca=use_pca
                )
                
                # Save anonymized data
                self.anonymized_df.to_csv(self.output_file_var.get(), index=False)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.update_anonymization_results(tier_mapping, techniques))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Anonymization failed: {str(e)}"))
                
        threading.Thread(target=anonymize).start()
        
    def update_anonymization_results(self, tier_mapping, techniques):
        # Update summary
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, f"Original dataset: {self.df.shape}\n")
        self.summary_text.insert(tk.END, f"Anonymized dataset: {self.anonymized_df.shape}\n\n")
        
        # Tier distribution
        tier_counts = {}
        for col, tier in tier_mapping.items():
            tier_type = tier.split(' ')[0] + ' ' + tier.split(' ')[1]  # Extract "Tier 1", "Tier 2", etc.
            tier_counts[tier_type] = tier_counts.get(tier_type, 0) + 1
            
        self.summary_text.insert(tk.END, "Tier Distribution:\n")
        for tier, count in tier_counts.items():
            self.summary_text.insert(tk.END, f"  {tier}: {count} columns\n")
            
        self.summary_text.insert(tk.END, "\nApplied Techniques:\n")
        for col, technique in techniques.items():
            if col != 'PCA':
                self.summary_text.insert(tk.END, f"  {col}: {technique}\n")
                
        if 'PCA' in techniques:
            self.summary_text.insert(tk.END, f"  PCA: {techniques['PCA']}\n")
            
        # Update comparison tree
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
            
        # Add sample comparisons for first few columns
        sample_cols = min(5, len(self.df.columns))
        for i, col in enumerate(self.df.columns[:sample_cols]):
            if col in self.df and col in self.anonymized_df:
                orig_sample = self.df[col].dropna().head(2).tolist() if len(self.df[col].dropna()) > 0 else ['No data']
                anon_sample = self.anonymized_df[col].dropna().head(2).tolist() if len(self.anonymized_df[col].dropna()) > 0 else ['No data']
                
                self.comparison_tree.insert('', tk.END, values=(
                    col, 
                    str(orig_sample), 
                    str(anon_sample)
                ))
                
        messagebox.showinfo("Success", "Anonymization completed!")
        self.notebook.select(self.results_tab)
        
    def run_quality_assessment(self):
        if self.df is None or self.anonymized_df is None:
            messagebox.showerror("Error", "Please complete anonymization first")
            return
            
        # Run quality assessment in a separate thread
        def assess_quality():
            try:
                # Create temporary files for quality assessment
                original_file = "temp_original.csv"
                anonymized_file = "temp_anonymized.csv"
                
                self.df.to_csv(original_file, index=False)
                self.anonymized_df.to_csv(anonymized_file, index=False)
                
                # Generate quality report
                report_generator = AnonymizationQualityReport()
                self.quality_metrics = report_generator.generate_report(
                    original_file, anonymized_file, "quality_report"
                )
                
                # Clean up temporary files
                os.remove(original_file)
                os.remove(anonymized_file)
                
                # Update UI in main thread
                self.root.after(0, self.update_quality_results)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Quality assessment failed: {str(e)}"))
                
        threading.Thread(target=assess_quality).start()
        
    def update_quality_results(self):
        # Update metrics text
        self.metrics_text.delete(1.0, tk.END)
        
        if self.quality_metrics:
            stats = self.quality_metrics['statistical_metrics']
            privacy = self.quality_metrics['privacy_metrics']
            
            self.metrics_text.insert(tk.END, "STATISTICAL PRESERVATION METRICS\n")
            self.metrics_text.insert(tk.END, "=" * 40 + "\n")
            self.metrics_text.insert(tk.END, f"Mean Correlation: {stats.get('mean_correlation', 0):.3f}\n")
            self.metrics_text.insert(tk.END, f"Standard Deviation Correlation: {stats.get('std_correlation', 0):.3f}\n")
            self.metrics_text.insert(tk.END, f"Correlation Matrix Preservation: {stats.get('correlation_preservation', 0):.3f}\n")
            self.metrics_text.insert(tk.END, f"Average Reconstruction Error (MSE): {stats.get('mean_mse', 0):.3f}\n")
            self.metrics_text.insert(tk.END, f"Distribution Similarity (KS Statistic): {stats.get('mean_ks_statistic', 0):.3f}\n\n")
            
            self.metrics_text.insert(tk.END, "PRIVACY PROTECTION METRICS\n")
            self.metrics_text.insert(tk.END, "=" * 40 + "\n")
            self.metrics_text.insert(tk.END, f"Uniqueness Reduction: {privacy.get('uniqueness_reduction', 0):.1%}\n")
            self.metrics_text.insert(tk.END, f"Variance Preservation: {privacy.get('variance_preservation', 0):.1%}\n")
            
        messagebox.showinfo("Success", "Quality assessment completed!")
        self.notebook.select(self.quality_tab)

def main():
    root = tk.Tk()
    app = DataAnonymizationUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
