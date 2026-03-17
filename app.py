"""
Streamlit Dashboard for Cancer Classification System

Interactive web application for:
- Uploading gene expression data
- Running Hybrid GA-PSO gene selection
- Training Transformer model
- Visualizing results and gene importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import sys
import traceback
import gc
import os

# Import our modules with error handling
try:
    from data_preprocessing import DataPreprocessor, generate_synthetic_dataset
    from optimization import HybridGAPSO
    from transformer_model import GeneTransformerClassifier
    from evaluation import ModelEvaluator
    from interpretability import GeneImportanceAnalyzer
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Cancer Classification using GA-PSO & Transformer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure matplotlib for dark theme FIRST
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#1E2127'
plt.rcParams['axes.edgecolor'] = '#FAFAFA'
plt.rcParams['axes.labelcolor'] = '#FAFAFA'
plt.rcParams['text.color'] = '#FAFAFA'
plt.rcParams['xtick.color'] = '#FAFAFA'
plt.rcParams['ytick.color'] = '#FAFAFA'
plt.rcParams['grid.color'] = '#3E4147'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.facecolor'] = '#1E2127'
plt.rcParams['legend.edgecolor'] = '#3E4147'

# Custom CSS for dark mode with high contrast
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #00D9FF;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #00FF9F;
    margin-top: 1.5rem;
    border-bottom: 2px solid #00FF9F;
    padding-bottom: 0.5rem;
}
/* Improved button styling */
.stButton > button {
    background-color: #00D9FF;
    color: #0E1117;
    border: none;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #00FF9F;
    box-shadow: 0 0 15px rgba(0, 255, 159, 0.5);
}
[data-testid="stMetricValue"] {
    color: #00D9FF;
    font-size: 1.8rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MEMORY MANAGEMENT & UTILITIES
# ============================================================================
def estimate_memory_usage(n_samples, n_features):
    """Estimate memory usage in MB for dataset."""
    # Roughly 8 bytes per float64 + overhead
    return (n_samples * n_features * 8) / (1024 * 1024)

def cleanup_memory():
    """Clean up unused memory."""
    gc.collect()

def check_memory_warning(n_samples, n_features, max_mb=2000):
    """Check if dataset exceeds safe memory limit."""
    estimated_mb = estimate_memory_usage(n_samples, n_features)
    if estimated_mb > max_mb:
        return True, estimated_mb
    return False, estimated_mb

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize all session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'genes_selected' not in st.session_state:
        st.session_state.genes_selected = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'transformer' not in st.session_state:
        st.session_state.transformer = None
    if 'history' not in st.session_state:
        st.session_state.history = None

initialize_session_state()


def main():
    # Title
    st.markdown('<p class="main-header">🧬 Intelligent Cancer Classification</p>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #A0A0A0; font-size: 1.1rem;">Using Hybrid GA-PSO Optimization and Transformer Networks</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.header("1. Data Loading")
    
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload CSV File", "Generate Synthetic Data"]
    )
    
    test_size = st.sidebar.slider(
        "Test set proportion",
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Fraction of data reserved for testing"
    )
    
    # ========== DATA LOADING SECTION ==========
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload gene expression CSV",
            type=['csv'],
            help="CSV file with genes as columns and 'label' column for classes"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("Load Data", key="load_data_btn"):
                try:
                    with st.spinner("Loading and preprocessing data..."):
                        # Write file properly
                        file_content = uploaded_file.getvalue()
                        file_size_mb = len(file_content) / (1024 * 1024)
                        
                        # Validate file has content
                        if file_size_mb == 0:
                            st.error("❌ Uploaded file is empty")
                            st.session_state.data_loaded = False
                        else:
                            if file_size_mb > 500:
                                st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take time...")
                            
                            # Write to temp file with proper closure
                            temp_path = "temp_data.csv"
                            with open(temp_path, "wb") as f:
                                f.write(file_content)
                                f.flush()  # Force flush to disk
                            
                            # Verify file was written
                            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                                st.error("❌ Failed to save uploaded file")
                                st.session_state.data_loaded = False
                            else:
                                time.sleep(0.5)  # Brief delay to ensure file is readable
                                
                                # Load and preprocess
                                preprocessor = DataPreprocessor(random_state=42)
                                data = preprocessor.preprocess_pipeline(temp_path, test_size=test_size)
                                
                                # Check memory usage
                                is_large, mem_mb = check_memory_warning(len(data['X_train']), data['n_features'])
                                if is_large:
                                    st.info(f"💾 Dataset size: {mem_mb:.1f} MB. Recommend using reduced gene target or sampling.")
                                
                                st.session_state.data = data
                                st.session_state.data_loaded = True
                                st.success("✅ Data loaded successfully!")
                                cleanup_memory()
                except ValueError as e:
                    st.error(f"❌ Data validation error: {str(e)}")
                    st.session_state.data_loaded = False
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    st.error(traceback.format_exc())
                    st.session_state.data_loaded = False
    
    else:  # Generate synthetic data
        st.sidebar.subheader("Synthetic Data Parameters")
        n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000, 100)
        n_genes = st.sidebar.slider("Number of genes", 1000, 20000, 5000, 1000)
        n_classes = st.sidebar.slider("Number of classes", 2, 10, 5, 1)
        n_informative = st.sidebar.slider("Informative genes", 50, 500, 200, 50,
            help="Number of genes with class-specific expression patterns")
        
        # Memory warning for synthetic data
        estimated_mb = estimate_memory_usage(n_samples, n_genes)
        if estimated_mb > 500:
            st.sidebar.warning(f"⚠️ Estimated size: {estimated_mb:.1f} MB")
        
        if st.sidebar.button("Generate Data", key="generate_data_btn"):
            try:
                with st.spinner("Generating synthetic dataset..."):
                    dataset_path = generate_synthetic_dataset(
                        n_samples=n_samples,
                        n_genes=n_genes,
                        n_classes=n_classes,
                        n_informative=n_informative,
                        output_file="synthetic_data.csv"
                    )
                    
                    # Verify file was created properly
                    time.sleep(0.5)
                    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
                        st.error("❌ Failed to generate synthetic dataset")
                        st.session_state.data_loaded = False
                    else:
                        preprocessor = DataPreprocessor(random_state=42)
                        data = preprocessor.preprocess_pipeline(dataset_path, test_size=test_size)
                        
                        st.session_state.data = data
                        st.session_state.data_loaded = True
                        st.success("✅ Synthetic data generated and loaded!")
                        cleanup_memory()
            except ValueError as e:
                st.error(f"❌ Data validation error: {str(e)}")
                st.session_state.data_loaded = False
            except Exception as e:
                st.error(f"❌ Error generating data: {str(e)}")
                st.error(traceback.format_exc())
                st.session_state.data_loaded = False
    
    # ========== GENE SELECTION SECTION ==========
    if st.session_state.data_loaded:
        st.sidebar.header("2. Gene Selection (GA-PSO)")
        st.sidebar.subheader("Optimization Parameters")
        
        max_genes = st.session_state.data.get('n_features', 500)
        default_genes = min(150, max_genes)
        
        # Check dataset size and adjust defaults
        n_train_samples = len(st.session_state.data['X_train'])
        is_large_dataset = n_train_samples > 10000 or max_genes > 15000
        
        # Use smaller defaults for large datasets
        default_particles = 20 if is_large_dataset else 30
        default_generations = 15 if is_large_dataset else 30
        
        n_particles = st.sidebar.slider("Population size", 10, 50, default_particles, 5, key="particles")
        n_generations = st.sidebar.slider("Generations", 5, 100, default_generations, 5, key="generations")
        n_selected_genes = st.sidebar.slider(
            "Target genes", 1, max_genes, default_genes, 1,
            help="Number of genes to keep after selection"
        )
        
        # Large dataset warning with recommendation
        if is_large_dataset:
            st.sidebar.warning("⚠️ Large dataset detected. Consider:")
            st.sidebar.caption("• Reduce generations (15-20)")
            st.sidebar.caption("• Reduce population size (10-20)")
            st.sidebar.caption("• Select fewer target genes")
        
        if st.sidebar.button("Run Gene Selection", key="run_gapso_btn"):
            try:
                with st.spinner("Running Hybrid GA-PSO optimization..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    X_train_opt = st.session_state.data['X_train']
                    y_train_opt = st.session_state.data['y_train']
                    
                    status_text.text(f"Optimizing with {n_particles} particles × {n_generations} generations...")
                    status_text.text(f"Variance pre-filtering and data subsampling applied automatically.")
                    
                    optimizer = HybridGAPSO(
                        n_particles=n_particles,
                        n_generations=n_generations,
                        n_selected_genes=n_selected_genes,
                        random_state=42
                    )
                    
                    results = optimizer.optimize(
                        X_train_opt, 
                        y_train_opt,
                        verbose=True
                    )
                    
                    st.session_state.optimization_results = results
                    st.session_state.genes_selected = True
                    st.success(f"✅ Selected {len(results['selected_genes'])} genes in {results['optimization_time']:.1f}s!")
                    progress_bar.progress(100)
                    cleanup_memory()
            except Exception as e:
                st.error(f"❌ Gene selection failed: {str(e)}")
                st.error(traceback.format_exc())
                st.session_state.genes_selected = False
    
    # ========== MODEL TRAINING SECTION ==========
    if st.session_state.genes_selected:
        st.sidebar.header("3. Train Transformer")
        st.sidebar.subheader("Model Parameters")
        
        embed_dim = st.sidebar.selectbox("Embedding dimension", [64, 128, 256], index=1)
        num_heads = st.sidebar.selectbox("Attention heads", [2, 4, 8], index=1)
        epochs = st.sidebar.slider("Training epochs", 10, 100, 50, 10)
        
        # Batch size recommendation based on dataset size
        n_train = len(st.session_state.data['X_train'])
        default_batch = 32
        if n_train > 100000:
            default_batch = 128
        elif n_train > 50000:
            default_batch = 64
        batch_size = st.sidebar.slider("Batch size", 16, 256, default_batch, 16)
        
        if st.sidebar.button("Train Model", key="train_model_btn"):
            try:
                with st.spinner("Training Transformer model..."):
                    data = st.session_state.data
                    results = st.session_state.optimization_results
                    selected_genes = results['selected_genes']
                    
                    X_train_selected = data['X_train'][:, selected_genes]
                    X_test_selected = data['X_test'][:, selected_genes]
                    
                    transformer = GeneTransformerClassifier(
                        n_genes=len(selected_genes),
                        n_classes=data['n_classes'],
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_transformer_blocks=2,
                        random_state=42
                    )
                    
                    progress_bar = st.progress(0)
                    history = transformer.train(
                        X_train_selected, data['y_train'],
                        X_test_selected, data['y_test'],
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    st.session_state.transformer = transformer
                    st.session_state.history = history
                    st.session_state.X_train_selected = X_train_selected
                    st.session_state.X_test_selected = X_test_selected
                    st.session_state.selected_genes = selected_genes
                    st.session_state.model_trained = True
                    st.success("✅ Model trained successfully!")
                    progress_bar.progress(100)
                    cleanup_memory()
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.session_state.model_trained = False
    
    # ========== MAIN CONTENT AREA ==========
    if not st.session_state.data_loaded:
        st.info("👈 Please load data from the sidebar to get started.")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Load Data**: Upload your gene expression CSV or generate synthetic data
        2. **Gene Selection**: Run Hybrid GA-PSO to select optimal genes
        3. **Train Model**: Train the Transformer neural network
        4. **View Results**: Analyze performance metrics and gene importance
        
        **Expected CSV Format:**
        - Rows: Samples/patients
        - Columns: Gene expression values
        - Required column: `label` (cancer class/type)
        
        ### 💾 Large Dataset Tips
        - **Reduce genes before upload**: Pre-filter to ~10,000-15,000 genes
        - **Adjust GA-PSO parameters**: Use 15-20 generations and 15-25 population size
        - **Reduce target genes**: Select 100-200 genes instead of 500+
        - **Increase batch size**: Use 64-128 for datasets >50K samples
        - **Monitor memory**: Keep total data size under 2GB RAM
        """)
        return
    
    # ========== DATA OVERVIEW ==========
    st.markdown('<p class="sub-header">📊 Data Overview</p>', unsafe_allow_html=True)
    
    data = st.session_state.data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Genes", data['n_features'])
    col2.metric("Training Samples", len(data['X_train']))
    col3.metric("Test Samples", len(data['X_test']))
    col4.metric("Classes", data['n_classes'])
    
    st.markdown("#### Class Distribution")
    train_dist = pd.Series(data['y_train']).value_counts().sort_index()
    test_dist = pd.Series(data['y_test']).value_counts().sort_index()
    dist_df = pd.DataFrame({'Training': train_dist, 'Testing': test_dist})
    st.bar_chart(dist_df)
    
    # ========== GENE SELECTION RESULTS ==========
    if st.session_state.genes_selected:
        st.markdown('<p class="sub-header">🧬 Gene Selection Results</p>', unsafe_allow_html=True)
        
        results = st.session_state.optimization_results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected Genes", results['n_selected'])
        col2.metric("Best Fitness", f"{results['best_fitness']:.4f}")
        reduction = (1 - results['n_selected'] / data['n_features']) * 100
        col3.metric("Gene Reduction", f"{reduction:.1f}%")
        col4.metric("Optimization Time", f"{results['optimization_time']:.1f}s")
        
        st.markdown("#### Optimization Progress")
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(results['fitness_history'], label='Best Fitness', linewidth=3, color='#00D9FF')
            ax.plot(results['mean_fitness_history'], label='Mean Fitness', linewidth=3, color='#00FF9F', alpha=0.8)
            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('Fitness', fontsize=12)
            ax.set_title('GA-PSO Convergence', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not plot fitness history: {str(e)}")
        
        with st.expander("View Selected Gene Indices"):
            selected_genes_list = results['selected_genes'].tolist()
            if 'feature_names' in data:
                feature_names = data['feature_names']
                selected_names = [feature_names[i] for i in selected_genes_list[:50]]
                st.write(f"First 50 selected genes: {', '.join(selected_names)}")
            else:
                st.write(f"First 50 genes: {selected_genes_list[:50]}")
    
    # ========== MODEL RESULTS ==========
    if st.session_state.model_trained:
        st.markdown('<p class="sub-header">🤖 Model Performance</p>', unsafe_allow_html=True)
        
        try:
            transformer = st.session_state.transformer
            X_test_selected = st.session_state.X_test_selected
            
            start_time = time.time()
            y_pred = transformer.predict(X_test_selected)
            y_pred_proba = transformer.predict_proba(X_test_selected)
            inference_time = time.time() - start_time
            
            unique_labels = np.unique(np.concatenate([data['y_train'], data['y_test']]))
            class_names = [f"Class {int(lbl)}" for lbl in unique_labels]
            evaluator = ModelEvaluator(n_classes=len(unique_labels), class_names=class_names)
            metrics = evaluator.evaluate(data['y_test'], y_pred, y_pred_proba)
            evaluator.compute_gene_reduction(
                data['n_features'],
                st.session_state.optimization_results['n_selected']
            )
            evaluator.add_timing(
                optimization_time=st.session_state.optimization_results['optimization_time'],
                training_time=transformer.training_time,
                inference_time=inference_time
            )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("F1-Score", f"{metrics['f1_macro']:.4f}")
            col3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A")
            col4.metric("Total Time", f"{evaluator.metrics['total_time']:.1f}s")
            
            st.markdown("#### Training History")
            fig = evaluator.plot_training_history(st.session_state.history)
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("#### Confusion Matrix")
            fig = evaluator.plot_confusion_matrix(data['y_test'], y_pred)
            st.pyplot(fig)
            plt.close(fig)
            
            if metrics['roc_auc']:
                st.markdown("#### ROC Curves")
                fig = evaluator.plot_roc_curves(data['y_test'], y_pred_proba)
                st.pyplot(fig)
                plt.close(fig)
            
            st.markdown('<p class="sub-header">🔍 Gene Importance Analysis</p>', 
                       unsafe_allow_html=True)
            
            from interpretability import GeneImportanceAnalyzer
            feature_names = [data['feature_names'][i] for i in st.session_state.selected_genes]
            analyzer = GeneImportanceAnalyzer(transformer.model, feature_names)
            
            max_top = len(st.session_state.selected_genes)
            default_top = min(20, max_top)
            top_k = st.slider(
                "Number of top genes to display",
                min_value=1,
                max_value=max_top,
                value=default_top,
                step=1,
            )
            
            with st.spinner("Computing gene importance..."):
                fig = analyzer.plot_top_genes(X_test_selected, top_k=top_k)
                st.pyplot(fig)
                plt.close(fig)
            
            st.markdown("#### Gene Expression Heatmap")
            with st.spinner("Generating heatmap..."):
                # For large datasets, sample data for visualization
                X_viz = X_test_selected
                y_viz = data['y_test']
                
                if len(X_test_selected) > 500:
                    sample_indices = np.random.choice(len(X_test_selected), 500, replace=False)
                    X_viz = X_test_selected[sample_indices]
                    y_viz = data['y_test'][sample_indices]
                    st.caption(f"Showing {len(sample_indices)} samples out of {len(X_test_selected)} for visualization")
                
                fig = analyzer.plot_gene_heatmap(X_viz, y_viz, top_k=20)
                st.pyplot(fig)
                plt.close(fig)
            
            cleanup_memory()
            
            st.markdown("#### Evaluation Summary")
            report = evaluator.generate_summary_report()
            st.code(report, language=None)
            
            st.markdown("#### Download Results")
            report_buffer = io.StringIO()
            report_buffer.write("CANCER CLASSIFICATION RESULTS\n")
            report_buffer.write("="*60 + "\n\n")
            report_buffer.write(report)
            report_buffer.write("\nSelected Genes:\n")
            for i, gene_idx in enumerate(st.session_state.selected_genes[:50]):
                report_buffer.write(f"{i+1}. {feature_names[i]}\n")
            
            st.download_button(
                label="Download Report",
                data=report_buffer.getvalue(),
                file_name="cancer_classification_report.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"❌ Error displaying results: {str(e)}")
            st.error(f"Details: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
