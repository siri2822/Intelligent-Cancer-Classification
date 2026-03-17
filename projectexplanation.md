# Overview

This is a research-grade bioinformatics application for cancer classification using gene expression data. The system combines hybrid optimization (Genetic Algorithm + Particle Swarm Optimization) for intelligent gene selection with Transformer deep learning networks for classification. It processes high-dimensional genomic datasets (20,000+ genes) and reduces them to 100-200 most informative features while maintaining high classification accuracy.

The application provides an interactive Streamlit web interface for uploading datasets, running optimization pipelines, training models, and visualizing results with publication-quality outputs.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Application Structure

**Modular Pipeline Design**: The system follows a clear sequential pipeline architecture with six independent modules:
- `data_preprocessing.py` - Data loading and normalization
- `optimization.py` - Hybrid GA-PSO feature selection
- `transformer_model.py` - Deep learning classifier
- `evaluation.py` - Metrics computation
- `interpretability.py` - Gene importance analysis
- `app.py` - Streamlit web interface orchestrating all modules

**Rationale**: This modular approach allows each component to be developed, tested, and updated independently. It also enables researchers to use individual modules programmatically outside the web interface.

## Frontend Architecture

**Streamlit Dashboard**: Single-page application (`app.py`) with custom CSS styling for improved UX. Provides interactive controls for:
- File upload (CSV format)
- Synthetic data generation
- Hyperparameter configuration
- Real-time progress tracking
- Results visualization

**Alternatives Considered**: Flask/FastAPI + React frontend was considered but rejected in favor of Streamlit for rapid prototyping and research workflows.

**Pros**: Quick development, automatic reactivity, built-in state management
**Cons**: Limited customization compared to full web frameworks, not ideal for production-scale deployments

## Data Processing Pipeline

**Scikit-learn Based Preprocessing**: Uses StandardScaler for Z-score normalization and SimpleImputer for missing value handling. Train/test split with stratification ensures balanced class distribution.

**Design Choice**: Chose Z-score normalization over min-max scaling because gene expression data often contains outliers, and Z-score is more robust to extreme values while preserving relative relationships.

## Optimization Engine

**Hybrid GA-PSO Algorithm**: Custom implementation combining two metaheuristic algorithms:
- Genetic Algorithm for exploration (crossover, mutation)
- Particle Swarm Optimization for exploitation (velocity-based convergence)

**Fitness Function**: Uses MLPClassifier with cross-validation to evaluate gene subsets. This provides a quick approximation of subset quality without full Transformer training.

**Rationale**: Pure GA can get stuck in local optima; pure PSO converges too quickly. The hybrid approach balances exploration and exploitation, typically finding better gene subsets than either algorithm alone.

**Parameters**:
- Population size: 30 particles
- Generations: 50 iterations
- Target genes: 150 selected from 20,000+
- Crossover rate: 0.8, Mutation rate: 0.01

## Deep Learning Architecture

**Transformer Neural Network**: Uses TensorFlow/Keras to implement a Transformer-based classifier adapted for gene expression data (sequential tabular data, not text).

**Components**:
- Custom TransformerBlock layer with multi-head self-attention
- Positional embeddings (despite genes having no inherent order, this adds learnable position information)
- Feed-forward networks with residual connections
- Layer normalization and dropout for regularization

**Alternatives Considered**: CNNs and standard MLPs were considered. CNNs assume spatial locality which doesn't apply to gene data. Transformers were chosen because self-attention can learn arbitrary gene-gene interactions without locality assumptions.

**Pros**: Captures complex non-linear relationships, state-of-the-art for sequence modeling
**Cons**: Requires more training data than simpler models, computationally expensive

## Evaluation Framework

**Comprehensive Metrics Suite**: Implements sklearn-based evaluation covering:
- Classification metrics: Accuracy, F1-score (macro/weighted), ROC-AUC
- Efficiency metrics: Computation time, gene reduction percentage
- Visualization: Confusion matrices, ROC curves

**Multi-class ROC-AUC**: Uses one-vs-rest approach with label binarization for handling multiple cancer types.

## Interpretability System

**Gradient-Based Feature Importance**: Uses TensorFlow's GradientTape to compute feature attributions. Originally designed to use SHAP but switched to gradient methods due to installation complexity.

**Methods**:
- Vanilla gradients: ∂output/∂input
- Integrated gradients: More stable attribution via path integration

**Rationale**: Provides researchers with insights into which genes drive model predictions, essential for biomarker discovery and clinical interpretation.

# External Dependencies

## Core Machine Learning Frameworks

- **TensorFlow/Keras**: Deep learning framework for Transformer model implementation
- **Scikit-learn**: Classical ML utilities (preprocessing, MLP for fitness evaluation, metrics)
- **NumPy/Pandas**: Data manipulation and numerical computing

## Web Interface

- **Streamlit**: Web application framework providing the interactive dashboard

## Visualization

- **Matplotlib**: Primary plotting library for charts and graphs
- **Seaborn**: Statistical visualization built on matplotlib for enhanced aesthetics

## Data Format

- **Input**: CSV files with gene expression values (columns) and class labels
- **Expected Format**: Rows = samples, columns = genes, one column for cancer type labels
- **No External Database**: Application operates on uploaded files, no persistent storage

## Future Considerations

The current architecture assumes in-memory processing. For production deployment with large-scale datasets, consider:
- Database integration (PostgreSQL) for dataset management
- Background task queue (Celery) for long-running optimizations
- Model versioning system (MLflow) for experiment tracking
- API layer (FastAPI) for programmatic access
