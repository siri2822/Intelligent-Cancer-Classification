# Intelligent Cancer Classification Using GA–PSO and Transformer Networks

A research-grade Python project that combines Hybrid Genetic Algorithm-Particle Swarm Optimization (GA-PSO) for gene selection with Transformer deep learning for cancer classification from gene expression data.

## 🎯 Project Overview

This system implements a state-of-the-art pipeline for cancer classification that addresses the high-dimensionality challenge in genomic data through:

1. **Hybrid GA-PSO Optimization**: Intelligent feature selection that combines:
   - **Genetic Algorithm (GA)**: Exploration through crossover and mutation
   - **Particle Swarm Optimization (PSO)**: Exploitation via velocity and position updates
   
2. **Transformer Neural Network**: Deep learning model with:
   - Multi-head self-attention mechanism
   - Positional embeddings for gene sequences
   - Dense classification layers

3. **Comprehensive Evaluation**: Complete metrics suite including accuracy, F1-score, ROC-AUC, computation time, and gene reduction analysis

4. **Model Interpretability**: Gradient-based gene importance analysis to identify influential biomarkers

---

> [!IMPORTANT]
> For a detailed technical breakdown of the system architecture and workflow diagrams, please refer to the **[Architecture Documentation (ARCHITECTURE.md)](./ARCHITECTURE.md)**.

---

## 🚀 Features

- **Flexible Data Input**: Upload custom CSV datasets or generate synthetic gene expression data
- **Advanced Optimization**: Hybrid GA-PSO reduces 20,000 genes to 100-200 most informative features
- **Deep Learning**: Transformer architecture captures complex gene interactions
- **Interactive Dashboard**: Streamlit web interface for end-to-end workflow
- **Research-Ready Outputs**: Publication-quality visualizations and comprehensive reports
- **Modular Architecture**: Clean, well-documented code structure

## 📁 Project Structure

```
.
├── app.py                      # Streamlit dashboard (main entry point)
├── data_preprocessing.py       # Data loading, cleaning, normalization
├── optimization.py             # Hybrid GA-PSO algorithm
├── transformer_model.py        # Transformer neural network
├── evaluation.py               # Metrics and evaluation
├── interpretability.py         # Gene importance analysis
└── README.md                   # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.11
- pip package manager

### Required Packages
All dependencies are already installed:
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- plotly
- streamlit

## 💻 Usage

### 1. Launch the Dashboard

```bash
streamlit run app.py --server.port 5000
```

### 2. Using the Web Interface

#### Step 1: Load Data
- A new **Test set proportion** slider in the sidebar lets you specify the fraction of samples reserved for testing.  If the chosen value would yield fewer test examples than there are classes, the app will automatically disable stratified splitting and warn you.
- **Option A**: Upload your gene expression CSV file
  - Format: Rows = samples, Columns = genes, Required column: `label`
- **Option B**: Generate synthetic data for testing
  - Configure samples, genes, and classes

#### Step 2: Run Gene Selection
- Set population size (10-50 particles)
- Set generations (10-100 iterations)
- Set target number of genes (50-500)
- Click "Run Gene Selection"

#### Step 3: Train Transformer
- Configure embedding dimension (64, 128, 256)
- Set number of attention heads (2, 4, 8)
- Set training epochs (10-100)
- Click "Train Model"

#### Step 4: Analyze Results
- View accuracy, F1-score, ROC-AUC
- Examine confusion matrix and ROC curves
- Analyze top influential genes
- Download comprehensive report

### 3. Using Individual Modules

#### Data Preprocessing
```python
from data_preprocessing import DataPreprocessor, generate_synthetic_dataset

# Generate synthetic dataset
dataset_path = generate_synthetic_dataset(
    n_samples=1000,
    n_genes=20000,
    n_classes=5
)

# Preprocess data
preprocessor = DataPreprocessor(random_state=42)
data = preprocessor.preprocess_pipeline(dataset_path, test_size=0.2)
```

#### Gene Selection with Hybrid GA-PSO
```python
from optimization import HybridGAPSO

optimizer = HybridGAPSO(
    n_particles=30,
    n_generations=50,
    n_selected_genes=150,
    random_state=42
)

results = optimizer.optimize(data['X_train'], data['y_train'])
selected_genes = results['selected_genes']
```

#### Train Transformer Model
```python
from transformer_model import GeneTransformerClassifier

# Select features
X_train_selected = data['X_train'][:, selected_genes]
X_test_selected = data['X_test'][:, selected_genes]

# Initialize and train
transformer = GeneTransformerClassifier(
    n_genes=len(selected_genes),
    n_classes=data['n_classes'],
    embed_dim=128,
    num_heads=4
)

transformer.train(
    X_train_selected, data['y_train'],
    X_test_selected, data['y_test'],
    epochs=50
)
```

#### Evaluate Model
```python
from evaluation import ModelEvaluator

y_pred = transformer.predict(X_test_selected)
y_pred_proba = transformer.predict_proba(X_test_selected)

evaluator = ModelEvaluator(n_classes=data['n_classes'])
metrics = evaluator.evaluate(data['y_test'], y_pred, y_pred_proba)
evaluator.compute_gene_reduction(20000, len(selected_genes))
```

#### Analyze Gene Importance
```python
from interpretability import GeneImportanceAnalyzer

analyzer = GeneImportanceAnalyzer(transformer.model, feature_names)
fig = analyzer.plot_top_genes(X_test_selected, top_k=20)
```

## 🧬 Algorithm Details

### Hybrid GA-PSO Optimization

**Genetic Algorithm Component:**
- **Selection**: Tournament selection
- **Crossover**: Single-point crossover (rate: 0.8)
- **Mutation**: Bit-flip mutation (rate: 0.01)

**PSO Component:**
- **Velocity Update**: v(t+1) = w·v(t) + c1·r1·(pbest - x) + c2·r2·(gbest - x)
- **Position Update**: Binary via sigmoid transformation
- **Parameters**: w=0.7, c1=1.5, c2=1.5

**Fitness Function:**
- Multi-layer Perceptron (MLP) classifier
- 3-fold cross-validation accuracy
- Early stopping for efficiency

### Transformer Architecture

```
Input (selected genes)
    ↓
Embedding Layer (project to embed_dim)
    ↓
Positional Encoding
    ↓
Transformer Blocks (×2)
    ├─ Multi-Head Self-Attention
    ├─ Layer Normalization
    ├─ Feed-Forward Network
    └─ Residual Connections
    ↓
Global Average Pooling
    ↓
Dense Layers (128 → 64)
    ↓
Softmax Output (cancer classes)
```

## 📊 Expected Results

**Gene Reduction:**
- Original: 20,000 genes
- Selected: 100-200 genes
- Reduction: ~99%

**Classification Performance:**
- Accuracy: 85-95% (dataset dependent)
- F1-Score: 0.80-0.93
- ROC-AUC: 0.88-0.96

**Computation Time:**
- Gene Selection: 2-10 minutes
- Model Training: 1-5 minutes
- Inference: <1 second

## 📈 Evaluation Metrics

1. **Accuracy**: Overall classification correctness
2. **F1-Score**: Harmonic mean of precision and recall
3. **ROC-AUC**: Area under ROC curve (one-vs-rest for multi-class)
4. **Computation Time**: End-to-end processing time
5. **Gene Reduction %**: Feature dimensionality reduction

## 🔬 Research Applications

- **Cancer Subtype Classification**: Identify cancer types from gene expression
- **Biomarker Discovery**: Find influential genes for diagnosis
- **Precision Medicine**: Personalized treatment recommendations
- **Drug Target Identification**: Discover potential therapeutic targets

## 🎨 Visualizations

The system generates:
- **Optimization Convergence**: GA-PSO fitness over generations
- **Training Curves**: Loss and accuracy during training
- **Confusion Matrix**: Classification performance by class
- **ROC Curves**: One-vs-rest for each cancer type
- **Gene Importance**: Top 20 most influential genes
- **Expression Heatmaps**: Gene patterns across samples

## ⚙️ Configuration Options

### GA-PSO Parameters
- `n_particles`: Population size (default: 30)
- `n_generations`: Optimization iterations (default: 50)
- `n_selected_genes`: Target genes to select (default: 150)
- `crossover_rate`: GA crossover probability (default: 0.8)
- `mutation_rate`: GA mutation probability (default: 0.01)

### Transformer Parameters
- `embed_dim`: Embedding dimension (default: 128)
- `num_heads`: Attention heads (default: 4)
- `num_transformer_blocks`: Transformer layers (default: 2)
- `learning_rate`: Optimizer learning rate (default: 0.001)

## 🔍 CSV Data Format

Your gene expression CSV should have:
```
Gene_00001, Gene_00002, ..., Gene_20000, label
5.234,      3.891,      ..., 7.123,      0
4.567,      5.234,      ..., 3.456,      1
...
```

- **Rows**: Patient samples
- **Columns**: Gene expression values (any number of genes)
- **label**: Cancer class/type (integer: 0, 1, 2, ...)

## 🧪 Testing

To quickly test the system:

```bash
# This will generate synthetic data and run the full pipeline
python data_preprocessing.py
python optimization.py
python transformer_model.py
python interpretability.py
```

## 📝 Citation

If you use this code in your research, please cite:

```
@software{cancer_classification_ga_pso_transformer,
  title={Intelligent Cancer Classification Using GA-PSO and Transformer Networks},
  year={2025},
  description={Hybrid optimization and deep learning for gene expression analysis}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature selection algorithms (mRMR, LASSO, etc.)
- Ensemble models
- Cross-validation framework
- Support for real TCGA datasets
- Advanced hyperparameter tuning

## 📄 License

This project is provided for research and educational purposes.

## 🙏 Acknowledgments

- Inspired by advances in computational biology and deep learning
- Built with TensorFlow, scikit-learn, and Streamlit
- Gene expression analysis methodologies from cancer genomics research

## 📧 Support

For questions or issues:
1. Check the example usage in individual module files
2. Review the Streamlit dashboard help sections
3. Examine the detailed comments in source code

---

**Note**: This is a research-grade implementation. For clinical applications, additional validation, regulatory compliance, and domain expert review are required.
#
