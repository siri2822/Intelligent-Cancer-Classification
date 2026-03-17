"""
Model Interpretability Module

This module provides gene importance analysis using gradient-based methods
as an alternative to SHAP (which has installation issues).

Methods:
- Gradient-based feature importance
- Integrated gradients
- Visualization of top influential genes
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def setup_dark_plot_style():
    """Configure matplotlib for dark theme with high contrast."""
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


class GeneImportanceAnalyzer:
    """
    Analyze gene importance for Transformer model predictions.
    
    Uses gradient-based methods to compute feature importance:
    - Vanilla gradients: Direct gradient of output w.r.t. input
    - Integrated gradients: More stable attribution method
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize importance analyzer.
        
        Args:
            model: Trained Keras model
            feature_names (list, optional): Names of genes/features
        """
        self.model = model
        self.feature_names = feature_names
        
        if feature_names is not None:
            self.n_features = len(feature_names)
        else:
            self.n_features = None
    
    def compute_gradients(self, X, target_class=None):
        """
        Compute gradients of output w.r.t. input features.
        
        Args:
            X (array): Input samples
            target_class (int, optional): Target class for gradient computation
                                         If None, uses predicted class
            
        Returns:
            array: Gradients for each sample and feature
        """
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor, training=False)
            
            if target_class is None:
                # Use predicted class
                target_class = tf.argmax(predictions, axis=1)
            
            # Get scores for target class
            if isinstance(target_class, int):
                target_scores = predictions[:, target_class]
            else:
                # Multiple samples, different classes
                batch_indices = tf.range(tf.shape(predictions)[0])
                # Ensure both indices have matching dtype (int32) before stacking
                target_class = tf.cast(target_class, tf.int32)
                indices = tf.stack([batch_indices, target_class], axis=1)
                target_scores = tf.gather_nd(predictions, indices)
        
        # Compute gradients
        gradients = tape.gradient(target_scores, X_tensor)
        
        return gradients.numpy()
    
    def compute_integrated_gradients(self, X, baseline=None, steps=50):
        """
        Compute integrated gradients for more stable feature attribution.
        
        Integrated gradients interpolate between a baseline and the input,
        computing gradients along the path and integrating them.
        
        Args:
            X (array): Input samples
            baseline (array, optional): Baseline input (default: zeros)
            steps (int): Number of interpolation steps
            
        Returns:
            array: Integrated gradients for each sample and feature
        """
        if baseline is None:
            baseline = np.zeros_like(X)
        
        # Generate interpolated inputs
        alphas = np.linspace(0, 1, steps)
        
        integrated_grads = np.zeros_like(X)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (X - baseline)
            
            # Compute gradients at this point
            grads = self.compute_gradients(interpolated)
            
            # Accumulate
            integrated_grads += grads
        
        # Average and scale by input - baseline
        integrated_grads = integrated_grads / steps * (X - baseline)
        
        return integrated_grads
    
    def compute_feature_importance(self, X, y=None, method='integrated', 
                                   aggregate='mean_abs'):
        """
        Compute feature importance scores.
        
        Args:
            X (array): Input samples
            y (array, optional): True labels for class-specific importance
            method (str): 'gradient' or 'integrated'
            aggregate (str): How to aggregate across samples
                           'mean_abs', 'mean', 'sum_abs', 'sum'
            
        Returns:
            array: Feature importance scores
        """
        print(f"Computing feature importance using {method} method...")
        
        if method == 'integrated':
            attributions = self.compute_integrated_gradients(X)
        else:
            attributions = self.compute_gradients(X, target_class=y)
        
        # Aggregate across samples
        if aggregate == 'mean_abs':
            importance = np.abs(attributions).mean(axis=0)
        elif aggregate == 'mean':
            importance = attributions.mean(axis=0)
        elif aggregate == 'sum_abs':
            importance = np.abs(attributions).sum(axis=0)
        elif aggregate == 'sum':
            importance = attributions.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return importance
    
    def get_top_genes(self, X, top_k=20, method='integrated'):
        """
        Get top-k most important genes.
        
        Args:
            X (array): Input samples
            top_k (int): Number of top genes to return
            method (str): Importance computation method
            
        Returns:
            tuple: (gene_indices, importance_scores, gene_names)
        """
        importance = self.compute_feature_importance(X, method=method)
        
        # Get top-k indices
        top_indices = np.argsort(importance)[::-1][:top_k]
        top_scores = importance[top_indices]
        
        if self.feature_names is not None:
            top_names = [self.feature_names[i] for i in top_indices]
        else:
            top_names = [f"Gene_{i}" for i in top_indices]
        
        return top_indices, top_scores, top_names
    
    def plot_top_genes(self, X, top_k=20, method='integrated', 
                      save_path=None, title=None):
        """
        Visualize top-k most important genes.
        
        Args:
            X (array): Input samples
            top_k (int): Number of top genes to visualize
            method (str): Importance computation method
            save_path (str, optional): Path to save figure
            title (str, optional): Custom plot title
            
        Returns:
            matplotlib figure
        """
        setup_dark_plot_style()
        
        print(f"\nAnalyzing top {top_k} influential genes...")
        
        top_indices, top_scores, top_names = self.get_top_genes(
            X, top_k=top_k, method=method
        )
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.4)))
        
        # Reverse order for better visualization (highest at top)
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_scores[::-1], color='#00D9FF', edgecolor='#00FF9F', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], color='#FAFAFA')
        ax.set_xlabel('Importance Score', fontsize=12, color='#FAFAFA')
        ax.set_ylabel('Gene', fontsize=12, color='#FAFAFA')
        
        if title is None:
            title = f'Top {top_k} Most Influential Genes'
        ax.set_title(title, fontsize=14, fontweight='bold', color='#00D9FF')
        
        ax.grid(axis='x', alpha=0.3, color='#3E4147')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
            print(f"Gene importance plot saved to {save_path}")
        
        # Print summary
        n_print = min(10, len(top_names))
        print(f"\nTop {n_print} genes:")
        for i in range(n_print):
            print(f"  {i+1}. {top_names[i]}: {top_scores[i]:.4f}")
        
        return fig
    
    def plot_gene_heatmap(self, X, y, top_k=20, method='integrated',
                         save_path=None):
        """
        Plot heatmap of top genes across samples, grouped by class.
        
        Args:
            X (array): Input samples
            y (array): True labels
            top_k (int): Number of top genes
            method (str): Importance computation method
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib figure
        """
        setup_dark_plot_style()
        
        top_indices, top_scores, top_names = self.get_top_genes(
            X, top_k=top_k, method=method
        )
        
        # Get expression values for top genes
        X_top = X[:, top_indices]
        
        # Sort samples by class
        sort_idx = np.argsort(y)
        X_sorted = X_top[sort_idx]
        y_sorted = y[sort_idx]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(X_sorted.T, cmap='viridis', center=0,
                   yticklabels=top_names, cbar_kws={'label': 'Expression Level'},
                   ax=ax)
        
        # Add class boundaries with high contrast
        unique_classes = np.unique(y_sorted)
        boundaries = []
        for cls in unique_classes:
            boundary = np.where(y_sorted == cls)[0][-1] + 0.5
            boundaries.append(boundary)
            if boundary < len(y_sorted):
                ax.axvline(boundary, color='#00FF9F', linewidth=2, linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Samples (sorted by class)', fontsize=12, color='#FAFAFA')
        ax.set_ylabel('Top Genes', fontsize=12, color='#FAFAFA')
        ax.set_title(f'Expression Heatmap: Top {top_k} Genes', 
                    fontsize=14, fontweight='bold', color='#00D9FF')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
            print(f"Gene heatmap saved to {save_path}")
        
        return fig
    
    def plot_class_specific_importance(self, X, y, n_classes, top_k=10,
                                      method='integrated', save_path=None):
        """
        Plot gene importance for each class separately.
        
        Args:
            X (array): Input samples
            y (array): True labels
            n_classes (int): Number of classes
            top_k (int): Top genes per class
            method (str): Importance method
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 8))
        
        if n_classes == 1:
            axes = [axes]
        
        for cls in range(n_classes):
            # Get samples for this class
            class_mask = (y == cls)
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            # Compute importance
            top_indices, top_scores, top_names = self.get_top_genes(
                X_class, top_k=top_k, method=method
            )
            
            # Plot
            ax = axes[cls]
            y_pos = np.arange(len(top_names))
            ax.barh(y_pos, top_scores[::-1], color=f'C{cls}')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names[::-1])
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(f'Class {cls}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        fig.suptitle(f'Top {top_k} Genes per Class', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class-specific importance saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import generate_synthetic_dataset, DataPreprocessor
    from optimization import HybridGAPSO
    from transformer_model import GeneTransformerClassifier
    
    print("Setting up test...")
    
    # Generate small dataset for quick test
    dataset_path = generate_synthetic_dataset(
        n_samples=200, n_genes=500, n_classes=3
    )
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(dataset_path)
    
    # Quick gene selection
    optimizer = HybridGAPSO(n_particles=10, n_generations=3, n_selected_genes=50)
    results = optimizer.optimize(data['X_train'], data['y_train'], verbose=False)
    
    selected_genes = results['selected_genes']
    X_train_selected = data['X_train'][:, selected_genes]
    X_test_selected = data['X_test'][:, selected_genes]
    
    # Train transformer
    transformer = GeneTransformerClassifier(
        n_genes=len(selected_genes),
        n_classes=data['n_classes'],
        embed_dim=32,
        num_heads=2
    )
    transformer.train(X_train_selected, data['y_train'], epochs=5, verbose=0)
    
    # Analyze gene importance
    print("\nAnalyzing gene importance...")
    analyzer = GeneImportanceAnalyzer(
        transformer.model,
        feature_names=[data['feature_names'][i] for i in selected_genes]
    )
    
    fig = analyzer.plot_top_genes(X_test_selected, top_k=20)
    plt.show()
    
    print("\nInterpretability analysis complete!")
