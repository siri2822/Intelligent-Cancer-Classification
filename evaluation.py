"""
Evaluation Metrics Module

This module provides comprehensive evaluation metrics for the cancer classification system:
- Accuracy
- F1-score (macro and weighted)
- ROC-AUC (one-vs-rest for multi-class)
- Computation time tracking
- Gene reduction percentage
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import time


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


class ModelEvaluator:
    """
    Comprehensive evaluation suite for cancer classification models.
    
    Tracks and computes:
    - Classification metrics (accuracy, F1, AUC)
    - Computational efficiency (time, gene reduction)
    - Confusion matrices and ROC curves
    """
    
    def __init__(self, n_classes, class_names=None):
        """
        Initialize evaluator.
        
        Args:
            n_classes (int): Number of classes
            class_names (list, optional): Names of classes for visualization
        """
        self.n_classes = n_classes
        self.class_names = class_names or [f"Class {i}" for i in range(n_classes)]
        
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Compute all evaluation metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array, optional): Predicted probabilities for ROC-AUC
            
        Returns:
            dict: Dictionary of all metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60 + "\n")
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        self.metrics['accuracy'] = accuracy
        
        # F1 Scores
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        self.metrics['f1_macro'] = f1_macro
        self.metrics['f1_weighted'] = f1_weighted
        
        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                if self.n_classes == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class: one-vs-rest
                    y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
                    roc_auc = roc_auc_score(y_true_bin, y_pred_proba, 
                                           average='macro', multi_class='ovr')
                self.metrics['roc_auc'] = roc_auc
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC: {e}")
                self.metrics['roc_auc'] = None
        else:
            self.metrics['roc_auc'] = None
        
        # Print metrics
        print("Classification Metrics:")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  F1-Score (Macro):  {f1_macro:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        if self.metrics['roc_auc'] is not None:
            print(f"  ROC-AUC (OvR):     {self.metrics['roc_auc']:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        # `classification_report` can raise a ValueError if the number of
        # provided `target_names` does not match the number of classes
        # actually present in `y_true`/`y_pred`.  This is common when a split
        # or random dataset happens to omit one of the labels that were
        # originally expected.  We'll handle that gracefully by aligning the
        # names with the unique labels detected or by falling back to the
        # report without names.
        try:
            # determine unique labels present in the data so we can pass them
            # explicitly; sklearn will sort them for us
            labels_present = np.unique(np.concatenate([y_true, y_pred]))
            # if class_names length matches the number of labels, use them
            if len(self.class_names) == len(labels_present):
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=labels_present,
                    target_names=self.class_names,
                )
            else:
                # generate or trim names to line up with the actual labels
                if len(self.class_names) > len(labels_present):
                    adjusted_names = [
                        self.class_names[int(lbl)] if int(lbl) < len(self.class_names) else f"Class {int(lbl)}"
                        for lbl in labels_present
                    ]
                else:
                    adjusted_names = [
                        self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                        for i in range(len(labels_present))
                    ]
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=labels_present,
                    target_names=adjusted_names,
                )
        except ValueError as e:
            # fallback: omit `target_names` altogether if all else fails
            print(f"Warning: classification_report name mismatch: {e}")
            report = classification_report(y_true, y_pred)

        print(report)
        
        return self.metrics
    
    def compute_gene_reduction(self, original_genes, selected_genes):
        """
        Calculate gene reduction percentage.
        
        Args:
            original_genes (int): Original number of genes
            selected_genes (int): Number of selected genes
            
        Returns:
            float: Percentage of genes reduced
        """
        reduction = (1 - selected_genes / original_genes) * 100
        self.metrics['gene_reduction_pct'] = reduction
        self.metrics['original_genes'] = original_genes
        self.metrics['selected_genes'] = selected_genes
        
        print(f"\nGene Selection Statistics:")
        print(f"  Original genes:    {original_genes}")
        print(f"  Selected genes:    {selected_genes}")
        print(f"  Reduction:         {reduction:.2f}%")
        
        return reduction
    
    def add_timing(self, optimization_time=0, training_time=0, inference_time=0):
        """
        Add computation time metrics.
        
        Args:
            optimization_time (float): Time for gene selection
            training_time (float): Time for model training
            inference_time (float): Time for inference
        """
        self.metrics['optimization_time'] = optimization_time
        self.metrics['training_time'] = training_time
        self.metrics['inference_time'] = inference_time
        self.metrics['total_time'] = optimization_time + training_time + inference_time
        
        print(f"\nComputational Time:")
        print(f"  Gene Selection:    {optimization_time:.2f} seconds")
        print(f"  Model Training:    {training_time:.2f} seconds")
        print(f"  Inference:         {inference_time:.2f} seconds")
        print(f"  Total Time:        {self.metrics['total_time']:.2f} seconds")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib figure
        """
        setup_dark_plot_style()
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax, annot_kws={'color': '#FAFAFA', 'fontweight': 'bold'})
        ax.set_xlabel('Predicted Label', fontsize=12, color='#FAFAFA')
        ax.set_ylabel('True Label', fontsize=12, color='#FAFAFA')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#00D9FF')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Predicted probabilities
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib figure
        """
        setup_dark_plot_style()
        
        # Binarize labels for one-vs-rest
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # High contrast colors for dark mode
        colors = ['#00D9FF', '#00FF9F', '#FF6B9D', '#FFD700', '#9D4EDD', '#06FFA5', '#FF006E']
        
        # Plot ROC curve for each class
        for i in range(self.n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=3, color=colors[i % len(colors)],
                   label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Random Classifier', alpha=0.7)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, color='#FAFAFA')
        ax.set_ylabel('True Positive Rate', fontsize=12, color='#FAFAFA')
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold', color='#00D9FF')
        ax.legend(loc='lower right', fontsize=10, facecolor='#1E2127', edgecolor='#3E4147')
        ax.grid(alpha=0.3, color='#3E4147')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
            print(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            history: Keras history object
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib figure
        """
        setup_dark_plot_style()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves with high contrast colors
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=3, color='#FF6B9D')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=3, color='#00D9FF')
        ax1.set_xlabel('Epoch', fontsize=12, color='#FAFAFA')
        ax1.set_ylabel('Loss', fontsize=12, color='#FAFAFA')
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold', color='#00D9FF')
        ax1.legend(fontsize=10, facecolor='#1E2127', edgecolor='#3E4147')
        ax1.grid(alpha=0.3, color='#3E4147')
        
        # Accuracy curves with high contrast colors
        ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=3, color='#00FF9F')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=3, color='#FFD700')
        ax2.set_xlabel('Epoch', fontsize=12, color='#FAFAFA')
        ax2.set_ylabel('Accuracy', fontsize=12, color='#FAFAFA')
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold', color='#00D9FF')
        ax2.legend(fontsize=10, facecolor='#1E2127', edgecolor='#3E4147')
        ax2.grid(alpha=0.3, color='#3E4147')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
            print(f"Training history saved to {save_path}")
        
        return fig
    
    def generate_summary_report(self):
        """
        Generate a summary report of all metrics.
        
        Returns:
            str: Formatted summary report
        """
        report = "\n" + "="*60 + "\n"
        report += "COMPREHENSIVE EVALUATION SUMMARY\n"
        report += "="*60 + "\n\n"
        
        if 'accuracy' in self.metrics:
            report += "Classification Performance:\n"
            report += f"  Accuracy:           {self.metrics['accuracy']:.4f}\n"
            report += f"  F1-Score (Macro):   {self.metrics['f1_macro']:.4f}\n"
            report += f"  F1-Score (Weighted): {self.metrics['f1_weighted']:.4f}\n"
            if self.metrics.get('roc_auc'):
                report += f"  ROC-AUC:            {self.metrics['roc_auc']:.4f}\n"
        
        if 'gene_reduction_pct' in self.metrics:
            report += f"\nFeature Selection:\n"
            report += f"  Original genes:     {self.metrics['original_genes']}\n"
            report += f"  Selected genes:     {self.metrics['selected_genes']}\n"
            report += f"  Reduction:          {self.metrics['gene_reduction_pct']:.2f}%\n"
        
        if 'total_time' in self.metrics:
            report += f"\nComputational Efficiency:\n"
            report += f"  Optimization time:  {self.metrics['optimization_time']:.2f}s\n"
            report += f"  Training time:      {self.metrics['training_time']:.2f}s\n"
            report += f"  Inference time:     {self.metrics['inference_time']:.2f}s\n"
            report += f"  Total time:         {self.metrics['total_time']:.2f}s\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report
    
    def get_metrics(self):
        """Get all computed metrics as dictionary."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate predictions
    n_samples = 100
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, 20, replace=False)
    y_pred[error_indices] = (y_pred[error_indices] + 1) % n_classes
    
    # Simulate probabilities
    y_pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Evaluate
    evaluator = ModelEvaluator(n_classes=n_classes)
    evaluator.evaluate(y_true, y_pred, y_pred_proba)
    evaluator.compute_gene_reduction(original_genes=20000, selected_genes=150)
    evaluator.add_timing(optimization_time=120.5, training_time=45.3, inference_time=0.8)
    
    print(evaluator.generate_summary_report())
