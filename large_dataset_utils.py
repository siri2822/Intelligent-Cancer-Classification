"""
Utilities for handling large gene expression datasets efficiently.

This module provides tools for:
- Chunked data processing
- Feature sampling and filtering
- Memory-efficient batch processing
- Data validation for large files
"""

import numpy as np
import pandas as pd
import gc
from typing import Tuple, Optional


class LargeDatasetHandler:
    """Handle large gene expression datasets with memory efficiency."""
    
    def __init__(self, max_memory_mb=2000):
        """
        Initialize the handler.
        
        Args:
            max_memory_mb (int): Maximum memory to use in MB
        """
        self.max_memory_mb = max_memory_mb
    
    @staticmethod
    def estimate_memory(n_samples: int, n_features: int) -> float:
        """
        Estimate memory usage in MB.
        
        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            
        Returns:
            float: Estimated memory in MB
        """
        # 8 bytes per float64 + overhead
        return (n_samples * n_features * 8) / (1024 * 1024)
    
    def filter_genes_by_variance(self, 
                                 X: np.ndarray, 
                                 percentile: float = 80) -> np.ndarray:
        """
        Filter genes by variance to reduce dimensions.
        
        Keeps only genes with variance in top percentile.
        
        Args:
            X (np.ndarray): Gene expression matrix (samples × genes)
            percentile (float): Keep genes above this percentile (0-100)
            
        Returns:
            np.ndarray: Filtered gene indices
        """
        variances = np.var(X, axis=0)
        threshold = np.percentile(variances, percentile)
        selected_indices = np.where(variances >= threshold)[0]
        print(f"Variance filtering: {len(selected_indices)} genes kept (top {100-percentile:.0f}%)")
        return selected_indices
    
    def random_gene_sampling(self,
                            n_features: int,
                            target_features: int = 10000) -> np.ndarray:
        """
        Randomly sample genes to reduce dimensions.
        
        Args:
            n_features (int): Total number of features
            target_features (int): Target number of features to keep
            
        Returns:
            np.ndarray: Sampled gene indices
        """
        if n_features <= target_features:
            return np.arange(n_features)
        
        sampled = np.random.choice(n_features, target_features, replace=False)
        sampled = np.sort(sampled)
        print(f"Random sampling: {len(sampled)} genes selected from {n_features}")
        return sampled
    
    def load_and_filter_csv(self,
                           filepath: str,
                           max_genes: int = 15000,
                           filter_variance: bool = True,
                           variance_percentile: float = 70,
                           output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CSV and filter genes before full processing.
        
        Args:
            filepath (str): Path to CSV file
            max_genes (int): Maximum genes to keep
            filter_variance (bool): Whether to filter by variance
            variance_percentile (float): Percentile for variance filtering
            output_path (str, optional): Save filtered data
            
        Returns:
            pd.DataFrame: Filtered dataset
        """
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Original shape: {df.shape}")
        
        # Separate features and labels
        if 'label' in df.columns:
            X = df.drop(columns=['label'])
            y = df['label']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Filter genes
        if filter_variance:
            selected_genes = self.filter_genes_by_variance(
                X.values, 
                percentile=variance_percentile
            )
            X = X.iloc[:, selected_genes]
        
        # Further sample if still too many
        if X.shape[1] > max_genes:
            selected_genes = self.random_gene_sampling(X.shape[1], max_genes)
            X = X.iloc[:, selected_genes]
        
        # Reconstruct dataframe
        filtered_df = X.copy()
        filtered_df['label'] = y.values
        
        print(f"Filtered shape: {filtered_df.shape}")
        
        if output_path:
            filtered_df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
        
        return filtered_df
    
    @staticmethod
    def process_in_chunks(X: np.ndarray,
                         chunk_size: int = 10000) -> list:
        """
        Process large array in chunks.
        
        Args:
            X (np.ndarray): Input array
            chunk_size (int): Size of each chunk
            
        Returns:
            list: List of array chunks
        """
        chunks = []
        for i in range(0, len(X), chunk_size):
            chunks.append(X[i:i+chunk_size])
        return chunks
    
    @staticmethod
    def cleanup_memory():
        """Explicitly cleanup memory."""
        gc.collect()


def reduce_dataset_size(csv_path: str,
                       output_path: str,
                       max_samples: Optional[int] = None,
                       max_genes: int = 10000,
                       use_variance_filter: bool = True):
    """
    Standalone function to pre-process large CSV files.
    
    Usage:
        python -c "from large_dataset_utils import reduce_dataset_size; \\
                   reduce_dataset_size('large_data.csv', 'filtered_data.csv')"
    
    Args:
        csv_path (str): Input CSV file path
        output_path (str): Output CSV file path
        max_samples (int, optional): Maximum samples to keep
        max_genes (int): Maximum genes to keep after filtering
        use_variance_filter (bool): Use variance filtering
    """
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Original size: {df.shape}")
    
    # Sample rows if needed
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled to {len(df)} rows")
    
    # Filter genes
    if use_variance_filter and df.shape[1] > 2:
        handler = LargeDatasetHandler()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1]
        
        selected_genes = handler.filter_genes_by_variance(X, percentile=70)
        df = df.iloc[:, selected_genes].copy()
        df['label'] = y.values
    
    # Sample genes if still too many
    if df.shape[1] > max_genes:
        handler = LargeDatasetHandler()
        X_cols = df.columns[:-1].tolist()
        label_col = df.columns[-1]
        
        selected_indices = handler.random_gene_sampling(len(X_cols), max_genes)
        selected_cols = [X_cols[i] for i in selected_indices]
        df = df[selected_cols + [label_col]].copy()
    
    # Save
    df.to_csv(output_path, index=False)
    new_mem = LargeDatasetHandler.estimate_memory(len(df), df.shape[1] - 1)
    print(f"Final size: {df.shape}")
    print(f"Estimated memory: {new_mem:.1f} MB")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        reduce_dataset_size(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python large_dataset_utils.py <input.csv> <output.csv>")
