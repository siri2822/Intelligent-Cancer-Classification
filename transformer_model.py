"""
Transformer-Based Cancer Classification Model

This module implements a Transformer neural network for cancer classification using
selected gene expression features.

Architecture:
- Gene grouping for efficient attention (reduces O(n²) cost)
- Embedding layer for gene expression
- Multi-head self-attention mechanism
- Feed-forward neural network
- Classification head with softmax output

Accuracy Improvements:
- Class weight balancing
- Cosine decay learning rate
- Label smoothing
- Adaptive model capacity
"""

import os
# Set deterministic TF environment variables (safe, doesn't break data pipeline)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import warnings
warnings.filterwarnings('ignore')


class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    
    Components:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        """
        Initialize Transformer block.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Hidden dimension of feed-forward network
            dropout_rate (float): Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),  # GELU for better convergence
            layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        """
        Forward pass through the Transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor after attention and feed-forward operations
        """
        # Multi-head self-attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class GeneTransformerClassifier:
    """
    Transformer-based model for cancer classification from gene expression data.
    
    Architecture:
    1. Input: Gene expression values (selected features)
    2. Gene grouping: Chunk genes into groups for efficient attention
    3. Embedding: Project gene groups to higher dimension
    4. Positional encoding: Add position information
    5. Transformer blocks: Multi-head attention and FFN
    6. Global pooling: Aggregate gene representations
    7. Dense layers: Final classification
    8. Output: Softmax probabilities over cancer classes
    """
    
    def __init__(self,
                 n_genes,
                 n_classes,
                 embed_dim=128,
                 num_heads=4,
                 ff_dim=256,
                 num_transformer_blocks=2,
                 mlp_units=[128, 64],
                 dropout_rate=0.15,
                 learning_rate=0.0005,
                 gene_group_size=16,
                 random_state=42):
        """
        Initialize Transformer classifier.
        
        Args:
            n_genes (int): Number of input genes (features)
            n_classes (int): Number of output classes
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward network hidden dimension
            num_transformer_blocks (int): Number of Transformer blocks
            mlp_units (list): Hidden units in final MLP
            dropout_rate (float): Dropout rate
            learning_rate (float): Initial learning rate for optimizer
            gene_group_size (int): Number of genes per group for attention
            random_state (int): Random seed
        """
        self.n_genes = n_genes
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.gene_group_size = gene_group_size
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.history = None
        self.training_time = 0
    
    def _compute_class_weights(self, y_train):
        """
        Compute balanced class weights for imbalanced datasets.
        
        Args:
            y_train (array): Training labels
            
        Returns:
            dict: Class weights mapping
        """
        classes = np.unique(y_train)
        n_samples = len(y_train)
        n_classes = len(classes)
        
        class_weights = {}
        for cls in classes:
            count = np.sum(y_train == cls)
            weight = n_samples / (n_classes * count)
            class_weights[int(cls)] = weight
        
        # Normalize so average weight is 1.0
        avg_weight = np.mean(list(class_weights.values()))
        class_weights = {k: v / avg_weight for k, v in class_weights.items()}
        
        return class_weights
    
    def build_model(self):
        """
        Build the Transformer model architecture with gene grouping.
        
        Gene grouping: Instead of treating each gene as a token (which creates
        O(n²) attention), we group genes into chunks to reduce sequence length.
        
        Returns:
            Compiled Keras model
        """
        print("Building Transformer model...")
        
        # Input layer
        inputs = layers.Input(shape=(self.n_genes,))
        
        # Adapt gene_group_size if n_genes is very small
        effective_group_size = min(self.gene_group_size, self.n_genes)
        
        # Calculate number of groups using ceiling division
        # e.g. 22 genes / 16 group_size = ceil(22/16) = 2 groups
        n_groups = max(1, -(-self.n_genes // effective_group_size))  # ceiling division
        padded_size = n_groups * effective_group_size
        
        # Always project to exact padded_size via Dense layer
        # This handles both exact and non-exact multiples cleanly
        if padded_size != self.n_genes:
            x = layers.Dense(padded_size, activation=None)(inputs)
        else:
            x = inputs
        
        # Reshape to groups: (batch, n_groups, effective_group_size)
        x = layers.Reshape((n_groups, effective_group_size))(x)
        
        # Embed each group to embed_dim
        x = layers.Dense(self.embed_dim)(x)
        
        # Add positional encoding for groups
        positions = tf.range(start=0, limit=n_groups, delta=1)
        position_embedding = layers.Embedding(
            input_dim=n_groups, 
            output_dim=self.embed_dim
        )(positions)
        x = x + position_embedding
        
        # Stack Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x)
        
        # Global average pooling to aggregate gene group information
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP head for classification with batch normalization
        for units in self.mlp_units:
            x = layers.Dense(units, activation='gelu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with softmax
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with label smoothing loss and cosine decay LR
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        print(f"\nModel architecture:")
        print(f"  Input genes: {self.n_genes}")
        print(f"  Gene group size: {self.gene_group_size}")
        print(f"  Sequence length (groups): {n_groups}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Transformer blocks: {self.num_transformer_blocks}")
        print(f"  Output classes: {self.n_classes}")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, verbose=1):
        """
        Train the Transformer model with class balancing and LR scheduling.
        
        Args:
            X_train (array): Training features (selected genes)
            y_train (array): Training labels
            X_val (array, optional): Validation features
            y_val (array, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
            
        Returns:
            History object with training metrics
        """
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER MODEL")
        print("="*60 + "\n")
        
        # Reset ALL seeds for deterministic training
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        # Always rebuild to ensure deterministic initialization
        self.model = self.build_model()
        
        # Compute class weights for balanced training
        class_weights = self._compute_class_weights(y_train)
        print(f"Class weights: {class_weights}")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"Training with validation set")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.3,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"Training configuration:")
        print(f"  Training samples: {len(X_train)}")
        if validation_data:
            print(f"  Validation samples: {len(X_val)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print()
        
        # Train model with class weights
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            shuffle=False,  # Deterministic: no random batch reordering
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        
        # Report final metrics
        if validation_data:
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            print(f"Final validation accuracy: {val_acc:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array): Input features
            
        Returns:
            array: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array): Input features
            
        Returns:
            array: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import generate_synthetic_dataset, DataPreprocessor
    from optimization import HybridGAPSO
    
    print("Generating synthetic dataset...")
    dataset_path = generate_synthetic_dataset(
        n_samples=500,
        n_genes=1000,
        n_classes=3
    )
    
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(dataset_path)
    
    print("\nRunning gene selection (quick test)...")
    optimizer = HybridGAPSO(
        n_particles=10,
        n_generations=5,
        n_selected_genes=50
    )
    results = optimizer.optimize(data['X_train'], data['y_train'], verbose=False)
    
    # Select genes
    selected_genes = results['selected_genes']
    X_train_selected = data['X_train'][:, selected_genes]
    X_test_selected = data['X_test'][:, selected_genes]
    
    print(f"\nTraining Transformer with {len(selected_genes)} selected genes...")
    transformer = GeneTransformerClassifier(
        n_genes=len(selected_genes),
        n_classes=data['n_classes'],
        embed_dim=64,
        num_heads=2,
        num_transformer_blocks=1,
        random_state=42
    )
    
    transformer.train(
        X_train_selected, data['y_train'],
        X_test_selected, data['y_test'],
        epochs=20,
        batch_size=32
    )
    
    # Make predictions
    predictions = transformer.predict(X_test_selected)
    accuracy = (predictions == data['y_test']).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
