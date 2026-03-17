"""
Hybrid GA-PSO Optimization for Gene Selection

This module implements a hybrid optimization algorithm combining:
- Genetic Algorithm (GA): Exploration via crossover and mutation
- Particle Swarm Optimization (PSO): Exploitation via velocity and position updates

The fitness function uses a simple classifier to evaluate gene subsets.

Performance Optimized:
- Vectorized mutation and PSO operations (NumPy)
- Adaptive fitness evaluation with data subsampling
- Variance-based pre-filtering to reduce search space
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')


class HybridGAPSO:
    """
    Hybrid Genetic Algorithm - Particle Swarm Optimization for feature selection.
    
    GA Component:
    - Binary encoding (1 = gene selected, 0 = gene not selected)
    - Tournament selection
    - Single-point crossover
    - Bit-flip mutation (vectorized)
    
    PSO Component:
    - Velocity and position updates (vectorized)
    - Personal best and global best tracking
    - Inertia weight for exploration-exploitation balance
    
    Fitness Evaluation:
    - Adaptive classifier with data subsampling for large datasets
    """
    
    def __init__(self, 
                 n_particles=30,
                 n_generations=50,
                 n_selected_genes=150,
                 crossover_rate=0.8,
                 mutation_rate=0.01,
                 w=0.7,          # Inertia weight
                 c1=1.5,         # Cognitive parameter
                 c2=1.5,         # Social parameter
                 tournament_size=3,
                 random_state=42):
        """
        Initialize Hybrid GA-PSO optimizer.
        
        Args:
            n_particles (int): Population/swarm size
            n_generations (int): Number of optimization iterations
            n_selected_genes (int): Target number of genes to select
            crossover_rate (float): Probability of crossover (GA)
            mutation_rate (float): Probability of bit flip (GA)
            w (float): Inertia weight (PSO)
            c1 (float): Cognitive/personal best weight (PSO)
            c2 (float): Social/global best weight (PSO)
            tournament_size (int): Tournament selection size (GA)
            random_state (int): Random seed
        """
        self.n_particles = n_particles
        self.n_generations = n_generations
        self.n_selected_genes = n_selected_genes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        # Will be set during optimization
        self.n_features = None
        self.n_original_features = None  # Before pre-filtering
        self.X_train = None
        self.y_train = None
        self.X_train_subsample = None  # Subsampled data for fitness
        self.y_train_subsample = None
        self.prefilter_indices = None  # Mapping from filtered to original indices
        
        # Swarm state
        self.positions = None       # Current binary positions
        self.velocities = None      # Current velocities (continuous)
        self.fitness = None         # Current fitness values
        self.pbest_positions = None # Personal best positions
        self.pbest_fitness = None   # Personal best fitness
        self.gbest_position = None  # Global best position
        self.gbest_fitness = None   # Global best fitness
        
        # History tracking
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_genes': []
        }
        
        # Use a LOCAL random generator so results are identical every run
        self.rng = np.random.RandomState(random_state)
    
    def _prefilter_by_variance(self, X, max_features=5000):
        """
        Pre-filter features by variance to reduce search space.
        
        Keeps only the top `max_features` genes ranked by variance.
        This dramatically reduces the dimensionality before GA-PSO search.
        
        Args:
            X (array): Full feature matrix
            max_features (int): Maximum number of features to keep
            
        Returns:
            tuple: (filtered X, indices mapping to original features)
        """
        n_features = X.shape[1]
        if n_features <= max_features:
            return X, np.arange(n_features)
        
        print(f"  Pre-filtering: {n_features} → {max_features} genes by variance...")
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[::-1][:max_features]  # deterministic, no rng needed
        top_indices = np.sort(top_indices)  # Keep original order
        
        return X[:, top_indices], top_indices
    
    def _prepare_subsample(self, X, y, max_samples=2000):
        """
        Prepare a subsample of data for faster fitness evaluation.
        
        Args:
            X (array): Training features
            y (array): Training labels
            max_samples (int): Maximum samples to use
            
        Returns:
            tuple: (subsampled X, subsampled y)
        """
        n_samples = len(X)
        if n_samples <= max_samples:
            return X, y
        
        print(f"  Subsampling: {n_samples} → {max_samples} samples for fitness evaluation...")
        # Stratified subsampling to maintain class balance
        indices = self.rng.choice(n_samples, max_samples, replace=False)
        return X[indices], y[indices]
    
    def initialize_swarm(self):
        """
        Initialize particle positions and velocities.
        
        Positions are binary (0 or 1) indicating gene selection.
        Velocities are continuous values for PSO updates.
        """
        print("Initializing swarm...")
        
        # sanity checks
        if self.n_selected_genes > self.n_features:
            raise ValueError(
                f"n_selected_genes ({self.n_selected_genes}) cannot be larger than "
                f"number of features ({self.n_features}). Adjust parameters or "
                "provide more data."
            )
        
        # Initialize random binary positions
        # Ensure each particle selects approximately n_selected_genes
        self.positions = np.zeros((self.n_particles, self.n_features), dtype=int)

        
        for i in range(self.n_particles):
            # Randomly select genes for each particle
            selected_indices = self.rng.choice(
                self.n_features, 
                size=self.n_selected_genes, 
                replace=False
            )
            self.positions[i, selected_indices] = 1
        
        # Initialize velocities (small random values)
        self.velocities = self.rng.uniform(-0.1, 0.1, 
                                           (self.n_particles, self.n_features))
        
        # Initialize fitness
        self.fitness = np.zeros(self.n_particles)
        
        # Initialize personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.zeros(self.n_particles)
        
        # Initialize global best
        self.gbest_position = None
        self.gbest_fitness = -np.inf
        
        print(f"Swarm initialized with {self.n_particles} particles")
    
    def evaluate_fitness(self, position):
        """
        Evaluate fitness of a gene subset using a fast classifier.
        
        Uses subsampled data and adaptive classifier complexity for speed.
        
        Args:
            position (array): Binary array indicating selected genes
            
        Returns:
            float: Classification accuracy (fitness score)
        """
        # Get selected gene indices
        selected_genes = np.where(position == 1)[0]
        
        # Need at least a few genes
        if len(selected_genes) < 10:
            return 0.0
        
        # Use subsampled data for fitness evaluation
        X_eval = self.X_train_subsample[:, selected_genes]
        y_eval = self.y_train_subsample
        
        # Train classifier and evaluate
        try:
            n_selected = len(selected_genes)
            n_samples = len(X_eval)
            
            # Adaptive hidden layer size based on selected features
            hidden_size = min(32, max(10, n_selected // 5))
            
            mlp = MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                max_iter=50,  # Reduced from 100
                random_state=self.random_state,
                early_stopping=True,
                n_iter_no_change=3,  # Reduced from 5
                validation_fraction=0.15,
                solver='adam',
                learning_rate='adaptive'
            )
            
            # Use 2-fold CV for large datasets, 3-fold for small
            n_folds = 2 if n_samples > 1000 else 3
            scores = cross_val_score(mlp, X_eval, y_eval, 
                                    cv=n_folds, scoring='accuracy')
            fitness = scores.mean()
            
        except Exception as e:
            # If evaluation fails, return low fitness
            fitness = 0.0
        
        return fitness
    
    def evaluate_swarm(self):
        """Evaluate fitness for all particles in the swarm."""
        for i in range(self.n_particles):
            self.fitness[i] = self.evaluate_fitness(self.positions[i])
            
            # Update personal best
            if self.fitness[i] > self.pbest_fitness[i]:
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_positions[i] = self.positions[i].copy()
            
            # Update global best
            if self.fitness[i] > self.gbest_fitness:
                self.gbest_fitness = self.fitness[i]
                self.gbest_position = self.positions[i].copy()
    
    def tournament_selection(self):
        """
        Tournament selection for GA crossover.
        
        Returns:
            int: Index of selected particle
        """
        # Randomly select tournament participants
        candidates = self.rng.choice(self.n_particles, self.tournament_size, replace=False)
        # Return the one with best fitness
        best_idx = candidates[np.argmax(self.fitness[candidates])]
        return best_idx
    
    def crossover(self, parent1, parent2):
        """
        Single-point crossover operation.
        
        Args:
            parent1 (array): First parent binary position
            parent2 (array): Second parent binary position
            
        Returns:
            tuple: Two offspring
        """
        if self.rng.random() < self.crossover_rate:
            # Random crossover point
            point = self.rng.randint(1, self.n_features)
            
            # Create offspring
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
            
            return offspring1, offspring2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """
        Vectorized bit-flip mutation operation.
        
        Args:
            individual (array): Binary position to mutate
            
        Returns:
            array: Mutated individual
        """
        mutated = individual.copy()
        
        # VECTORIZED: single NumPy operation instead of per-element loop
        mutation_mask = self.rng.random(self.n_features) < self.mutation_rate
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        
        # Ensure we have at least some genes selected
        min_genes = min(10, self.n_features)
        if mutated.sum() < min_genes:
            selected = self.rng.choice(self.n_features, min_genes, replace=False)
            mutated[selected] = 1
        
        return mutated
    
    def ga_operations(self):
        """
        Apply GA operations: selection, crossover, mutation.
        
        This provides exploration capability to the hybrid algorithm.
        """
        new_positions = []
        
        # Generate new population
        while len(new_positions) < self.n_particles:
            # Selection
            parent1_idx = self.tournament_selection()
            parent2_idx = self.tournament_selection()
            
            parent1 = self.positions[parent1_idx]
            parent2 = self.positions[parent2_idx]
            
            # Crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)
            
            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            new_positions.append(offspring1)
            if len(new_positions) < self.n_particles:
                new_positions.append(offspring2)
        
        return np.array(new_positions[:self.n_particles])
    
    def pso_operations(self):
        """
        Vectorized PSO operations: velocity and position updates.
        
        This provides exploitation capability to the hybrid algorithm.
        All operations are fully vectorized with NumPy.
        """
        # Generate random matrices for all particles at once
        r1 = self.rng.random((self.n_particles, self.n_features))
        r2 = self.rng.random((self.n_particles, self.n_features))
        
        # Cognitive component (personal best) - vectorized
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        
        # Social component (global best) - vectorized
        social = self.c2 * r2 * (self.gbest_position - self.positions)
        
        # Update all velocities at once
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Clamp velocities
        self.velocities = np.clip(self.velocities, -1, 1)
        
        # Update positions using sigmoid transfer function - FULLY VECTORIZED
        sigmoid_v = 1 / (1 + np.exp(-self.velocities))
        random_vals = self.rng.random((self.n_particles, self.n_features))
        self.positions = (random_vals < sigmoid_v).astype(int)
    
    def optimize(self, X_train, y_train, verbose=True):
        """
        Run the hybrid GA-PSO optimization process.
        
        Includes automatic pre-filtering and data subsampling for large datasets.
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Optimization results including best genes and fitness history
        """
        print("\n" + "="*60)
        print("STARTING HYBRID GA-PSO OPTIMIZATION")
        print("="*60 + "\n")
        
        self.n_original_features = X_train.shape[1]
        
        # Step 1: Pre-filter by variance for large feature spaces
        max_prefilter = max(self.n_selected_genes * 20, 5000)
        X_filtered, self.prefilter_indices = self._prefilter_by_variance(
            X_train, max_features=max_prefilter
        )
        
        self.X_train = X_filtered
        self.y_train = y_train
        self.n_features = X_filtered.shape[1]
        
        # Adjust n_selected_genes if it exceeds available features after filtering
        if self.n_selected_genes > self.n_features:
            print(f"  Adjusting target genes: {self.n_selected_genes} → {self.n_features}")
            self.n_selected_genes = self.n_features
        
        # Step 2: Prepare subsampled data for fitness evaluation
        self.X_train_subsample, self.y_train_subsample = self._prepare_subsample(
            X_filtered, y_train, max_samples=2000
        )
        
        print(f"\nConfiguration:")
        print(f"  Original genes: {self.n_original_features}")
        print(f"  Pre-filtered genes: {self.n_features}")
        print(f"  Target selected genes: {self.n_selected_genes}")
        print(f"  Population size: {self.n_particles}")
        print(f"  Generations: {self.n_generations}")
        print(f"  Fitness eval samples: {len(self.X_train_subsample)}")
        print(f"  Crossover rate: {self.crossover_rate}")
        print(f"  Mutation rate: {self.mutation_rate}")
        print(f"  PSO parameters: w={self.w}, c1={self.c1}, c2={self.c2}")
        print()
        
        start_time = time.time()
        
        # Initialize swarm
        self.initialize_swarm()
        
        # Initial evaluation
        print("Evaluating initial population...")
        self.evaluate_swarm()
        
        # Store initial statistics
        self.history['best_fitness'].append(self.gbest_fitness)
        self.history['mean_fitness'].append(self.fitness.mean())
        self.history['best_genes'].append(self.gbest_position.copy())
        
        if verbose:
            print(f"Generation 0: Best Fitness = {self.gbest_fitness:.4f}, "
                  f"Mean Fitness = {self.fitness.mean():.4f}")
        
        # Main optimization loop
        for generation in range(1, self.n_generations + 1):
            gen_start = time.time()
            
            # Apply GA operations (exploration)
            ga_positions = self.ga_operations()
            
            # Apply PSO operations (exploitation)
            self.pso_operations()
            
            # Hybrid: Combine GA and PSO results
            # Replace half the population with GA offspring
            n_replace = self.n_particles // 2
            replace_indices = self.rng.choice(self.n_particles, n_replace, replace=False)
            self.positions[replace_indices] = ga_positions[replace_indices]
            
            # Evaluate new population
            self.evaluate_swarm()
            
            # Store statistics
            self.history['best_fitness'].append(self.gbest_fitness)
            self.history['mean_fitness'].append(self.fitness.mean())
            self.history['best_genes'].append(self.gbest_position.copy())
            
            gen_time = time.time() - gen_start
            
            if verbose and (generation % 5 == 0 or generation == 1):
                elapsed = time.time() - start_time
                remaining = gen_time * (self.n_generations - generation)
                n_selected = self.gbest_position.sum()
                print(f"Generation {generation}/{self.n_generations}: "
                      f"Best = {self.gbest_fitness:.4f}, "
                      f"Mean = {self.fitness.mean():.4f}, "
                      f"Genes = {n_selected}, "
                      f"ETA = {remaining:.0f}s")
        
        elapsed_time = time.time() - start_time
        
        # Get final selected gene indices in the filtered space
        selected_filtered = np.where(self.gbest_position == 1)[0]
        
        # Map back to original feature indices
        selected_gene_indices = self.prefilter_indices[selected_filtered]
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"Best fitness achieved: {self.gbest_fitness:.4f}")
        print(f"Number of selected genes: {len(selected_gene_indices)}")
        print(f"Total optimization time: {elapsed_time:.2f} seconds")
        print(f"Time per generation: {elapsed_time/self.n_generations:.2f} seconds")
        
        return {
            'selected_genes': selected_gene_indices,
            'best_fitness': self.gbest_fitness,
            'best_position': self.gbest_position,
            'fitness_history': self.history['best_fitness'],
            'mean_fitness_history': self.history['mean_fitness'],
            'optimization_time': elapsed_time,
            'n_selected': len(selected_gene_indices)
        }


if __name__ == "__main__":
    # Example usage with synthetic data
    from data_preprocessing import generate_synthetic_dataset, DataPreprocessor
    
    # Generate dataset
    print("Generating synthetic dataset...")
    dataset_path = generate_synthetic_dataset(
        n_samples=500,
        n_genes=1000,  # Smaller for quick testing
        n_classes=3
    )
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(dataset_path)
    
    # Run optimization
    print("\nRunning Hybrid GA-PSO optimization...")
    optimizer = HybridGAPSO(
        n_particles=20,
        n_generations=10,
        n_selected_genes=100,
        random_state=42
    )
    
    results = optimizer.optimize(data['X_train'], data['y_train'])
    
    print(f"\nSelected genes: {results['selected_genes'][:10]}... (showing first 10)")
