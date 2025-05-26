"""Data splitting strategies for federated learning experiments."""

from typing import List, Any, Optional
import numpy as np
from torch.utils.data import Dataset, Subset


class DataSplitter:
    """Handles different strategies for splitting data across federated nodes."""
    
    def __init__(self, strategy: str = "random", skew_factor: float = 0.8):
        """Initialize the data splitter.
        
        Args:
            strategy: The splitting strategy to use ("random" or "skewed")
            skew_factor: The skew factor for skewed splitting (0.5 = random, 1.0 = completely skewed)
        """
        self.strategy = strategy
        self.skew_factor = skew_factor
    
    def split(self, dataset: Dataset, num_partitions: int, num_classes: int = 10) -> List[Subset]:
        """Split the dataset into partitions based on the configured strategy.
        
        Args:
            dataset: The dataset to split
            num_partitions: Number of partitions to create
            num_classes: Number of classes in the dataset (for skewed splitting)
            
        Returns:
            List of dataset subsets
        """
        if self.strategy == "skewed":
            return self._create_skewed_split(dataset, num_partitions, num_classes)
        else:
            return self._create_random_split(dataset, num_partitions)
    
    def _create_random_split(self, dataset: Dataset, num_partitions: int) -> List[Subset]:
        """Create random data partitions.
        
        Args:
            dataset: The dataset to split
            num_partitions: Number of partitions to create
            
        Returns:
            List of randomly partitioned dataset subsets
        """
        print("Creating random data split")
        
        total_size = len(dataset)
        partition_size = total_size // num_partitions
        
        # Create random indices
        indices = np.random.permutation(total_size)
        
        partitions = []
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_partitions - 1 else total_size
            partition_indices = indices[start_idx:end_idx]
            partitions.append(Subset(dataset, partition_indices))
        
        print(f"Created {num_partitions} random partitions with ~{partition_size} samples each")
        return partitions
    
    def _create_skewed_split(self, dataset: Dataset, num_partitions: int, num_classes: int) -> List[Subset]:
        """Create skewed data partitions based on class distribution.
        
        Args:
            dataset: The dataset to split
            num_partitions: Number of partitions to create
            num_classes: Number of classes in the dataset
            
        Returns:
            List of skewed partitioned dataset subsets
        """
        print(f"Creating skewed data split with skew factor: {self.skew_factor}")
        
        # Get labels from dataset
        labels = self._extract_labels(dataset)
        
        # Group indices by class
        class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Partition classes among nodes
        classes_per_node = np.array_split(np.arange(num_classes), num_partitions)
        
        print("Class distribution per node:")
        for i, classes in enumerate(classes_per_node):
            print(f"Node {i}: Primary classes {list(classes)}")
        
        # Create skewed partitions
        partitioned_indices = [[] for _ in range(num_partitions)]
        
        for class_idx in range(num_classes):
            # Find which partition this class primarily belongs to
            primary_partition = self._find_primary_partition(class_idx, classes_per_node)
            
            # Distribute samples of this class
            class_samples = class_indices[class_idx]
            np.random.shuffle(class_samples)
            
            num_samples = len(class_samples)
            primary_samples = int(num_samples * self.skew_factor)
            
            # Assign majority to primary partition
            partitioned_indices[primary_partition].extend(class_samples[:primary_samples])
            
            # Distribute remaining samples randomly among other partitions
            remaining_samples = class_samples[primary_samples:]
            for sample_idx in remaining_samples:
                random_partition = np.random.randint(0, num_partitions)
                partitioned_indices[random_partition].append(sample_idx)
        
        # Create Subset objects
        partitions = []
        for partition_indices in partitioned_indices:
            np.random.shuffle(partition_indices)  # Shuffle within partition
            partitions.append(Subset(dataset, partition_indices))
        
        # Print partition statistics
        self._print_partition_statistics(partitions, labels)
        
        return partitions
    
    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        """Extract labels from dataset.
        
        Args:
            dataset: The dataset to extract labels from
            
        Returns:
            Array of labels
        """
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            return np.array(dataset.labels)
        else:
            # Fallback: extract labels by iterating through dataset
            labels = []
            for _, label in dataset:
                labels.append(label)
            return np.array(labels)
    
    def _find_primary_partition(self, class_idx: int, classes_per_node: List[np.ndarray]) -> int:
        """Find the primary partition for a class.
        
        Args:
            class_idx: The class index to find partition for
            classes_per_node: List of class arrays per node
            
        Returns:
            Index of the primary partition
        """
        for partition_idx, classes in enumerate(classes_per_node):
            if class_idx in classes:
                return partition_idx
        return class_idx % len(classes_per_node)
    
    def _print_partition_statistics(self, partitions: List[Subset], labels: np.ndarray):
        """Print statistics about the created partitions.
        
        Args:
            partitions: List of dataset subsets
            labels: Array of all dataset labels
        """
        print("\nPartition statistics:")
        for i, partition in enumerate(partitions):
            partition_labels = [labels[idx] for idx in partition.indices]
            unique, counts = np.unique(partition_labels, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"Node {i}: {len(partition)} samples, class distribution: {class_dist}") 