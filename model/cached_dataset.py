"""
Optimized dataset implementations for faster data loading and memory efficiency.

This module provides several dataset wrapper classes that improve training performance:
- CachedDataset: Simple in-memory caching of dataset items
- DynamicCachedDataset: LRU-based caching with memory management
- SampledDataset: Random sampling of a subset of the dataset
"""

from torch.utils.data import Dataset
import numpy as np
import time

class CachedDataset(Dataset):
    """A simple wrapper dataset that caches dataset items in memory.

    This basic caching implementation stores dataset items in a dictionary for fast retrieval,
    which can significantly speed up training when dataset loading is a bottleneck.
    Unlike DynamicCachedDataset, this implementation has no eviction policy.

    Args:
        dataset: The dataset to cache
        cache_size: Maximum number of samples to cache (default: all)
        preload: Whether to preload items at initialization (default: False)
    """
    def __init__(self, dataset, cache_size=None, preload=False):
        self.dataset = dataset
        self.cache = {}
        self.cache_size = cache_size if cache_size is not None else len(dataset)
        self.cache_size = min(self.cache_size, len(dataset))

        # Initialize tracking variables
        self._dataset_size = len(dataset)

        # Silently preload without printing to console
        if preload:
            for i in range(self.cache_size):
                self.cache[i] = self.dataset[i]

    def __getitem__(self, idx):
        if idx < self.cache_size:
            if idx not in self.cache:
                self.cache[idx] = self.dataset[idx]
            return self.cache[idx]
        else:
            return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class DynamicCachedDataset(Dataset):
    """A wrapper dataset that dynamically caches dataset items with LRU eviction policy.

    This class provides memory-efficient caching with automatic garbage collection and
    device placement. It uses an OrderedDict for O(1) operations on both insertion and
    deletion, implementing a Least Recently Used (LRU) eviction policy.

    Args:
        dataset: The dataset to cache
        cache_size: Maximum number of samples to keep in cache (default: 1000)
        preload: Whether to preload samples at initialization (default: False)
        preload_size: Number of samples to preload (default: min(100, cache_size))
        device: Device to place tensors on (default: 'cpu')
    """
    def __init__(self, dataset, cache_size=1000, preload=False, preload_size=None, device='cpu'):
        import gc
        self.dataset = dataset
        self.cache = {}
        # Configure cache size with safety limits
        if cache_size <= 0:
            self.cache_size = 0
            print("Caching completely disabled (cache_size <= 0)")
        else:
            # Apply hard limit of 1000 items to prevent excessive memory usage
            # This protects against misconfiguration in the config file
            self.cache_size = min(cache_size, len(dataset), 1000)

        self.device = device

        # Use OrderedDict for O(1) operations on both insertion and deletion
        from collections import OrderedDict
        self.access_order = OrderedDict()

        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.load_times = []

        # Memory monitoring - more aggressive garbage collection
        self.last_gc_time = time.time()
        self.gc_interval = 30  # Run garbage collection every 30 seconds

        # Set preload size - limit to prevent memory issues
        if preload_size is None:
            preload_size = min(100, self.cache_size)
        else:
            preload_size = min(preload_size, self.cache_size, 500)  # Hard limit of 500 preloaded items

        # Initialize tracking variables
        self._last_hit_rate = 0
        self._last_fetch_time_ms = 0
        self._last_cache_size_mb = 0
        self._last_cleanup_size = 0
        self._last_eviction_error = None
        self._emergency_clear = False

        # Preload initial cache items if requested (without console output)
        if preload and preload_size > 0:
            start_time = time.time()

            # Clean memory before preloading to maximize available space
            gc.collect()

            for i in range(preload_size):
                try:
                    # Time each sample load for performance tracking
                    sample_start = time.time()
                    self.cache[i] = self.dataset[i]
                    self.access_order[i] = None  # Only key order matters in OrderedDict
                    sample_time = time.time() - sample_start
                    self.load_times.append(sample_time)

                    # Regular memory cleanup during preloading to prevent OOM
                    if i > 0 and i % 100 == 0:
                        gc.collect()
                except Exception:
                    # Skip problematic samples silently to avoid initialization errors
                    continue

            # Final garbage collection after preloading
            gc.collect()

            # Calculate stats but don't print to avoid console clutter
            self._preload_time = time.time() - start_time
            self._avg_load_time = sum(self.load_times) / len(self.load_times) if self.load_times else 0

    def __getitem__(self, idx):
        import gc
        import sys
        import torch
        start_time = time.time()

        # Fast path: if caching is disabled, load item and move to device if needed
        if self.cache_size <= 0:
            item = self.dataset[idx]
            # Move tensors to GPU immediately if device is CUDA
            if self.device != 'cpu' and isinstance(self.device, torch.device) and self.device.type == 'cuda':
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            item[k] = v.to(self.device, non_blocking=True)
                elif isinstance(item, torch.Tensor):
                    item = item.to(self.device, non_blocking=True)
                elif isinstance(item, tuple) and all(isinstance(x, torch.Tensor) for x in item):
                    item = tuple(x.to(self.device, non_blocking=True) for x in item)
            return item

        # Periodic garbage collection based on time
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            gc.collect()
            self.last_gc_time = current_time

            # Track memory usage internally but don't print to avoid console clutter
            if hasattr(sys, 'getsizeof'):
                cache_size_bytes = sum(sys.getsizeof(v) for v in self.cache.values())
                self._last_cache_size_mb = cache_size_bytes / (1024*1024)

        # Check if item is in cache
        if idx in self.cache:
            # Update access order (move to end) - O(1) operation with OrderedDict
            self.access_order.pop(idx, None)
            self.access_order[idx] = None
            self.cache_hits += 1
            return self.cache[idx]

        # Item not in cache, fetch it
        load_start = time.time()
        try:
            item = self.dataset[idx]
        except Exception as e:
            # Track error but don't print to avoid console clutter
            self._last_load_error = str(e)
            # Return a fallback item if possible
            if len(self.cache) > 0:
                fallback_idx = next(iter(self.cache.keys()))
                return self.cache[fallback_idx]
            else:
                raise  # Re-raise if we can't recover

        load_time = time.time() - load_start
        self.load_times.append(load_time)

        # Keep a moving average of the last 100 load times
        if len(self.load_times) > 100:
            self.load_times.pop(0)

        self.cache_misses += 1

        # Emergency cleanup if cache exceeds size limit by 10%
        if len(self.cache) >= self.cache_size * 1.1:
            # Track cleanup metrics for monitoring
            self._last_cleanup_size = len(self.cache)
            # Batch removal: evict 20% of least recently used items at once
            # This is more efficient than removing one at a time
            items_to_remove = int(self.cache_size * 0.2)
            for _ in range(items_to_remove):
                try:
                    if len(self.access_order) > 0:
                        lru_idx, _ = next(iter(self.access_order.items()))
                        if lru_idx in self.cache:
                            del self.cache[lru_idx]
                        self.access_order.pop(lru_idx, None)
                except (StopIteration, RuntimeError):
                    break
            # Force garbage collection
            gc.collect()

        # Standard cache insertion logic
        if len(self.cache) < self.cache_size:
            # If cache has space, simply add the new item
            self.cache[idx] = item
            self.access_order[idx] = None
        else:
            # LRU eviction: remove least recently used item (first item in OrderedDict)
            # This is an O(1) operation with OrderedDict
            try:
                lru_idx, _ = next(iter(self.access_order.items()))
                # Explicitly delete to ensure memory is freed immediately
                if lru_idx in self.cache:
                    del self.cache[lru_idx]
                self.access_order.pop(lru_idx, None)

                # Periodic garbage collection to prevent memory fragmentation
                if self.cache_misses % 100 == 0:
                    gc.collect()
            except (StopIteration, RuntimeError) as e:
                # Record error details for diagnostics without console output
                self._last_eviction_error = str(e)
                # Emergency recovery: completely clear cache if it's nearly full
                # This prevents cascading failures when the cache becomes corrupted
                if len(self.cache) > self.cache_size * 0.9:
                    self._emergency_clear = True
                    self.cache.clear()
                    self.access_order.clear()
                    gc.collect()  # Force immediate memory recovery

            # Add new item
            self.cache[idx] = item
            self.access_order[idx] = None

        # Track performance metrics internally but don't print to avoid console clutter
        if self.cache_hits + self.cache_misses > 0 and (self.cache_hits + self.cache_misses) % 1000 == 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            fetch_time = time.time() - start_time
            self._last_hit_rate = hit_rate
            self._last_fetch_time_ms = fetch_time * 1000

        return item

    def __len__(self):
        return len(self.dataset)

    def cache_stats(self):
        """Return comprehensive statistics about the cache performance.

        Returns:
            Dictionary containing cache metrics including size, hit rate, and load times
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses * 100 if total_accesses > 0 else 0
        avg_load_time = sum(self.load_times) / len(self.load_times) if self.load_times else 0

        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_utilization": len(self.cache) / self.cache_size * 100 if self.cache_size > 0 else 0,
            "dataset_size": len(self.dataset),
            "cache_coverage": len(self.cache) / len(self.dataset) * 100,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "avg_load_time_ms": avg_load_time * 1000
        }


class SampledDataset(Dataset):
    """A wrapper dataset that randomly samples a subset of another dataset.

    This class enables training on a random subset of the data, which can significantly
    speed up experimentation and debugging while maintaining dataset diversity.
    The sampling is done once at initialization with a fixed random seed for reproducibility.

    Args:
        dataset: The dataset to sample from
        sample_ratio: Ratio of samples to use (0.0-1.0, default: 0.5)
        seed: Random seed for reproducible sampling (default: 42)
    """
    def __init__(self, dataset, sample_ratio=0.5, seed=42):
        self.dataset = dataset
        self.sample_ratio = sample_ratio

        # Determine subset size based on the sample ratio
        self.num_samples = int(len(dataset) * sample_ratio)

        # Create reproducible random sample using fixed seed
        np.random.seed(seed)
        self.indices = np.random.choice(
            len(dataset),
            size=self.num_samples,
            replace=False  # Sample without replacement for unique indices
        )

        # Store metadata for monitoring without console output
        self._sample_ratio = sample_ratio
        self._num_samples = self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for SampledDataset of size {self.num_samples}")
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.num_samples
