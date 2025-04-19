import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from model.utils import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class AUAIRDataset(Dataset):
    """
    Dataset class for AU-AIR aerial drone imagery dataset with memory optimization features

    This class handles loading and preprocessing of the AU-AIR dataset, which contains
    aerial drone imagery with object annotations. It supports image caching for faster
    training iterations and includes robust error handling for corrupted or missing data.

    The dataset loads annotations in various formats and converts bounding boxes to the
    center-based format (cx, cy, w, h) required by Deformable DETR, with appropriate
    normalization to [0,1] range.

    Args:
        root_dir: Root directory containing the dataset images and annotations
        annotations_file: Path to the JSON file with object annotations
        split: Dataset split to use ('train', 'val', or 'test')
        transform: Transforms to apply to images and targets (resize, normalization, etc.)
        cache_images: Whether to cache images in memory for faster access during training
        cache_limit: Maximum number of images to store in cache to prevent memory overflow
    """
    def __init__(self, root_dir, annotations_file, split='train', transform=None, cache_images=True, cache_limit=1000):
        import gc
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Limit cache size to prevent memory issues
        self.cache_images = cache_images
        self.cache_limit = min(cache_limit, 500)  # Hard limit of 500 items to prevent memory issues
        self.img_cache = {}

        # Cache statistics for monitoring
        self.cache_hits = 0
        self.cache_misses = 0

        # Run garbage collection before loading dataset
        gc.collect()

        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)

        # Extract actual annotations from the nested structure
        if isinstance(annotations_data, dict) and 'annotations' in annotations_data:
            # This is the expected structure from split_dataset.py
            self.annotations = annotations_data['annotations']
            self.categories = annotations_data['categories']
            print(f"Loaded {len(self.annotations)} annotations from {annotations_file} (from 'annotations' key)")
        elif isinstance(annotations_data, list):
            # Directly provided annotations list
            self.annotations = annotations_data
            print(f"Loaded {len(self.annotations)} annotations from {annotations_file} (direct list)")
        elif isinstance(annotations_data, dict):
            # Convert dict to list if needed
            self.annotations = [annotations_data[key] for key in annotations_data.keys()]
            print(f"Loaded {len(self.annotations)} annotations from {annotations_file} (converted from dict)")
        else:
            raise ValueError(f"Unexpected annotations format in {annotations_file}")

        # Get class names from config or use default
        if 'class_names' in annotations_data:
            self.class_names = annotations_data['class_names']
        elif hasattr(self, 'config') and 'dataset' in self.config and 'class_names' in self.config['dataset']:
            self.class_names = self.config['dataset']['class_names']
        else:
            self.class_names = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        # Filter annotations based on split
        if split == 'train' or split == 'val' or split == 'test':
            # Use indices from split files if provided
            split_file = os.path.join(root_dir, f'{split}_indices.json')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    indices = json.load(f)
                self.annotations = [self.annotations[i] for i in indices]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            import gc
            annotation = self.annotations[idx]

            # Skip cache and load image directly from disk
            self.cache_misses += 1
            possible_paths = [
                os.path.join(self.root_dir, 'images', annotation['image_name']),
                os.path.join('dataset', 'images', annotation['image_name']),
                os.path.join('..', 'dataset', 'images', annotation['image_name'])
            ]

            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break

            if img_path is None:
                raise FileNotFoundError(f"Image not found for {annotation['image_name']}")

            # Run garbage collection
            gc.collect()

            # Load the image
            image = Image.open(img_path).convert('RGB')

            # Get bounding boxes and labels
            boxes = []
            labels = []

            for bbox in annotation['bbox']:
                # Convert [top, left, height, width] to [x1, y1, x2, y2]
                x1 = bbox['left']
                y1 = bbox['top']
                x2 = x1 + bbox['width']
                y2 = y1 + bbox['height']

                # Get actual image dimensions from the image or annotation
                if hasattr(image, 'width') and hasattr(image, 'height'):
                    # Use actual image dimensions if available
                    image_width = float(image.width)
                    image_height = float(image.height)
                elif 'image_width' in annotation and 'image_height' in annotation:
                    # Use annotation dimensions if available
                    image_width = float(annotation['image_width'])
                    image_height = float(annotation['image_height'])
                else:
                    # Get dimensions from config if available
                    if hasattr(self, 'img_size') and isinstance(self.img_size, (list, tuple)) and len(self.img_size) == 2:
                        image_width = float(self.img_size[1])  # Width is typically the second dimension
                        image_height = float(self.img_size[0])  # Height is typically the first dimension
                    else:
                        # Last resort fallback to standard dimensions
                        image_width = 1280.0  # More standard default
                        image_height = 720.0  # More standard default

                # Ensure box is within image boundaries
                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Store xyxy format
                boxes.append([x1, y1, x2, y2])

                # Get class index (subtract 1 because classes in dataset are 1-indexed)
                class_idx = bbox['class'] - 1
                labels.append(class_idx)

            # Always create tensors on CPU to prevent device mismatch errors
            # device = getattr(self, 'device', None)  # Not used anymore

            # Convert to tensors - create on CPU first to prevent memory leaks
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Convert boxes to center format for DETR
            if len(boxes) > 0:
                boxes = box_xyxy_to_cxcywh(boxes)

                # Normalize box coordinates to [0,1] range
                boxes[:, 0] /= image_width
                boxes[:, 1] /= image_height
                boxes[:, 2] /= image_width
                boxes[:, 3] /= image_height

            # Create target dictionary with consistent dimensions - on CPU first
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'orig_size': torch.as_tensor([image_height, image_width]),
                'size': torch.as_tensor([image_height, image_width])
            }

            # Force garbage collection to prevent memory leaks
            gc.collect()

            # Apply transformations (only basic transforms, no augmentation)
            if self.transform is not None:
                image, target = self.transform(image, target)

            return image, target
        except Exception as e:
            # More detailed error reporting
            import traceback
            import datetime

            # Get current timestamp for error logging
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create detailed error message
            error_msg = f"[{timestamp}] Error loading sample at index {idx}: {e}\n"
            try:
                error_msg += f"Annotation: {annotation['image_name'] if 'image_name' in annotation else 'unknown'}\n"
            except:
                error_msg += f"Annotation: Could not access annotation data\n"
            error_msg += traceback.format_exc()

            # Print a shorter version to console
            print(f"[{timestamp}] Dataset error at index {idx}: {str(e)[:100]}...")

            # Log the detailed error for debugging
            try:
                with open('dataset_errors.log', 'a') as f:
                    f.write(f"\n{'-'*50}\n{error_msg}\n{'-'*50}\n")
            except Exception as log_error:
                print(f"Could not write to error log: {log_error}")

            # Try to return a valid sample from elsewhere in the dataset
            if idx > 0:
                try:
                    # Try to get a valid sample from the beginning
                    return self.__getitem__(0)
                except Exception:
                    # If that fails, try a few more indices
                    for fallback_idx in [1, 2, 5, 10]:
                        if fallback_idx < len(self) and fallback_idx != idx:
                            try:
                                return self.__getitem__(fallback_idx)
                            except Exception:
                                continue

            # Create an empty sample as a last resort fallback with dimensions matching the config
            img_size = getattr(self, 'img_size', 800)
            img = torch.zeros((3, img_size, img_size), dtype=torch.float32)
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'orig_size': torch.as_tensor([img_size, img_size]),
                'size': torch.as_tensor([img_size, img_size])
            }
            return img, target


def collate_fn(batch):
    """
    Custom collate function optimized for object detection batches with GPU memory optimization

    This function handles the creation of batches for object detection data, with several
    optimizations for memory efficiency and performance:

    1. Handles variable-sized images by padding to the largest dimensions in the batch
    2. Uses tensor operations for finding max dimensions and pre-allocating memory
    3. Provides fallback mechanisms for invalid or empty samples
    4. Updates target dictionaries with the new padded dimensions
    5. Efficiently handles edge cases like missing targets
    6. Optimized to create tensors directly on GPU when possible to maximize GPU memory usage

    Args:
        batch: List of (image, target) tuples from the dataset

    Returns:
        Tuple of (padded_images, targets) where padded_images is a tensor of shape
        [batch_size, channels, max_height, max_width] and targets is a list of dictionaries
    """
    images = []
    targets = []

    # Extract all valid images and targets from the batch
    for img_target in batch:
        if img_target is None:
            continue

        if isinstance(img_target, tuple) and len(img_target) == 2:
            image, target = img_target
            images.append(image)
            targets.append(target)
        else:
            continue  # Skip invalid samples silently for speed

    if len(images) == 0:
        # Create a minimal dummy batch for empty batches
        # Check if we can create directly on GPU to maximize GPU memory usage
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        except:
            device = torch.device('cpu')

        dummy_img = torch.zeros((3, 64, 64), dtype=torch.float32, device=device)
        dummy_target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
            'labels': torch.zeros((0,), dtype=torch.int64, device=device),
            'image_id': torch.tensor([0], device=device),
            'orig_size': torch.as_tensor([64, 64], device=device),
            'size': torch.as_tensor([64, 64], device=device)
        }
        return torch.stack([dummy_img]), [dummy_target]

    # Optimize: Use tensor operations for finding max dimensions
    heights = torch.tensor([img.shape[1] for img in images])
    widths = torch.tensor([img.shape[2] for img in images])
    max_h = heights.max().item()
    max_w = widths.max().item()

    # Optimize: Pre-allocate padded images tensor for better memory efficiency
    batch_size = len(images)
    channels = images[0].shape[0]

    # Always create on CPU to prevent device mismatch errors
    device = torch.device('cpu')
    padded_images = torch.zeros((batch_size, channels, max_h, max_w),
                          dtype=images[0].dtype, device=device)

    # Fill padded_images tensor with actual image data
    for i, img in enumerate(images):
        # Always ensure image is on CPU
        if img.device.type == 'cuda':
            img = img.cpu()

        h, w = img.shape[1], img.shape[2]
        padded_images[i, :, :h, :w] = img

        # Update image size in target
        if i < len(targets):
            # Update size information
            targets[i]['size'] = torch.tensor([max_h, max_w], device='cpu')

            # Ensure all tensors in target are on the same device
            for k, v in targets[i].items():
                if isinstance(v, torch.Tensor) and v.device.type == 'cuda':
                    targets[i][k] = v.cpu()

    # Handle missing targets if needed (rare case)
    if len(targets) < len(padded_images):
        # Create empty targets on CPU
        device = torch.device('cpu')
        # Add empty targets to match the number of images
        for i in range(len(targets), len(padded_images)):
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=device),
                'image_id': torch.tensor([i], device=device),
                'orig_size': torch.as_tensor([max_h, max_w], device=device),
                'size': torch.as_tensor([max_h, max_w], device=device)
            })

    return padded_images, targets


def build_dataset(root_dir, annotations_file=None, split='train', img_size=None, cache_images=None, cache_limit=None, config=None, device=None):
    """
    Builds and configures a dataset for the specified split with appropriate transforms

    This function creates an AU-AIR dataset with consistent configuration for both training
    and validation splits. It uses the same transforms for all splits (no data augmentation)
    to maintain consistency.

    Args:
        root_dir: Root directory containing the dataset
        annotations_file: Path to annotations JSON file (defaults to metadata/{split}_annotations.json)
        split: Dataset split to use ('train', 'val', or 'test')
        img_size: Base image size for resizing (height)
        cache_images: Whether to enable image caching
        cache_limit: Maximum number of images to cache in memory
        config: Configuration dictionary with dataset parameters
        device: Device to use for tensor operations (ignored, included for compatibility)

    Returns:
        Configured AUAIRDataset instance ready for use in a DataLoader
    """
    import gc
    from model.transforms import create_validation_transforms

    # Run garbage collection before creating dataset
    gc.collect()

    # Get configuration values from config if provided
    if config is not None:
        if img_size is None and 'dataset' in config and 'img_size' in config['dataset']:
            img_size = config['dataset']['img_size'][0]  # Use height from config
        if cache_images is None and 'dataset' in config and 'cache_images' in config['dataset']:
            cache_images = config['dataset']['cache_images']
        if cache_limit is None and 'dataset' in config and 'cache_limit' in config['dataset']:
            cache_limit = config['dataset']['cache_limit']

    # Set defaults if not provided
    if img_size is None:
        img_size = 320
    if cache_images is None:
        cache_images = False
    if cache_limit is None:
        cache_limit = 0

    # Set maximum image size for resizing based on aspect ratio
    if config is not None and 'dataset' in config and 'img_size' in config['dataset'] and len(config['dataset']['img_size']) > 1:
        # Use width from config
        max_size = config['dataset']['img_size'][1]
    else:
        # Default max size
        max_size = 480

    # Create transforms for image preprocessing
    transform = create_validation_transforms(img_size, max_size, config)

    if annotations_file is None:
        annotations_file = os.path.join('metadata', f'{split}_annotations.json')

    # Create dataset with config
    dataset = AUAIRDataset(
        root_dir=root_dir,
        annotations_file=annotations_file,
        split=split,
        transform=transform,
        cache_images=cache_images,
        cache_limit=cache_limit
    )

    # Pass config to dataset if available
    if config is not None:
        dataset.config = config
        # Set image size from config for fallback in __getitem__
        if 'dataset' in config and 'img_size' in config['dataset']:
            dataset.img_size = config['dataset']['img_size']

    # Store device in dataset for later use
    if device is not None:
        dataset.device = device

    # Run garbage collection after creating dataset
    gc.collect()

    return dataset


def memory_efficient_collate(batch, dataset=None):
    """Memory-efficient collate function for object detection batches.

    Args:
        batch: List of (image, target) tuples from the dataset
        dataset: Optional dataset instance to access config and device information

    Returns:
        Tuple of (padded_images, targets) where padded_images is a tensor with consistent dimensions
    """
    import gc
    # Run garbage collection before processing batch
    gc.collect()

    images = []
    targets = []

    # Extract all valid images and targets from the batch
    for img_target in batch:
        if img_target is None:
            continue

        if isinstance(img_target, tuple) and len(img_target) == 2:
            image, target = img_target
            # Keep the image on its current device (CPU or GPU)
            # We'll handle device placement later based on config
            images.append(image)

            # Keep the target on its current device
            # We'll handle device placement later based on config
            targets.append(target)
        else:
            continue  # Skip invalid samples

    # Run garbage collection
    gc.collect()

    if len(images) == 0:
        # Create a minimal dummy batch for empty batches
        # Create tensors on GPU for A100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_img = torch.zeros((3, 64, 64), dtype=torch.float32, device=device)
        dummy_target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
            'labels': torch.zeros((0,), dtype=torch.int64, device=device),
            'image_id': torch.tensor([0], device=device),
            'orig_size': torch.as_tensor([64, 64], device=device),
            'size': torch.as_tensor([64, 64], device=device)
        }
        # Run garbage collection
        gc.collect()
        return torch.stack([dummy_img]), [dummy_target]

    # Use tensor operations for finding max dimensions
    heights = torch.tensor([img.shape[1] for img in images])
    widths = torch.tensor([img.shape[2] for img in images])
    max_h = heights.max().item()
    max_w = widths.max().item()

    # Pre-allocate padded images tensor
    batch_size = len(images)
    channels = images[0].shape[0]

    # Get device from dataset or config or use CUDA by default for A100
    if dataset is not None:
        if hasattr(dataset, 'config') and 'inference' in dataset.config and 'device' in dataset.config['inference']:
            device_str = dataset.config['inference']['device']
            device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
        elif hasattr(dataset, 'device') and dataset.device is not None:
            # Use device from dataset if available
            device = dataset.device
        else:
            # Default to CUDA for A100
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # Default to CUDA for A100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    padded_images = torch.zeros((batch_size, channels, max_h, max_w),
                            dtype=images[0].dtype, device=device)

    # Fill padded_images tensor with actual image data
    for i, img in enumerate(images):
        # Move image to GPU if it's not already there
        if img.device.type != 'cuda':
            img = img.to(device, non_blocking=True)

        h, w = img.shape[1], img.shape[2]
        padded_images[i, :, :h, :w] = img

        # Update image size in target
        if i < len(targets):
            targets[i]['size'] = torch.tensor([max_h, max_w], device=device)

            # Move all tensors in target to GPU
            for k, v in targets[i].items():
                if isinstance(v, torch.Tensor) and v.device.type != 'cuda':
                    targets[i][k] = v.to(device, non_blocking=True)

    # Handle missing targets if needed
    if len(targets) < len(padded_images):
        for i in range(len(targets), len(padded_images)):
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=device),
                'image_id': torch.tensor([i], device=device),
                'orig_size': torch.as_tensor([max_h, max_w], device=device),
                'size': torch.as_tensor([max_h, max_w], device=device)
            })

    # Clean up intermediate tensors
    del heights, widths

    # Run garbage collection
    gc.collect()

    return padded_images, targets


def collate_fn_with_dataset(batch, dataset):
    """Wrapper function to pass dataset to memory_efficient_collate

    Args:
        batch: List of (image, target) tuples from the dataset
        dataset: Dataset instance to access config and device information

    Returns:
        Result of memory_efficient_collate
    """
    return memory_efficient_collate(batch, dataset)


def build_dataloader(dataset, batch_size=None, shuffle=True, config=None, drop_last=None):
    """
    Build a DataLoader with custom collation function for object detection

    This function creates a PyTorch DataLoader with a custom collation function
    that handles variable-sized images and properly formats targets for the model.

    Args:
        dataset: Dataset instance to load data from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset for randomized training
        config: Configuration dictionary with training parameters
        drop_last: Whether to drop the last incomplete batch (recommended for training)

    Returns:
        DataLoader instance configured for object detection tasks
    """
    import gc

    # Run garbage collection before creating dataloader
    gc.collect()

    # Get configuration values from config if provided
    num_workers = None
    pin_memory = None
    prefetch_factor = None
    persistent_workers = None

    if config is not None:
        if batch_size is None and 'training' in config and 'batch_size' in config['training']:
            batch_size = config['training']['batch_size']
        if 'training' in config and 'num_workers' in config['training']:
            num_workers = config['training']['num_workers']
        if 'training' in config and 'pin_memory' in config['training']:
            pin_memory = config['training']['pin_memory']
        if 'training' in config and 'prefetch_factor' in config['training']:
            prefetch_factor = config['training']['prefetch_factor']
        if 'training' in config:
            persistent_workers = True  # Default to True if config is provided
        if drop_last is None and 'training' in config:
            # Use drop_last=True for training, False for validation/test
            drop_last = True if hasattr(dataset, 'split') and dataset.split == 'train' else False

    # Set defaults if not provided
    if batch_size is None:
        batch_size = 16
    if num_workers is None:
        num_workers = 2
    if pin_memory is None:
        pin_memory = False
    if prefetch_factor is None:
        prefetch_factor = 2
    if persistent_workers is None:
        persistent_workers = True
    if drop_last is None:
        drop_last = False

    # Note: persistent_workers will be handled conditionally when creating the DataLoader

    # Setup sampler for shuffling data
    sampler = None
    if shuffle:
        # Use RandomSampler instead of shuffle parameter for better performance
        sampler = torch.utils.data.RandomSampler(dataset)
        shuffle = False

    # Create a partial function that binds the dataset to the collate function
    import functools
    collate_fn = functools.partial(collate_fn_with_dataset, dataset=dataset)

    # Create dataloader with custom collate function
    loader_args = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle if sampler is None else False,
        'sampler': sampler,
        'collate_fn': collate_fn,  # Use the partial function we created
        'drop_last': drop_last,
        # Always use spawn method for multiprocessing to prevent CUDA initialization issues
        'multiprocessing_context': 'spawn',
    }

    # Add optional parameters if they are specified
    if num_workers is not None:
        loader_args['num_workers'] = num_workers
        if prefetch_factor is not None and num_workers > 0:
            loader_args['prefetch_factor'] = prefetch_factor
        if persistent_workers is not None and num_workers > 0:
            loader_args['persistent_workers'] = persistent_workers

    # Disable pin_memory when tensors are already on CUDA
    # This prevents the "cannot pin 'torch.cuda.FloatTensor'" error
    if pin_memory is not None:
        # Check if dataset has tensors already on CUDA
        has_cuda_tensors = False
        if hasattr(dataset, 'device') and dataset.device.type == 'cuda':
            has_cuda_tensors = True

        # Only enable pin_memory if tensors are not already on CUDA
        loader_args['pin_memory'] = pin_memory and not has_cuda_tensors

    loader = DataLoader(**loader_args)

    # Run garbage collection after creating dataloader
    gc.collect()

    return loader


# Dummy transforms to complete the implementation
# These would be replaced with proper transforms in a real implementation
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if np.random.random() < self.p:
            # Flip image
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Flip boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes[:, 0] = 1 - boxes[:, 0]  # Flip center x
                target['boxes'] = boxes

        return img, target


class RandomSelect:
    def __init__(self, transform1, transform2, p=0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, img, target):
        if np.random.random() < self.p:
            return self.transform1(img, target)
        return self.transform2(img, target)


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        size = np.random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSizeCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w, h = img.size

        # Choose random crop size
        tw = np.random.randint(self.min_size, min(w, self.max_size))
        th = np.random.randint(self.min_size, min(h, self.max_size))

        # Choose random position
        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)

        # Crop image
        img = img.crop((x1, y1, x1 + tw, y1 + th))

        # Adjust boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            # Convert to [x1, y1, x2, y2] format for cropping
            boxes_xyxy = box_cxcywh_to_xyxy(boxes.clone())
            boxes_xyxy[:, 0::2] *= w
            boxes_xyxy[:, 1::2] *= h

            # Crop boxes
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2] - x1
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2] - y1

            # Clip boxes to crop area
            boxes_xyxy[:, 0::2].clamp_(min=0, max=tw)
            boxes_xyxy[:, 1::2].clamp_(min=0, max=th)

            # Check if boxes are valid
            keep = (boxes_xyxy[:, 3] > boxes_xyxy[:, 1]) & (boxes_xyxy[:, 2] > boxes_xyxy[:, 0])
            boxes_xyxy = boxes_xyxy[keep]

            # Convert back to [cx, cy, w, h] and normalize
            if len(boxes_xyxy) > 0:
                boxes = box_xyxy_to_cxcywh(boxes_xyxy)
                boxes[:, 0::2] /= tw
                boxes[:, 1::2] /= th
                target['boxes'] = boxes
                target['labels'] = target['labels'][keep]
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)

        target['orig_size'] = torch.tensor([h, w])
        target['size'] = torch.tensor([th, tw])

        return img, target


def resize(img, target, size, max_size=None):
    """
    Resize image and adjust target accordingly while preserving aspect ratio
    """
    # Get original dimensions
    w, h = img.size  # PIL Image size is (width, height)

    # Calculate target size while preserving aspect ratio
    if h <= w:
        # If height is smaller, resize height to target size
        new_h = size
        new_w = int(size * w / h)

        # Apply max_size constraint to width if needed
        if max_size is not None and new_w > max_size:
            new_w = max_size
            new_h = int(max_size * h / w)
    else:
        # If width is smaller, resize width to target size
        new_w = size
        new_h = int(size * h / w)

        # Apply max_size constraint to height if needed
        if max_size is not None and new_h > max_size:
            new_h = max_size
            new_w = int(max_size * w / h)

    # Resize image with proper aspect ratio
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Adjust target
    if 'boxes' in target and len(target['boxes']) > 0:
        # Boxes are already normalized, no need to adjust
        pass

    # Update size information in target
    target['orig_size'] = torch.tensor([h, w])
    target['size'] = torch.tensor([new_h, new_w])

    return img, target


# Add these to torchvision.transforms namespace for compatibility
T.RandomHorizontalFlip = RandomHorizontalFlip
T.RandomSelect = RandomSelect
T.RandomResize = RandomResize
T.RandomSizeCrop = RandomSizeCrop