import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from model.utils import box_xyxy_to_cxcywh


class AUAIRDataset(Dataset):
    """
    Dataset class for AU-AIR dataset
    
    Args:
        root_dir: root directory of the dataset
        annotations_file: path to annotations file
        split: train, val, or test
        transform: image transforms
    """
    def __init__(self, root_dir, annotations_file, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
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
        
        # Class names to class indices
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
            annotation = self.annotations[idx]
            
            # Load image
            img_path = os.path.join(self.root_dir, 'images', annotation['image_name'])
            
            # Check if file exists, if not, try other paths
            if not os.path.exists(img_path):
                img_path = os.path.join('dataset', 'images', annotation['image_name'])
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")
            
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
                
                # Normalize box coordinates
                image_width = annotation.get('image_width', 1920.0)  # Use default if not present
                image_height = annotation.get('image_height', 1080.0)
                
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
            
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # Convert boxes to center format for DETR
            if len(boxes) > 0:
                boxes = box_xyxy_to_cxcywh(boxes)
                
                # Normalize box coordinates
                image_width = annotation.get('image_width', 1920.0)
                image_height = annotation.get('image_height', 1080.0)
                boxes[:, 0] /= image_width
                boxes[:, 1] /= image_height
                boxes[:, 2] /= image_width
                boxes[:, 3] /= image_height
            
            # Create target dictionary
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'orig_size': torch.as_tensor([annotation.get('image_height', 1080.0), 
                                            annotation.get('image_width', 1920.0)]),
                'size': torch.as_tensor([annotation.get('image_height', 1080.0), 
                                       annotation.get('image_width', 1920.0)])
            }
            
            # Apply transformations
            if self.transform is not None:
                image, target = self.transform(image, target)
            
            return image, target
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}")
            # Return a dummy sample
            if idx > 0:
                return self.__getitem__(0)
            else:
                # Create an empty sample as a fallback
                img = torch.zeros((3, 800, 800), dtype=torch.float32)
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'image_id': torch.tensor([idx]),
                    'orig_size': torch.as_tensor([1080, 1920]),
                    'size': torch.as_tensor([1080, 1920])
                }
                return img, target


def collate_fn(batch):
    """
    Custom collate function for the dataloader
    
    Handles batches with images of different sizes by padding them to the max size in the batch
    Ensures all targets are properly included and aligned with images
    """
    images = []
    targets = []
    
    # Extract all images and targets from the batch
    for img_target in batch:
        if img_target is None:
            continue
            
        if isinstance(img_target, tuple) and len(img_target) == 2:
            image, target = img_target
            images.append(image)
            targets.append(target)
        else:
            print(f"Warning: Unexpected batch element format: {type(img_target)}. Skipping.")
    
    if len(images) == 0:
        print("Warning: Empty batch after filtering. Creating dummy batch.")
        # Create a dummy batch with a small black image and empty targets
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_img = torch.zeros((3, 100, 100), dtype=torch.float32, device=device)
        dummy_target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
            'labels': torch.zeros((0,), dtype=torch.int64, device=device),
            'image_id': torch.tensor([0], device=device),
            'orig_size': torch.as_tensor([100, 100], device=device),
            'size': torch.as_tensor([100, 100], device=device)
        }
        return torch.stack([dummy_img]), [dummy_target]
    
    # Find max dimensions in the batch
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    # Pad all images to the same size
    padded_images = []
    for i, img in enumerate(images):
        # Calculate padding needed for each image
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        # Apply padding (right and bottom padding only)
        padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
        padded_img = torch.nn.functional.pad(img, padding, value=0)
        padded_images.append(padded_img)
        
        # Update image size in target
        if i < len(targets):
            targets[i]['size'] = torch.tensor([max_h, max_w])
    
    # Ensure targets length is the same as images length
    if len(targets) < len(padded_images):
        print(f"Warning: {len(targets)} targets for {len(padded_images)} images. Adding empty targets.")
        device = padded_images[0].device
        
        # Add empty targets to match the number of images
        while len(targets) < len(padded_images):
            dummy_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=device),
                'image_id': torch.tensor([len(targets)], device=device),
                'orig_size': torch.as_tensor([max_h, max_w], device=device),
                'size': torch.as_tensor([max_h, max_w], device=device)
            }
            targets.append(dummy_target)
    
    # Stack padded images
    images = torch.stack(padded_images)
    
    return images, targets


def build_dataset(root_dir, annotations_file=None, split='train', img_size=800):
    """
    Build AU-AIR dataset with proper transforms
    
    Args:
        root_dir: root directory of the dataset
        annotations_file: path to annotations file
        split: train, val, or test
        img_size: target image size
        
    Returns:
        dataset: dataset object
    """
    # Import proper transforms
    from model.transforms import create_training_transforms, create_validation_transforms
    
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    
    if split == 'train':
        # For training, use data augmentation
        transform = create_training_transforms(scales, max_size)
    else:
        # For validation and testing, just resize
        transform = create_validation_transforms(img_size, max_size)
    
    # If annotations_file is not provided, use default based on split
    if annotations_file is None:
        annotations_file = os.path.join('metadata', f'{split}_annotations.json')
        print(f"Using default annotations file: {annotations_file}")
    
    dataset = AUAIRDataset(
        root_dir=root_dir,
        annotations_file=annotations_file,
        split=split,
        transform=transform
    )
    
    return dataset


def build_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=False):
    """
    Build dataloader for the dataset
    
    Args:
        dataset: dataset object
        batch_size: batch size
        shuffle: whether to shuffle the dataset
        num_workers: number of workers for data loading
        pin_memory: whether to pin memory for faster data transfer to GPU
        prefetch_factor: number of samples to prefetch per worker
        persistent_workers: whether to keep workers alive between epochs
        
    Returns:
        dataloader: dataloader object
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    
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
    Resize image and adjust target accordingly
    """
    # Size is the target height
    h, w = img.size
    target_h = size
    target_w = size
    
    if max_size is not None:
        target_h = min(target_h, max_size)
        target_w = min(target_w, max_size)
    
    # Resize image
    img = img.resize((target_h, target_w))
    
    # Adjust target
    if 'boxes' in target and len(target['boxes']) > 0:
        # Boxes are already normalized
        pass
    
    target['orig_size'] = torch.tensor([h, w])
    target['size'] = torch.tensor([target_h, target_w])
    
    return img, target


# Add these to torchvision.transforms namespace for compatibility
T.RandomHorizontalFlip = RandomHorizontalFlip
T.RandomSelect = RandomSelect
T.RandomResize = RandomResize
T.RandomSizeCrop = RandomSizeCrop 