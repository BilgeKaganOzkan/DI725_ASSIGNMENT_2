"""
Deformable DETR Model Training File

This file contains all the necessary functions and workflows for training
the Deformable DETR model on the AU-AIR dataset. Deformable DETR is an
AI-based object detection model that is a more efficient and faster version
of the standard DETR model.

Usage:
    python train.py --config config/deformable_detr_config.yaml --device cuda --wandb

Parameters:
    --config: YAML file containing model and training configuration
    --device: Device to use for training (cuda, cpu)
    --wandb: Enables Weights & Biases integration
    --resume: Checkpoint file to continue training
    --dataset-path: Location of the dataset
    --annotations-file: Location of the annotation file

© 2023 AU-AIR Dataset and Deformable DETR Implementation
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import wandb
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

# Configure PyTorch to suppress compile errors and fall back to eager mode
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from model.deformable_detr import DeformableDetrModel
from model.criterion import DeformableDETRLoss
from model.dataset import build_dataset, build_dataloader
from model.utils import HungarianMatcher, get_num_parameters, MetricLogger, SmoothedValue

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from model.utils import box_cxcywh_to_xyxy
import torch
import torch.nn.functional as F
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train Deformable DETR on AU-AIR dataset")
    parser.add_argument('--config', type=str, default='config/deformable_detr_config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset-path', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--annotations-file', type=str, default=None,
                       help='Path to annotations file (default: metadata/{split}_annotations.json)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint for resuming training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (cuda or cpu)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling for performance analysis')
    return parser.parse_args()


def set_seed(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, 
                   grad_accumulation=1, scaler=None, clip_max_norm=0, print_freq=10, 
                   visualize_freq=None, output_dir=None, wandb_enabled=False):
    """
    Trains the model for one epoch.
    
    Parameters:
        model: Model to be trained
        criterion: Loss function
        optimizer: Optimization algorithm
        data_loader: Training data loader
        device: Training device (CPU/GPU)
        epoch: Current epoch number
        grad_accumulation: Number of gradient accumulation steps
        scaler: Gradient scaler for mixed precision training
        clip_max_norm: Maximum norm for gradient clipping
        print_freq: Frequency of printing progress information
        visualize_freq: Visualization frequency
        output_dir: Output directory
        wandb_enabled: Whether Weights & Biases integration is enabled
    
    Returns:
        Average training loss, samples selected for visualization
    """
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    
    # Get current optimization settings
    for group in optimizer.param_groups:
        if 'lr' in group:
            metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            break
    
    header = f"Epoch: [{epoch}]"
    
    # Counter for gradient accumulation
    batch_count = 0
    
    # Record start time
    start_time = time.time()
    
    # Data for visualizing examples
    visualize_samples = []
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        
        # Losses
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Scale losses for gradient accumulation
        if grad_accumulation > 1:
            losses = losses / grad_accumulation
            for k in loss_dict:
                loss_dict[k] = loss_dict[k] / grad_accumulation
        
        # Backward
        if scaler:
            # Mixed precision training
            scaler.scale(losses).backward()
            if (batch_count + 1) % grad_accumulation == 0 or (batch_count + 1) == len(data_loader):
                if clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Full precision training
            losses.backward()
            if (batch_count + 1) % grad_accumulation == 0 or (batch_count + 1) == len(data_loader):
                if clip_max_norm > 0:
                    torch.nn.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        # Save samples for wandb logging and visualization
        if wandb_enabled and visualize_freq and batch_count % visualize_freq == 0:
            # Add if not enough samples have been collected yet
            if len(visualize_samples) < 2:  # Take at most 2 examples
                visualize_samples.append((samples.detach().clone(), targets, outputs))
        
        # Update statistics
        metric_logger.update(loss=losses.item())
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})
        
        batch_count += 1
    
    # Visualize training examples
    if wandb_enabled and visualize_samples:
        import wandb
        for i, (samples, targets, outputs) in enumerate(visualize_samples):
            wandb_images = visualize_sample(samples, targets, outputs, num_samples=2, 
                                        save_dir=os.path.join(output_dir, f'train_vis_epoch_{epoch}') if output_dir else None,
                                        wandb_log=True)
            # Send images to wandb
            if wandb_images:
                wandb.log({f"train_samples_{i+1}": wandb_images})
    
    # Update training statistics at the end of the epoch
    end_time = time.time()
    epoch_time = end_time - start_time
    samples_per_sec = len(data_loader.dataset) / epoch_time
    
    # Show training statistics
    print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
    print(f"Samples per second: {samples_per_sec:.2f}")
    
    # Wandb metrics
    if wandb_enabled:
        import wandb
        # Send training loss values to wandb
        log_dict = {
            'train/loss': metric_logger.meters['loss'].global_avg,
            'train/samples_per_sec': samples_per_sec,
            'train/epoch_time': epoch_time
        }
        
        # Add other loss values
        for k, meter in metric_logger.meters.items():
            if k != 'loss':
                log_dict[f'train/{k}'] = meter.global_avg
        
        # Add learning rate
        if 'lr' in metric_logger.meters:
            log_dict['train/lr'] = metric_logger.meters['lr'].value
        
        # Send to wandb
        wandb.log(log_dict, step=epoch)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, criterion, data_loader, device, epoch=None,
              print_freq=10, visualize_freq=None, output_dir=None, wandb_enabled=False):
    """
    Evaluates the model on validation dataset.
    
    Parameters:
        model: Model to be evaluated
        criterion: Loss function
        data_loader: Validation data loader
        device: Evaluation device (CPU/GPU)
        epoch: Current epoch number
        print_freq: Frequency of printing progress information
        visualize_freq: Visualization frequency
        output_dir: Output directory
        wandb_enabled: Whether Weights & Biases integration is enabled
    
    Returns:
        Average validation loss, samples selected for visualization
    """
    model.eval()
    criterion.eval()
    
    metric_logger = MetricLogger(delimiter="  ")
    header = "Validation:"
    
    # Record start time
    start_time = time.time()
    
    # Data for visualizing examples
    visualize_samples = []
    
    with torch.no_grad():
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            
            # Losses
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Save samples for wandb visualization
            if wandb_enabled and visualize_freq and len(visualize_samples) < 2:
                visualize_samples.append((samples.detach().clone(), targets, outputs))
            
            # Update statistics
            metric_logger.update(loss=losses.item())
            for k, v in loss_dict.items():
                metric_logger.update(**{k: v.item()})
    
    # Visualize validation examples
    if wandb_enabled and visualize_samples:
        import wandb
        for i, (samples, targets, outputs) in enumerate(visualize_samples):
            wandb_images = visualize_sample(samples, targets, outputs, num_samples=2, 
                                        save_dir=os.path.join(output_dir, f'val_vis_epoch_{epoch}') if output_dir else None,
                                        wandb_log=True)
            # Send images to wandb
            if wandb_images:
                wandb.log({f"val_samples_{i+1}": wandb_images})
    
    # Update validation statistics at the end of the epoch
    end_time = time.time()
    val_time = end_time - start_time
    samples_per_sec = len(data_loader.dataset) / val_time
    
    # Show validation statistics
    print(f"Validation completed in {val_time:.2f} seconds")
    print(f"Samples per second: {samples_per_sec:.2f}")
    
    # Wandb metrics
    if wandb_enabled:
        import wandb
        # Send validation loss values to wandb
        log_dict = {
            'val/loss': metric_logger.meters['loss'].global_avg,
            'val/samples_per_sec': samples_per_sec,
            'val/epoch_time': val_time
        }
        
        # Add other loss values
        for k, meter in metric_logger.meters.items():
            if k != 'loss':
                log_dict[f'val/{k}'] = meter.global_avg
        
        # Save to wandb
        epoch_step = epoch if epoch is not None else 0
        wandb.log(log_dict, step=epoch_step)
    
    print("Avg. validation loss:", metric_logger.meters['loss'].global_avg)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sample_dataset(dataset, fraction):
    """
    Randomly samples a fraction of the dataset.
    
    Parameters:
        dataset: Dataset to be sampled
        fraction: Sampling ratio (between 0-1)
    
    Returns:
        Sampled dataset
    """
    if fraction >= 1.0:
        return dataset
    
    # Select random indices from the dataset
    dataset_size = len(dataset)
    num_samples = int(dataset_size * fraction)
    indices = random.sample(range(dataset_size), num_samples)
    
    # Create Subset
    return Subset(dataset, indices)


def visualize_predictions(img, target, output, class_names):
    """
    Wandb için tahminleri görselleştir
    
    Args:
        img: Tensor image [C, H, W]
        target: Target dict with 'boxes', 'labels'
        output: Output dict with 'pred_logits', 'pred_boxes'
        class_names: List of class names
        
    Returns:
        wandb_img: Wandb Image object with annotations
    """
    # Image'ı numpy formatına dönüştür
    img_np = img.permute(1, 2, 0).numpy()
    
    # Normalize'ı geri al
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = img_np * std + mean
    img_np = (img_np * 255).astype(np.uint8)
    
    # Get dimensions
    height, width = img_np.shape[:2]
    
    # Ground truth boxes
    gt_boxes = []
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        # Convert to xyxy format
        from model.utils import box_cxcywh_to_xyxy
        boxes = box_cxcywh_to_xyxy(boxes)
        
        # Denormalize
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height
        
        # Create box data
        for box, label in zip(boxes, labels):
            label_idx = label.item()
            if label_idx < len(class_names):
                gt_boxes.append({
                    "position": {
                        "minX": box[0].item(),
                        "minY": box[1].item(),
                        "maxX": box[2].item(),
                        "maxY": box[3].item()
                    },
                    "class_id": label_idx,
                    "box_caption": class_names[label_idx]
                })
    
    # Prediction boxes
    pred_boxes = []
    if 'pred_logits' in output and 'pred_boxes' in output:
        # Get confidence scores
        scores = torch.nn.functional.softmax(output['pred_logits'], dim=-1)
        # Remove background class (last column)
        scores = scores[:, :-1]
        
        # Get max scores and labels
        max_scores, labels = scores.max(dim=1)
        
        # Get boxes
        boxes = output['pred_boxes']
        
        # Filter out low confidence predictions
        keep = max_scores > 0.5
        boxes = boxes[keep]
        labels = labels[keep]
        scores = max_scores[keep]
        
        if len(boxes) > 0:
            # Convert to xyxy format
            boxes = box_cxcywh_to_xyxy(boxes)
            
            # Denormalize
            boxes[:, 0::2] *= width
            boxes[:, 1::2] *= height
            
            # Create box data
            for box, label, score in zip(boxes, labels, scores):
                label_idx = label.item()
                if label_idx < len(class_names):
                    pred_boxes.append({
                        "position": {
                            "minX": box[0].item(),
                            "minY": box[1].item(),
                            "maxX": box[2].item(),
                            "maxY": box[3].item()
                        },
                        "class_id": label_idx,
                        "box_caption": f"{class_names[label_idx]} {score.item():.2f}",
                        "scores": {"confidence": score.item()}
                    })
    
    # Create wandb image
    return wandb.Image(img_np, boxes={
        "ground_truth": {
            "box_data": gt_boxes,
            "class_labels": {i: name for i, name in enumerate(class_names)}
        },
        "predictions": {
            "box_data": pred_boxes,
            "class_labels": {i: name for i, name in enumerate(class_names)}
        }
    })


def create_wandb_boxes(target, output):
    """
    Prepares bounding box data for W&B visualization.
    
    Parameters:
        target: Ground truth bounding boxes
        output: Bounding boxes predicted by the model
    
    Returns:
        Bounding box data in W&B format
    """
    from model.utils import box_cxcywh_to_xyxy
    import torch.nn.functional as F
    
    # Prepare wandb bounding box dict
    class_names = ['airplane', 'bird', 'drone', 'helicopter', 'person', 'truck', 'boat', 'car']
    box_data = []
    
    # Ground truth boxes
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        # Convert from center format to corner format
        boxes = box_cxcywh_to_xyxy(boxes)
        
        # Create box data for each ground truth
        for box, label_id in zip(boxes, labels):
            if isinstance(label_id, torch.Tensor):
                label_id = label_id.item()
            
            if label_id < len(class_names):
                box_data.append({
                    "position": {
                        "minX": box[0].item(),
                        "minY": box[1].item(),
                        "maxX": box[2].item(),
                        "maxY": box[3].item()
                    },
                    "class_id": label_id,
                    "box_caption": f"GT: {class_names[label_id]}",
                    "domain": "pixel",
                    "scores": {"score": 1.0}
                })
    
    # Prediction boxes
    if 'pred_logits' in output and 'pred_boxes' in output:
        # Get confidence scores
        logits = output['pred_logits']
        if isinstance(logits, torch.Tensor):
            scores = F.softmax(logits, dim=-1)
            # Remove background class (if it's the last class)
            scores = scores[:, :-1]
            
            # Get max scores and corresponding labels
            max_scores, pred_labels = scores.max(dim=1)
            
            # Get predicted boxes
            pred_boxes = output['pred_boxes']
            
            # Filter low confidence predictions (threshold = 0.5)
            keep_indices = max_scores > 0.5
            filtered_boxes = pred_boxes[keep_indices]
            filtered_labels = pred_labels[keep_indices]
            filtered_scores = max_scores[keep_indices]
            
            if len(filtered_boxes) > 0:
                # Convert from center format to corner format
                filtered_boxes = box_cxcywh_to_xyxy(filtered_boxes)
                
                # Create box data for each prediction
                for box, label_id, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                    if isinstance(label_id, torch.Tensor):
                        label_id = label_id.item()
                    
                    if label_id < len(class_names):
                        box_data.append({
                            "position": {
                                "minX": box[0].item(),
                                "minY": box[1].item(),
                                "maxX": box[2].item(),
                                "maxY": box[3].item()
                            },
                            "class_id": label_id,
                            "box_caption": f"Pred: {class_names[label_id]} ({score.item():.2f})",
                            "domain": "pixel",
                            "scores": {"confidence": score.item()}
                        })
    
    return {
        "boxes": {
            "box_data": box_data,
            "class_labels": {i: name for i, name in enumerate(class_names)}
        }
    }


def visualize_sample(images, targets, outputs, num_samples=2, save_dir=None, wandb_log=False):
    """
    Visualize images, ground truth boxes, and predicted boxes
    """
    
    # Class names and colors
    class_names = ['airplane', 'bird', 'drone', 'helicopter', 'person', 'truck', 'boat', 'car']
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
    
    # Limit the maximum number of samples to display
    num_samples = min(num_samples, len(images))
    
    # Prepare images
    all_wandb_images = []
    
    # Loop for each example
    for i in range(num_samples):
        # Get image and remove normalization
        img = images[i].detach().cpu().clone()
        # Reverse normalization
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # Convert to PIL image
        img_pil = to_pil_image(img)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(np.array(img_pil))
        
        # Draw ground truth boxes
        if targets and i < len(targets):
            target = targets[i]
            boxes = target['boxes']
            labels = target['labels']
            
            # Convert boxes to (x1, y1, x2, y2) format
            boxes = box_cxcywh_to_xyxy(boxes)
            
            # Loop for each box
            for j, (box, label_id) in enumerate(zip(boxes.detach().cpu(), labels.detach().cpu())):
                if isinstance(label_id, torch.Tensor):
                    label_id = label_id.item()
                
                # Get class name and color
                if 0 <= label_id < len(class_names):
                    label = class_names[label_id]
                    color = colors[label_id % len(colors)]
                    
                    # Draw box
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        box[0], box[1] - 5,
                        f"GT: {label}",
                        color='white', fontsize=10,
                        bbox=dict(facecolor=color, alpha=0.7)
                    )
        
        # Draw predicted boxes
        if outputs and i < len(outputs['pred_logits']):
            # Get confidence scores
            logits = outputs['pred_logits'][i]
            scores = F.softmax(logits, dim=-1)
            # Remove background class (if it's the last class)
            scores = scores[:, :-1]
            
            # Get highest scores and corresponding labels
            max_scores, pred_labels = scores.max(dim=1)
            
            # Get predicted boxes
            pred_boxes = outputs['pred_boxes'][i]
            
            # Filter low confidence predictions (threshold = 0.5)
            keep_indices = max_scores > 0.5
            filtered_boxes = pred_boxes[keep_indices]
            filtered_labels = pred_labels[keep_indices]
            filtered_scores = max_scores[keep_indices]
            
            # Convert boxes to (x1, y1, x2, y2) format
            filtered_boxes = box_cxcywh_to_xyxy(filtered_boxes)
            
            # Loop for each predicted box
            for box, label_id, score in zip(filtered_boxes.detach().cpu(), filtered_labels.detach().cpu(), filtered_scores.detach().cpu()):
                if isinstance(label_id, torch.Tensor):
                    label_id = label_id.item()
                
                # Get class name and color
                if 0 <= label_id < len(class_names):
                    label = class_names[label_id]
                    color = colors[label_id % len(colors)]
                    
                    # Draw box (with dashed line)
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none',
                        linestyle='--'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        box[0], box[3] + 10,
                        f"Pred: {label} ({score.item():.2f})",
                        color='white', fontsize=10,
                        bbox=dict(facecolor=color, alpha=0.7)
                    )
        
        # Remove axis labels
        ax.axis('off')
        plt.tight_layout()
        
        # Save image for wandb
        if wandb_log:
            import wandb
            target_wandb = targets[i] if i < len(targets) else {}
            output_wandb = {
                'pred_logits': outputs['pred_logits'][i:i+1] if outputs else None,
                'pred_boxes': outputs['pred_boxes'][i:i+1] if outputs else None
            }
            
            # Prepare wandb box data
            box_data = create_wandb_boxes(target_wandb, output_wandb)
            
            # Create wandb image
            wandb_image = wandb.Image(
                img_pil,
                boxes=box_data["boxes"] if box_data else None,
                caption=f"Sample {i+1}"
            )
            all_wandb_images.append(wandb_image)
        
        # Save to file
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'), bbox_inches='tight')
        
        plt.close(fig)
    
    return all_wandb_images


def main():
    """
    Main training function. Processes arguments, creates the model, and manages the training loop.
    """
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device - always try to use GPU first
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device used: {device}")
    
    # If using GPU, provide some information about the GPU being used
    if device.type == 'cuda':
        # Set CUDA optimization settings
        torch.backends.cuda.matmul.allow_tf32 = config['training'].get('use_tf32', True)  # Allow TF32 on matmul
        torch.backends.cudnn.allow_tf32 = config['training'].get('use_tf32', True)  # Allow TF32 on cudnn
        
        # Enable benchmarking for optimal kernel selection
        torch.backends.cudnn.benchmark = config['training'].get('benchmark_cudnn', True)
        
        # Optimize memory usage
        torch.cuda.empty_cache()
        # Set to deterministic for reproducibility if needed (but slower)
        torch.backends.cudnn.deterministic = False
        
        # Print GPU information
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        
        # Empty cache to start with a clean GPU memory
        torch.cuda.empty_cache()
    else:
        print("⚠️ WARNING: You are not using a GPU. Training on CPU will be very slow!")
        print("For GPU acceleration, run training with 'python train.py --config config/deformable_detr_config.yaml --device cuda'.")
    
    # Get mixed precision setting
    mixed_precision = config['training'].get('mixed_precision', False)
    if mixed_precision and device.type == 'cuda':
        print("Using mixed precision training (FP16) for faster performance")
        scaler = GradScaler()
    else:
        scaler = None
        if device.type == 'cuda':
            print("Using full precision training (FP32)")
    
    # Get gradient accumulation setting
    grad_accumulation = config['training'].get('grad_accumulation', 1)
    if grad_accumulation > 1:
        print(f"Using gradient accumulation with {grad_accumulation} steps")
        
    # Initialize wandb if enabled
    if args.wandb:
        # Simplified wandb configuration
        wandb_config = {
            # Model parameters
            "model_name": config['model']['name'],
            "hidden_dim": config['model']['hidden_dim'],
            "nheads": config['model']['nheads'],
            "num_encoder_layers": config['model']['num_encoder_layers'],
            "num_decoder_layers": config['model']['num_decoder_layers'],
            "num_queries": config['model']['num_queries'],
            
            # Training parameters
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['optimizer']['lr'],
            "lr_backbone": config['training']['lr_backbone'],
            "weight_decay": config['optimizer']['weight_decay'],
            "epochs": config['training']['epochs'],
            "lr_drop": config['training']['lr_drop'],
            "mixed_precision": mixed_precision,
            "grad_accumulation": grad_accumulation,
            
            # Dataset information
            "dataset_name": config['dataset']['name'],
            "num_classes": config['dataset']['num_classes'],
            "class_names": config['dataset']['class_names'],
            "img_size": config['dataset']['img_size'],
        }
        
        # Initialize wandb
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['name'],
            config=wandb_config,
            tags=config['wandb']['tags']
        )
    
    # Load datasets
    print("Loading datasets...")
    
    # Define annotations files based on split
    train_annotations_file = args.annotations_file
    val_annotations_file = None
    
    if train_annotations_file is None:
        train_annotations_file = os.path.join('metadata', 'train_annotations.json')
        val_annotations_file = os.path.join('metadata', 'val_annotations.json')
    
    # Load train and validation datasets
    train_dataset = build_dataset(
        root_dir=args.dataset_path,
        annotations_file=train_annotations_file,
        split='train',
        img_size=config['dataset']['img_size']
    )
    
    val_dataset = build_dataset(
        root_dir=args.dataset_path,
        annotations_file=val_annotations_file,
        split='val',
        img_size=config['dataset']['img_size']
    )
    
    # Apply dataset sampling if enabled
    sampling_enabled = config['dataset'].get('sampling', {}).get('enabled', False)
    if sampling_enabled:
        train_fraction = config['dataset']['sampling'].get('train_fraction', 1.0)
        val_fraction = config['dataset']['sampling'].get('val_fraction', 1.0)
        
        if train_fraction < 1.0:
            original_train_size = len(train_dataset)
            train_dataset = sample_dataset(train_dataset, train_fraction)
            print(f"Sampled training dataset from {original_train_size} to {len(train_dataset)} examples ({train_fraction:.0%})")
        
        if val_fraction < 1.0:
            original_val_size = len(val_dataset)
            val_dataset = sample_dataset(val_dataset, val_fraction)
            print(f"Sampled validation dataset from {original_val_size} to {len(val_dataset)} examples ({val_fraction:.0%})")
    
    print(f"Loaded datasets: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create data loaders with optimized settings
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    prefetch_factor = config['training'].get('prefetch_factor', 2)
    persistent_workers = num_workers > 0  # Keep workers alive between epochs
    
    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    val_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    # Build model with optimizations
    print("Building model...")
    model = DeformableDetrModel(
        num_classes=config['dataset']['num_classes'],
        hidden_dim=config['model']['hidden_dim'],
        nheads=config['model']['nheads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        num_feature_levels=config['model']['num_feature_levels'],
        enc_n_points=config['model']['enc_n_points'],
        dec_n_points=config['model']['dec_n_points'],
        num_queries=config['model']['num_queries']
    )
    
    # Get model statistics
    total_params = get_num_parameters(model)
    print(f"Model has {total_params:,} trainable parameters")
    
    # Move model to device
    model = model.to(device)
    
    # Use channels last memory format for better performance on CUDA
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("Using channels_last memory format for better GPU performance")
        
        # Optimize model with torch.compile if available (PyTorch 2.0+)
        if config['training'].get('compile_model', False):
            try:
                # Check if torch.compile is available (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    # Temporarily disable torch.compile to avoid Triton dependency issues
                    # print("Using torch.compile for optimized execution")
                    # compile_mode = "reduce-overhead"  # Options: 'default', 'reduce-overhead', 'max-autotune'
                    # model = torch.compile(model, mode=compile_mode)
                    print("Skipping torch.compile due to missing Triton dependency")
                else:
                    print("Warning: torch.compile is not available in your PyTorch version.")
            except Exception as e:
                print(f"Warning: Failed to compile model: {e}")
                
        # Alternative: Use JIT tracing for faster inference
        if not config['training'].get('compile_model', False) and config['training'].get('jit_optimize', False):
            try:
                # Create a sample input for tracing
                sample_input = torch.rand(1, 3, config['dataset']['img_size'][0], config['dataset']['img_size'][1], 
                                         device=device, dtype=torch.float32)
                sample_input = sample_input.to(memory_format=torch.channels_last)
                
                # Use torch.jit.trace
                print("Using torch.jit.trace for optimized execution")
                model = torch.jit.trace(model, sample_input)
                model = torch.jit.optimize_for_inference(model)
                print("JIT optimization applied")
            except Exception as e:
                print(f"Warning: Failed to JIT optimize model: {e}")
    
    # Watch model parameters automatically in wandb
    if args.wandb:
        wandb.watch(model, log="all", log_freq=100)
    
    # Create matcher and loss function
    matcher = HungarianMatcher(
        cost_class=1,
        cost_bbox=5,
        cost_giou=2
    )
    
    weight_dict = config['training']['loss_weights']
    
    criterion = DeformableDETRLoss(
        num_classes=config['dataset']['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['labels', 'boxes', 'cardinality']
    )
    
    # Compute total training steps for scheduler
    steps_per_epoch = len(train_dataloader) // grad_accumulation if grad_accumulation > 1 else len(train_dataloader)
    total_training_steps = steps_per_epoch * config['training']['epochs']
    
    # Set up optimizer and scheduler
    if config['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=float(config['optimizer']['lr']),
            weight_decay=float(config['optimizer']['weight_decay'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']['type']}")
    
    # Set up scheduler
    scheduler_type = config['optimizer'].get('scheduler_type', 'step')
    
    if scheduler_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config['optimizer'].get('lr_step_size', 30),
            gamma=config['optimizer'].get('lr_gamma', 0.1)
        )
    elif scheduler_type == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=float(config['optimizer']['lr']),
            total_steps=total_training_steps,
            pct_start=0.3
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    # Store lr_scheduler in config for train_one_epoch
    config['lr_scheduler'] = lr_scheduler
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...")
    best_val_loss = float('inf')
    
    # Enable TF32 for faster training on Ampere or newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory optimization
    torch.backends.cudnn.benchmark = True

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}:")
        
        # Train for one epoch
        train_loss, train_images, train_targets, train_outputs = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            grad_accumulation=grad_accumulation,
            scaler=scaler,
            clip_max_norm=0,
            print_freq=10,
            visualize_freq=None,
            output_dir=None,
            wandb_enabled=args.wandb
        )
        
        # Validate
        val_loss, val_images, val_targets, val_outputs = validate(
            model=model,
            criterion=criterion,
            data_loader=val_dataloader,
            device=device,
            epoch=epoch,
            print_freq=10,
            visualize_freq=None,
            output_dir=None,
            wandb_enabled=args.wandb
        )
        
        # Step scheduler (for StepLR - OneCycleLR is stepped in the train loop)
        if scheduler_type == 'step':
            lr_scheduler.step()
        
        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                'best_val_loss': best_val_loss,
                'config': config,
            }, os.path.join('checkpoints', 'best_model.pth'))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Also save checkpoint for the latest epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'best_val_loss': best_val_loss,
            'config': config,
        }, os.path.join('checkpoints', 'last_checkpoint.pth'))
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        samples_per_sec = len(train_dataset) / epoch_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds ({samples_per_sec:.2f} samples/second)")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'best_val_loss': best_val_loss,
                'epoch_time': epoch_time,
                'samples_per_sec': samples_per_sec
            })
            
            # Log validation predictions for visualization
            if epoch % 5 == 0 and len(val_images) > 0:
                wandb.log({
                    f"val_predictions_{epoch+1}": [
                        wandb.Image(
                            img.permute(1, 2, 0).numpy(), 
                            boxes=create_wandb_boxes(tgt, out)
                        )
                        for img, tgt, out in zip(val_images, val_targets, val_outputs)
                    ]
                })
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 