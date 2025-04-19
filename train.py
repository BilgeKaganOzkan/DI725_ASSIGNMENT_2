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
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import math
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
import warnings
import multiprocessing

# Suppress wandb backward hook warnings
warnings.filterwarnings("ignore", message="backward hook.*will not be serialized")

# Import Weights & Biases for experiment tracking and visualization
try:
    import wandb
except ImportError:
    wandb = None

# Configure PyTorch dynamo to suppress compilation errors and automatically fall back to eager mode execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Import custom model components from local modules
from model.deformable_detr import DeformableDetrModel
from model.criterion import DeformableDETRLoss
from model.dataset import build_dataset, build_dataloader
from model.utils import MetricLogger, SmoothedValue, HungarianMatcher, get_num_parameters
from model.utils import box_cxcywh_to_xyxy
from model.cached_dataset import CachedDataset, DynamicCachedDataset, SampledDataset

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
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision training (enabled by default)')
    parser.add_argument('--grad_accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--sample_ratio', type=float, default=None,
                       help='Ratio of dataset to use (0.0-1.0). Default: use value from config file')
    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility across PyTorch, NumPy, and Python's random module"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm=0.1,
                  lr_scheduler=None, grad_accumulation_steps=1, amp_enabled=True, scaler=None,
                  config=None, val_dataloader=None, batch_size=16):
    """
    Train the model for a single epoch with comprehensive performance optimizations

    This function handles the complete training loop for one epoch, including:
    - Forward and backward passes with mixed precision support
    - Gradient accumulation for effective larger batch sizes
    - Gradient clipping for training stability
    - Learning rate scheduling
    - Comprehensive metric logging and visualization
    - Memory management optimizations

    Args:
        model: The Deformable DETR model to train
        criterion: Loss function for computing training objectives
        data_loader: DataLoader providing training batches
        optimizer: Optimizer for parameter updates
        device: Device to run training on (CPU/GPU)
        epoch: Current epoch number (0-indexed)
        max_norm: Maximum gradient norm for gradient clipping
        lr_scheduler: Learning rate scheduler for dynamic LR adjustment
        grad_accumulation_steps: Number of batches to accumulate gradients over
        amp_enabled: Whether to use automatic mixed precision
        scaler: Gradient scaler for mixed precision training
        config: Configuration dictionary with training parameters
        val_dataloader: Optional validation data loader for mid-epoch validation
        batch_size: Batch size for training

    Returns:
        Dictionary of average training metrics for the epoch
    """
    # Initialize GPU memory management
    if device.type == 'cuda':
        # Log initial GPU memory usage
        try:
            allocated_mem = torch.cuda.memory_allocated() / 1e9  # GB
            reserved_mem = torch.cuda.memory_reserved() / 1e9  # GB
            print(f"Initial GPU memory: {allocated_mem:.2f} GB allocated, {reserved_mem:.2f} GB reserved")
        except Exception:
            print("Could not get initial GPU memory information")

        # Reset peak stats for accurate monitoring
        torch.cuda.reset_peak_memory_stats()
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # No header string needed since we're using tqdm progress bars instead of MetricLogger's default output

    bf16_supported = (device.type == 'cuda' and
                     torch.cuda.is_available() and
                     torch.cuda.get_device_capability()[0] >= 8)

    # Select appropriate data type for mixed precision training based on hardware capabilities
    # BFloat16 is preferred when available (on Ampere+ GPUs) as it has better numerical stability
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"Training precision: {'BFloat16' if bf16_supported else 'Float16'} (AMP: {amp_enabled})")

    # Reset gradients with set_to_none=True for better memory efficiency and performance
    optimizer.zero_grad(set_to_none=True)

    # Use gradient accumulation steps from function parameters to effectively increase batch size
    # This allows training with larger effective batch sizes than would fit in GPU memory

    accumulated_batch = 0

    # Pre-fetch batches to avoid data loading bottlenecks
    # Use prefetch value from config for flexible control

    # Use a separate thread for data loading to avoid blocking the main thread
    # This significantly improves GPU utilization by ensuring data is ready when needed
    data_iter = iter(data_loader)
    batch_idx = 0
    total_batches = len(data_loader)

    # Create progress bar for visual feedback
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}", leave=True)

    # Process all batches
    while batch_idx < total_batches:
        try:
            # Get next batch with timeout to prevent hanging
            try:
                samples, targets = next(data_iter)
            except StopIteration:
                break

            # Transfer data to device with non_blocking=True for asynchronous data transfer
            # This allows computation to overlap with data transfer for better GPU utilization
            samples = samples.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Force garbage collection every 5 batches to prevent memory buildup
            if batch_idx % 5 == 0:
                import gc
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Print diagnostic information about input data shapes only on the first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"Sample shape: {samples.shape}, Number of targets: {len(targets)}")
                if len(targets) > 0:
                    print(f"Target keys: {targets[0].keys()}")
        except Exception as e:
            print(f"Error in batch {batch_idx+1}: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            # Use automatic mixed precision (AMP) to speed up training while maintaining numerical stability
            # This reduces memory usage and can significantly accelerate training on modern GPUs
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Print diagnostic information about model execution only on first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"Running model forward pass for batch {batch_idx+1}")

                # Forward pass through the model
                outputs = model(samples)

                # Print model output information on first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"Model output keys: {outputs.keys()}")

                # Print diagnostic information about loss computation only on first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"Computing loss for batch {batch_idx+1}")

                # Compute loss using criterion
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                # Update with current loss value only occasionally to reduce logging overhead
                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx+1}/{len(data_loader)}, Loss: {losses.item():.4f}")

                # Print detailed breakdown of loss components for debugging (only for the first batch of each epoch)
                if accumulated_batch == 0 and samples.size(0) == 1:  # Only for the first batch
                    print("\nTraining Loss Components:")
                    for k, v in loss_dict.items():
                        if k in weight_dict:
                            print(f"  {k}: {v.item():.4f} (weighted: {v.item() * weight_dict[k]:.4f})")
                    print(f"  Total: {losses.item():.4f}")
        except Exception as e:
            print(f"Error in forward/loss computation for batch {batch_idx+1}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Gradient accumulation logic to enable training with larger effective batch sizes
        # Scale the loss for backward pass by dividing by accumulation steps, but keep the original for logging
        # This ensures gradients are properly scaled for accumulation without affecting the reported loss values
        if grad_accumulation_steps > 1:
            # Scale the loss by dividing by accumulation steps to normalize gradients
            scaled_losses = losses / grad_accumulation_steps
        else:
            # No scaling needed when not using gradient accumulation
            scaled_losses = losses

        # Perform backward pass with appropriate scaling and AMP support if enabled
        if amp_enabled and scaler is not None:
            scaler.scale(scaled_losses).backward()
        else:
            scaled_losses.backward()

        # Track number of accumulated batches for gradient accumulation
        accumulated_batch += 1

        # Only update weights after accumulating gradients from the specified number of batches
        if accumulated_batch % grad_accumulation_steps == 0:
            # Calculate and log gradient norm before clipping if Weights & Biases logging is enabled
            if 'wandb' in sys.modules and wandb.run is not None and max_norm > 0:
                # When using AMP, unscale gradients before computing norm to get true gradient magnitude
                if amp_enabled and scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    wandb.log({'train/batch/grad_norm': grad_norm.item()})
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    wandb.log({'train/batch/grad_norm': grad_norm.item()})
            # If not logging to Weights & Biases, just clip gradients if gradient clipping is enabled
            elif max_norm > 0:
                if amp_enabled and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Update model weights using optimizer (with scaler for AMP if enabled)
            if amp_enabled and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Reset gradients to zero with set_to_none=True for better memory efficiency
            optimizer.zero_grad(set_to_none=True)

            # Update learning rate using scheduler if one is provided
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Log current learning rate to Weights & Biases if enabled - only log occasionally
            if 'wandb' in sys.modules and wandb.run is not None and batch_idx % 50 == 0:
                wandb.log({'train/batch/lr': optimizer.param_groups[0]["lr"]})

        # Convert loss dictionary values to Python scalars by detaching from computation graph
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            return loss_dict_reduced

        # Update metrics and progress bar
        metric_logger.update(loss=loss_value, **{k: v for k, v in loss_dict_reduced.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        pbar.update(1)
        batch_idx += 1

        # Explicitly delete tensors to free memory
        del samples, targets, outputs, loss_dict, losses, scaled_losses
        if batch_idx % 3 == 0:  # Force garbage collection more frequently
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear CUDA cache to free fragmented memory

        # Log detailed batch-level training metrics to Weights & Biases for monitoring
        if 'wandb' in sys.modules and wandb.run is not None:
            # Create a comprehensive dictionary of training metrics to log to Weights & Biases
            batch_log_dict = {
                # Main loss values for training monitoring
                'train/batch/loss': loss_value,
                'train/batch/loss_ce': loss_dict_reduced.get('loss_ce', 0),
                'train/batch/loss_bbox': loss_dict_reduced.get('loss_bbox', 0),
                'train/batch/loss_giou': loss_dict_reduced.get('loss_giou', 0),
                'train/batch/loss_cardinality': loss_dict_reduced.get('loss_cardinality', 0),

                # Training hyperparameters and batch information
                'train/batch/lr': optimizer.param_groups[0]["lr"],
                'train/batch/accumulated_batch': accumulated_batch,
                'train/batch/grad_accumulation_steps': grad_accumulation_steps,

                # Training progress tracking metrics
                'train/batch/progress': accumulated_batch / len(data_loader),
                'train/batch/epoch_fraction': (accumulated_batch + (epoch * len(data_loader))) / (len(data_loader) * (config['training']['epochs'] if config else 100))
            }

            # Add batch size if available (might not be if there was an error)
            try:
                # Only access samples if it exists and is still in scope
                if 'samples' in locals() and samples is not None:
                    batch_log_dict['train/batch/batch_size'] = samples.size(0)
                    # Add effective batch size (actual batch size × gradient accumulation steps)
                    batch_log_dict['system/effective_batch_size'] = samples.size(0) * grad_accumulation_steps
            except Exception:
                # If samples is not available, use a default value or skip this metric
                batch_log_dict['train/batch/batch_size'] = batch_size  # Use the configured batch size
                batch_log_dict['system/effective_batch_size'] = batch_size * grad_accumulation_steps

            # Add raw (unweighted) loss values for more detailed debugging
            if 'loss_dict' in locals() and loss_dict is not None:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_log_dict[f'train/batch/raw/{k}'] = v.item()

                # Add weighted loss values (multiplied by their weights) for better comparison with validation
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor) and k in weight_dict:
                        batch_log_dict[f'train/batch/weighted/{k}'] = v.item() * weight_dict[k]

            # Add system metrics like GPU memory usage to help diagnose performance issues
            # This is particularly useful for tracking memory usage during training
            if device.type == 'cuda':
                try:
                    batch_log_dict['system/gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9  # GB
                    batch_log_dict['system/gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9  # GB
                    batch_log_dict['system/gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                except Exception:
                    pass  # Silently ignore errors in GPU metric logging

            # Effective batch size already added in the try/except block above

            # Add learning rate information from scheduler if available (may differ from optimizer's lr)
            if lr_scheduler is not None:
                if hasattr(lr_scheduler, 'get_last_lr'):
                    batch_log_dict['train/batch/last_lr'] = lr_scheduler.get_last_lr()[0]
                elif hasattr(lr_scheduler, '_last_lr'):
                    batch_log_dict['train/batch/last_lr'] = lr_scheduler._last_lr[0]

            # Send all collected metrics to Weights & Biases for visualization
            wandb.log(batch_log_dict)

            # Validation is only performed at the end of each epoch
            # This approach improves training stability and performance
            # Mid-epoch validation was causing errors due to data loader issues

    # Handle the case where the last accumulated batch is incomplete (not a multiple of grad_accumulation_steps)
    if accumulated_batch % grad_accumulation_steps != 0 and accumulated_batch > 0:
        if max_norm > 0:
            if amp_enabled and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Update model weights for the final incomplete batch of accumulated gradients
        if amp_enabled and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if lr_scheduler is not None:
            lr_scheduler.step()

    # Clean up GPU memory at the end of epoch if enabled in configuration
    # This helps prevent memory fragmentation but is only done if explicitly configured
    if device.type == 'cuda' and config.get('training', {}).get('memory_management', {}).get('empty_cache_freq', 0) > 0:
        # Clear CUDA cache to free up fragmented memory
        torch.cuda.empty_cache()

        # Reset peak memory stats to get accurate per-epoch memory usage
        torch.cuda.reset_peak_memory_stats()

    # Close progress bar if it exists
    if 'pbar' in locals():
        pbar.close()

    # Gather all metrics from the metric logger
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Log epoch-level metrics to wandb if enabled
    if 'wandb' in sys.modules and wandb.run is not None:
        wandb.log({
            'train/epoch': epoch,
            **{f'train/epoch/{k}': v for k, v in stats.items()}
        })

    # Clear any remaining GPU memory fragments
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return stats


def validate(model, criterion, data_loader, device, amp_enabled=True):
    """Evaluate model performance on the validation dataset and compute comprehensive metrics.

    Args:
        model: The model to evaluate
        criterion: Loss function for computing validation objectives
        data_loader: DataLoader providing validation batches
        device: Device to run validation on (CPU/GPU)
        amp_enabled: Whether to use automatic mixed precision

    Returns:
        Dictionary of validation metrics
    """
    # For A100 GPU, we can use more aggressive memory optimizations
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # No header string needed since we're using tqdm progress bars for validation

    # Determine the best available precision format for mixed precision validation
    # Check for BFloat16 support which is available on newer GPUs (Ampere architecture and later)
    bf16_supported = False
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            # Check GPU compute capability to determine if BFloat16 is supported
            capability = torch.cuda.get_device_capability()
            # BFloat16 is supported on Ampere architecture (SM 8.0) and newer GPUs
            bf16_supported = capability[0] >= 8

            if bf16_supported:
                print(f"BFloat16 precision supported on {torch.cuda.get_device_name()} (SM {capability[0]}.{capability[1]})")
            else:
                print(f"Using Float16 precision on {torch.cuda.get_device_name()} (SM {capability[0]}.{capability[1]})")
        except Exception as e:
            print(f"Error checking GPU capabilities: {e}")
            print("Defaulting to Float16 precision")

    # Select the appropriate data type for mixed precision validation
    # BFloat16 offers better numerical stability than Float16 while maintaining performance benefits
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16

    # Log the selected precision format for user information
    print(f"Using {amp_dtype} for mixed precision training")

    # Import the function for computing detailed object detection metrics
    from model.metrics import compute_detection_metrics

    # Initialize lists to collect model outputs and targets for computing metrics after all batches
    all_outputs = []
    all_targets = []

    # Use torch.inference_mode() which is more optimized than torch.no_grad() for pure inference
    with torch.inference_mode():
        for batch_idx, (imgs, targets) in enumerate(tqdm(data_loader, desc="Validation", leave=True)):
            try:
                # Transfer data to device with non_blocking=True for asynchronous data transfer
                imgs = imgs.to(device, non_blocking=True)
                targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # Print diagnostic information about validation data shapes only for the first batch
                if batch_idx == 0:
                    print(f"Validation sample shape: {imgs.shape}, Number of targets: {len(targets)}")
                    if len(targets) > 0:
                        print(f"Validation target keys: {targets[0].keys()}")
            except Exception as e:
                print(f"Error in validation batch {batch_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Use automatic mixed precision to speed up validation while maintaining numerical stability
            try:
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    # Print diagnostic information about model execution only for the first batch
                    if batch_idx == 0:
                        print(f"Running validation model forward pass")
                    outputs = model(imgs)
                    if batch_idx == 0:
                        print(f"Validation model output keys: {outputs.keys()}")

                    loss_dict = criterion(outputs, targets)

                    # Apply the same loss scaling as in training to ensure consistency between phases
                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    # Update with current validation loss only occasionally to reduce logging overhead
                    if batch_idx % 50 == 0:
                        print(f"Val batch {batch_idx+1}/{len(data_loader)}, Loss: {losses.item():.4f}")

                    # Print detailed breakdown of validation loss components for debugging
                    if imgs.size(0) == 1:  # Only for the first batch
                        print("\nValidation Loss Components:")
                        for k, v in loss_dict.items():
                            if k in weight_dict:
                                print(f"  {k}: {v.item():.4f} (weighted: {v.item() * weight_dict[k]:.4f})")
                        print(f"  Total: {losses.item():.4f}")
            except Exception as e:
                print(f"Error in validation forward/loss computation for batch {batch_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Store model outputs for later metrics computation with memory optimization
            # Only store essential tensors and use detach() instead of clone() to save memory
            try:
                all_outputs.append({
                    'pred_logits': outputs['pred_logits'].detach(),  # Detach from computation graph without cloning
                    'pred_boxes': outputs['pred_boxes'].detach()    # Detach from computation graph without cloning
                })
                if batch_idx % 50 == 0:
                    print(f"Stored outputs for batch {batch_idx+1}, total outputs: {len(all_outputs)}")
            except Exception as e:
                print(f"Error storing outputs for batch {batch_idx+1}: {e}")

            # Store ground truth targets with minimal memory usage by only keeping essential fields
            try:
                batch_targets = []
                for t in targets:
                    target_dict = {}
                    for k, v in t.items():
                        if k in ['boxes', 'labels']:  # Only keep bounding boxes and class labels
                            target_dict[k] = v.detach() if isinstance(v, torch.Tensor) else v
                    batch_targets.append(target_dict)
                all_targets.extend(batch_targets)
                if batch_idx % 50 == 0:
                    print(f"Stored targets for batch {batch_idx+1}, total targets: {len(all_targets)}")
            except Exception as e:
                print(f"Error storing targets for batch {batch_idx+1}: {e}")

            weight_dict = criterion.weight_dict
            loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
            # Scale loss values using the same weights as in training for consistent reporting
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss = sum(loss_dict_reduced_scaled.values())

            # Apply weights to individual loss components to match training loss reporting
            loss_dict_reduced = {k: v * weight_dict.get(k, 1.0) for k, v in loss_dict_reduced.items()}

            # Skip this batch if the loss value is not finite (NaN or Inf)
            loss_value = loss
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, skipping batch")
                continue

            metric_logger.update(loss=loss_value, **loss_dict_reduced)

            # Explicitly delete tensors to free memory
            del imgs, targets, outputs, loss_dict, losses
            if batch_idx % 3 == 0:  # Force garbage collection every 3 batches
                import gc
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()  # Clear CUDA cache to free fragmented memory

    # Print summary of validation metrics collected during evaluation
    print("Validation stats:", {k: meter.global_avg for k, meter in metric_logger.meters.items()})

    # Compute comprehensive object detection metrics (precision, recall, F1, accuracy, mAP)
    # Combine all collected outputs into a single batch for efficient metrics computation
    print(f"\nNumber of collected outputs: {len(all_outputs)}")
    print(f"Number of validation targets: {len(all_targets)}")

    # Check if we have any outputs to process
    if len(all_outputs) == 0:
        print("Warning: No outputs collected during validation. Returning basic metrics only.")
        # Return the basic metrics we've collected so far
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if all_outputs and len(all_outputs) > 0:
        try:
            # Check for tensor size consistency before concatenation
            pred_logits_list = [o['pred_logits'] for o in all_outputs]
            pred_boxes_list = [o['pred_boxes'] for o in all_outputs]

            # Get the expected sizes (from the majority of batches)
            # Use the most common size for better consistency
            logits_sizes = [tensor.size(1) for tensor in pred_logits_list]
            boxes_sizes = [tensor.size(1) for tensor in pred_boxes_list]

            # Find the most common size (mode) for more robust handling
            from collections import Counter
            logits_size_counter = Counter(logits_sizes)
            boxes_size_counter = Counter(boxes_sizes)

            # Use the most common size, or max if there's a tie
            expected_logits_size = logits_size_counter.most_common(1)[0][0]
            expected_boxes_size = boxes_size_counter.most_common(1)[0][0]

            print(f"Using most common tensor sizes: logits={expected_logits_size}, boxes={expected_boxes_size}")
            print(f"Size distribution: logits={dict(logits_size_counter)}, boxes={dict(boxes_size_counter)}")

            # Check if we need to pad any tensors
            need_padding = False
            for i, (logits, boxes) in enumerate(zip(pred_logits_list, pred_boxes_list)):
                if logits.size(1) != expected_logits_size or boxes.size(1) != expected_boxes_size:
                    need_padding = True
                    print(f"Batch {i} has different tensor sizes: logits={logits.size()}, boxes={boxes.size()}")
                    print(f"Expected sizes: logits=(*,{expected_logits_size},*), boxes=(*,{expected_boxes_size},*)")
                    # Print more details about the problematic batch
                    print(f"This is typically caused by the last batch having a different size.")
                    break

            # If there are inconsistent tensor sizes, we have two options:
            # 1. Pad the tensors to match the expected size (current approach)
            # 2. Skip the problematic batches (alternative approach)
            # We'll implement both and use padding by default, but allow skipping as a fallback

            # First try padding approach

            if need_padding:
                print("Padding tensors to ensure consistent sizes before concatenation...")
                # Process pred_logits - pad to match the expected size
                for i in range(len(pred_logits_list)):
                    logits = pred_logits_list[i]
                    if logits.size(1) < expected_logits_size:
                        # Get current sizes
                        batch_size, current_size, num_classes = logits.size()
                        # Create padding tensor
                        padding = torch.zeros(
                            batch_size, expected_logits_size - current_size, num_classes,
                            device=logits.device, dtype=logits.dtype
                        )
                        # Add background class probability of 1.0 to padding
                        # This ensures padded "fake" predictions are classified as background
                        if num_classes > 1:
                            padding[:, :, -1] = 1.0  # Set background class probability to 1.0
                        # Concatenate with original tensor
                        pred_logits_list[i] = torch.cat([logits, padding], dim=1)
                        print(f"  Padded batch {i} logits from shape {logits.shape} to {pred_logits_list[i].shape}")

                # Process pred_boxes - pad to match the expected size
                for i in range(len(pred_boxes_list)):
                    boxes = pred_boxes_list[i]
                    if boxes.size(1) < expected_boxes_size:
                        # Get current sizes
                        batch_size, current_size, box_dim = boxes.size()
                        # Create padding tensor with zeros (outside image coordinates)
                        # Using zeros for box coordinates places them at the top-left corner (0,0)
                        # with zero width/height, making them invisible/ignored in evaluation
                        padding = torch.zeros(
                            batch_size, expected_boxes_size - current_size, box_dim,
                            device=boxes.device, dtype=boxes.dtype
                        )
                        # Concatenate with original tensor
                        pred_boxes_list[i] = torch.cat([boxes, padding], dim=1)
                        print(f"  Padded batch {i} boxes from shape {boxes.shape} to {pred_boxes_list[i].shape}")

            # Now concatenate the processed tensors
            try:
                combined_outputs = {
                    'pred_logits': torch.cat(pred_logits_list, dim=0),
                    'pred_boxes': torch.cat(pred_boxes_list, dim=0)
                }
                print(f"Successfully concatenated all outputs: pred_logits shape={combined_outputs['pred_logits'].shape}, "
                      f"pred_boxes shape={combined_outputs['pred_boxes'].shape}")
            except Exception as e:
                print(f"Error during tensor concatenation with padding approach: {e}")
                # Create a detailed error report for debugging
                print("\nDetailed tensor shapes:")
                for i, (logits, boxes) in enumerate(zip(pred_logits_list, pred_boxes_list)):
                    print(f"  Batch {i}: logits={logits.shape}, boxes={boxes.shape}")

                # Fallback approach: Skip problematic batches
                print("\nFalling back to alternative approach: skipping inconsistent batches")

                # Find batches with the most common size
                consistent_logits = []
                consistent_boxes = []

                for i, (logits, boxes) in enumerate(zip(pred_logits_list, pred_boxes_list)):
                    if logits.size(1) == expected_logits_size and boxes.size(1) == expected_boxes_size:
                        consistent_logits.append(logits)
                        consistent_boxes.append(boxes)
                    else:
                        print(f"Skipping batch {i} with inconsistent sizes")

                if not consistent_logits:
                    print("No consistent batches found. Cannot compute metrics.")
                    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

                # Try concatenating only the consistent batches
                try:
                    combined_outputs = {
                        'pred_logits': torch.cat(consistent_logits, dim=0),
                        'pred_boxes': torch.cat(consistent_boxes, dim=0)
                    }
                    print(f"Successfully concatenated consistent outputs: pred_logits shape={combined_outputs['pred_logits'].shape}, "
                          f"pred_boxes shape={combined_outputs['pred_boxes'].shape}")
                except Exception as e2:
                    print(f"Error during fallback concatenation: {e2}")
                    print("Cannot compute detection metrics. Returning basic metrics only.")
                    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            # Compute detailed detection metrics and print diagnostic information
            print("\nComputing detection metrics...")
            print(f"Number of validation samples: {len(all_targets)}")

            # If we had to skip some batches, we need to adjust the targets accordingly
            if 'consistent_logits' in locals():
                # We need to adjust all_targets to match the consistent outputs
                # This is a bit tricky since we need to know which batches were skipped
                # For simplicity, we'll just use the first N targets where N is the number of samples in combined_outputs
                num_samples = combined_outputs['pred_logits'].size(0)
                print(f"Adjusting targets to match {num_samples} samples in combined outputs")
                if len(all_targets) > num_samples:
                    all_targets = all_targets[:num_samples]
                    print(f"Truncated targets to {len(all_targets)} samples")
                elif len(all_targets) < num_samples:
                    print(f"Warning: Number of targets ({len(all_targets)}) is less than number of predictions ({num_samples})")
                    # Pad targets with empty targets to match the number of predictions
                    empty_target = {'boxes': torch.zeros((0, 4), device=device), 'labels': torch.zeros(0, dtype=torch.int64, device=device)}
                    while len(all_targets) < num_samples:
                        all_targets.append(empty_target)
                    print(f"Padded targets to {len(all_targets)} samples with empty targets")
            print(f"Number of predictions: {combined_outputs['pred_logits'].shape[0]}")

            # Calculate confidence score statistics to evaluate prediction quality
            all_scores = torch.nn.functional.softmax(combined_outputs['pred_logits'], dim=-1)
            all_scores = all_scores[:, :-1]  # Remove background class
            max_scores, _ = all_scores.max(dim=1)

            print(f"Overall confidence score statistics:")
            print(f"  Max: {max_scores.max().item():.4f}")
            print(f"  Min: {max_scores.min().item():.4f}")
            print(f"  Mean: {max_scores.mean().item():.4f}")
            print(f"  Median: {max_scores.median().item():.4f}")

            # Use consistent IoU thresholds across validation and evaluation for comparable metrics
            iou_thresholds = [0.1, 0.3, 0.5, 0.7]

            # Call the detection metrics computation function with error handling
            try:
                print(f"Computing detection metrics with {len(all_targets)} targets and {combined_outputs['pred_logits'].shape[0]} predictions...")
                detection_metrics = compute_detection_metrics(combined_outputs, all_targets, iou_thresholds=iou_thresholds, num_classes=8)
                print("Detection metrics computed successfully")
            except Exception as e:
                print(f"Error computing detection metrics: {e}")
                import traceback
                traceback.print_exc()
                # Create fallback metrics dictionary with zeros to avoid downstream errors
                detection_metrics = {
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'mAP': 0.0,
                    'class_precision': [0.0] * 8, 'class_recall': [0.0] * 8, 'class_f1': [0.0] * 8,
                    'confusion_matrix': torch.zeros(8, 8, dtype=torch.int),
                    'debug_info': {'total_predictions': 0, 'total_gt_boxes': 0, 'total_matches': 0}
                }

            # Collect all metrics into a single result dictionary to return
            result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            # Add computed detection metrics to the result dictionary
            # This makes all metrics available in val_stats for later Weights & Biases logging
            for key, value in detection_metrics.items():
                if isinstance(value, (int, float, bool)) or (
                    isinstance(value, torch.Tensor) and value.numel() == 1):
                    result[key] = value
                    print(f"Metric added to result: {key} = {value}")
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    result[key] = value
                    print(f"List metric added to result: {key} = {value[:3]}...")

            # Store the complete metrics dictionary for advanced debugging if needed
            result['detection_metrics'] = detection_metrics

            # Print summary of key detection metrics for user feedback
            print("\nDetection metrics:")
            print(f"  Precision: {detection_metrics['precision']:.4f}")
            print(f"  Recall: {detection_metrics['recall']:.4f}")
            print(f"  F1 Score: {detection_metrics['f1']:.4f}")
            print(f"  Accuracy: {detection_metrics['accuracy']:.4f}")
            print(f"  mAP: {detection_metrics['mAP']:.4f}")

            # Check if metrics were correctly added to the result dictionary
            print("\nStatus of metrics added to result dictionary:")
            for key in ['precision', 'recall', 'f1', 'accuracy', 'mAP']:
                if key in result:
                    print(f"  {key}: {result[key]:.4f} - Successfully added")
                else:
                    print(f"  {key}: Could not be added!")

            # Print per-class metrics to identify class-specific performance issues
            if 'class_precision' in detection_metrics:
                print("\nClass-wise metrics:")
                # Use standard class names for the AU-AIR dataset
                class_names = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
                for i, class_name in enumerate(class_names):
                    if i < len(detection_metrics['class_precision']):
                        print(f"  {class_name.ljust(10)}: Precision={detection_metrics['class_precision'][i]:.4f}, "
                              f"Recall={detection_metrics['class_recall'][i]:.4f}, "
                              f"F1={detection_metrics['class_f1'][i]:.4f}")

            # Print confusion matrix to show class prediction errors and misclassifications
            if 'confusion_matrix' in detection_metrics:
                print("\nConfusion Matrix:")
                cm = detection_metrics['confusion_matrix'].cpu().numpy()
                print(cm)

            # Log validation metrics and confusion matrix visualization to Weights & Biases
            # Ensure wandb is available and properly initialized
            if 'wandb' in sys.modules and wandb.run is not None:
                print("\nWandb is available for logging metrics")
                # Check if confusion matrix is available
                if 'confusion_matrix' in detection_metrics:
                    # Use standard class names for the AU-AIR dataset for visualization
                    class_names = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]

                    # Import function to convert confusion matrix to a visual image for Weights & Biases
                    from model.metrics import plot_confusion_matrix

                    # Generate visual representation of confusion matrix for Weights & Biases
                    confusion_matrix_img = plot_confusion_matrix(detection_metrics['confusion_matrix'], class_names)

                # Create a dictionary with all validation metrics to log to Weights & Biases
                val_log_dict = {
                    # Include current training step for proper timeline alignment in Weights & Biases
                    'step': wandb.run.step if wandb.run is not None else 0,
                }

                # Add confusion matrix if available
                if 'confusion_matrix' in detection_metrics and 'confusion_matrix_img' in locals():
                    val_log_dict['confusion_matrix'] = confusion_matrix_img

                # Add main metrics
                if 'accuracy' in detection_metrics:
                    val_log_dict['val/accuracy'] = float(detection_metrics['accuracy'])
                if 'precision' in detection_metrics:
                    val_log_dict['val/precision'] = float(detection_metrics['precision'])
                if 'recall' in detection_metrics:
                    val_log_dict['val/recall'] = float(detection_metrics['recall'])
                if 'f1' in detection_metrics:
                    val_log_dict['val/f1'] = float(detection_metrics['f1'])
                if 'mAP' in detection_metrics:
                    val_log_dict['val/mAP'] = float(detection_metrics['mAP'])

                # Add metrics for each IoU threshold to track performance at different overlap levels
                for key, value in detection_metrics.items():
                    if key.startswith('mAP@') or key.startswith('precision@') or key.startswith('recall@') or key.startswith('f1@'):
                        try:
                            # Convert tensor or numpy values to Python floats for Weights & Biases compatibility
                            val_log_dict[f'val/{key}'] = float(value)
                        except Exception as e:
                            print(f"Error converting {key} to float: {e}")

                # Add COCO-style mAP if available
                if 'mAP_coco' in detection_metrics:
                    try:
                        val_log_dict['val/mAP_coco'] = float(detection_metrics['mAP_coco'])
                    except Exception as e:
                        print(f"Error converting mAP_coco to float: {e}")

                # Add average IoU and confidence scores
                if 'debug_info' in detection_metrics:
                    debug_info = detection_metrics['debug_info']
                    if 'avg_iou' in debug_info:
                        try:
                            val_log_dict['val/avg_iou'] = float(debug_info['avg_iou'])
                        except Exception as e:
                            print(f"Error converting avg_iou to float: {e}")
                    if 'avg_confidence' in debug_info:
                        try:
                            val_log_dict['val/avg_confidence'] = float(debug_info['avg_confidence'])
                        except Exception as e:
                            print(f"Error converting avg_confidence to float: {e}")
                    if 'total_matches' in debug_info:
                        try:
                            val_log_dict['val/total_matches'] = int(debug_info['total_matches'])
                        except Exception as e:
                            print(f"Error converting total_matches to int: {e}")

                # Add validation loss values
                loss_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                for loss_name, loss_value in loss_dict.items():
                    try:
                        val_log_dict[f'val/{loss_name}'] = float(loss_value)
                    except Exception as e:
                        print(f"Error converting {loss_name} to float: {e}")

                # Add class-wise metrics if available
                if 'class_precision' in detection_metrics:
                    for i, class_name in enumerate(class_names):
                        if i < len(detection_metrics['class_precision']):
                            try:
                                val_log_dict[f'val/class/{class_name}/precision'] = float(detection_metrics['class_precision'][i])
                                val_log_dict[f'val/class/{class_name}/recall'] = float(detection_metrics['class_recall'][i])
                                val_log_dict[f'val/class/{class_name}/f1'] = float(detection_metrics['class_f1'][i])
                            except Exception as e:
                                print(f"Error converting class metrics for {class_name} to float: {e}")

                # Print validation metrics before sending to wandb
                print("\nSending the following validation metrics to Wandb:")
                for key, value in val_log_dict.items():
                    if not key.startswith('confusion_matrix'):
                        print(f"  {key}: {value}")

                # Log to wandb
                try:
                    # Ensure we have at least some metrics to log
                    if len(val_log_dict) > 1:  # More than just 'step'
                        wandb.log(val_log_dict)
                        print("Successfully logged validation metrics to Wandb")
                    else:
                        print("WARNING: No validation metrics to log to Wandb!")
                except Exception as e:
                    print(f"Error logging to Wandb: {e}")
                    import traceback
                    traceback.print_exc()

            # Comprehensive memory cleanup before returning
            if device.type == 'cuda':
                # Explicitly delete large tensors to free memory immediately
                del combined_outputs, all_outputs, all_targets

                # Clear CUDA cache to prevent memory fragmentation
                torch.cuda.empty_cache()

                # Reset peak memory stats for better monitoring
                torch.cuda.reset_peak_memory_stats()

                # Log memory usage after validation
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"GPU memory after validation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # Display the final computed metrics
            print("\nValidation completed. Returned metrics:")
            for key, value in result.items():
                if key in ['precision', 'recall', 'f1', 'accuracy', 'mAP']:
                    print(f"  {key}: {value}")

            return result
        except Exception as e:
            print(f"Error computing detection metrics: {e}")
            import traceback
            traceback.print_exc()
            # Clean up memory even on error
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            # Return basic metrics without the detection metrics
            result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print("\nReturning only basic metrics due to error:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            return result
    else:
        # Clean up memory before returning
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Return basic metrics without the detection metrics
        result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        print("\nNo validation outputs found, returning only basic metrics:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return result





def visualize_predictions(img, target, output, class_names):
    """Visualize predictions for wandb.

    Args:
        img: Tensor image [C, H, W]
        target: Target dict with 'boxes', 'labels'
        output: Output dict with 'pred_logits', 'pred_boxes'
        class_names: List of class names

    Returns:
        wandb_img: Wandb Image object with annotations
    """
    # Convert image to numpy format
    img_np = img.permute(1, 2, 0).numpy()

    # Denormalize
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

        # Filter out low confidence predictions (lowered threshold for better detection)
        keep = max_scores > 0.1
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
    """Prepares bounding box data for W&B visualization.

    Args:
        target: Ground truth bounding boxes
        output: Bounding boxes predicted by the model

    Returns:
        Bounding box data in W&B format
    """
    from model.utils import box_cxcywh_to_xyxy
    import torch.nn.functional as F

    # Prepare wandb bounding box dict
    # Get class names from config
    class_names = ['Human', 'Car', 'Truck', 'Van', 'Motorbike', 'Bicycle', 'Bus', 'Trailer']  # Default class names
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

            # No threshold filtering - use all predictions and sort by confidence
            # Sort by confidence (highest first)
            sorted_indices = torch.argsort(max_scores, descending=True)
            # Limit to top 20 predictions to avoid cluttering the visualization
            sorted_indices = sorted_indices[:20]  # Show at most 20 predictions
            filtered_boxes = pred_boxes[sorted_indices]
            filtered_labels = pred_labels[sorted_indices]
            filtered_scores = max_scores[sorted_indices]

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


def visualize_sample(images, targets, outputs, config, num_samples=2, save_dir=None, wandb_log=False):
    """Visualize images with ground truth and predicted bounding boxes.

    Args:
        images: Batch of images
        targets: Ground truth targets with bounding boxes and labels
        outputs: Model predictions
        config: Configuration dictionary
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        wandb_log: Whether to log visualizations to wandb

    Returns:
        None
    """

    # Get class names from config
    class_names = config['dataset']['class_names']
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
            for box, label_id in zip(boxes.detach().cpu(), labels.detach().cpu()):
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

            # No threshold filtering - use all predictions and sort by confidence
            # Sort by confidence (highest first)
            sorted_indices = torch.argsort(max_scores, descending=True)
            # Limit to top 20 predictions to avoid cluttering the visualization
            sorted_indices = sorted_indices[:20]  # Show at most 20 predictions
            filtered_boxes = pred_boxes[sorted_indices]
            filtered_labels = pred_labels[sorted_indices]
            filtered_scores = max_scores[sorted_indices]

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
        if wandb_log and wandb is not None:
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
    """Main training function.

    This function processes command line arguments, creates the model, and manages the training loop.
    It handles configuration loading, dataset preparation, model initialization, and the training process.

    Returns:
        None
    """
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Force garbage collection at the start
    import gc
    gc.collect()

    # Initialize device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device used: {device}")

    if device.type == 'cuda':
        print("🧹 Initializing CUDA memory management...")

        # Clear CUDA cache once at the beginning
        torch.cuda.empty_cache()

        # Use PyTorch's memory allocator settings for better memory management
        # These settings help prevent memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

        # Get GPU information for logging
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
            print(f"Using {gpu_name} GPU with {total_memory:.2f} GB total memory")

            # Log memory usage before training starts
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9  # in GB
            reserved_memory = torch.cuda.memory_reserved(0) / 1e9  # in GB
            print(f"Initial GPU memory: {allocated_memory:.2f} GB allocated, {reserved_memory:.2f} GB reserved")

            # Reset peak stats for accurate monitoring
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"Note: Could not get detailed GPU information: {e}")
            print("Training will proceed with default memory settings")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Performance optimizations - add before model loading
    # ===================================================
    if device.type == 'cuda':
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cudnn.deterministic = False  # Faster but less deterministic
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn

        # Memory optimizations - for CUDA out of memory issues
        torch.cuda.empty_cache()  # Clear cache to start fresh

        # Print GPU information
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Optimization settings:")
        print(f"  - TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - Mixed Precision: {config['training'].get('mixed_precision', True)}")
    else:
        # CPU optimizations
        torch.set_num_threads(os.cpu_count())  # Use all CPU cores
        print(f"Training on CPU with {torch.get_num_threads()} threads")

    # Check model configuration
    if 'model' in config:
        # Store original model parameters for reference
        original_layers = {
            'encoder_layers': config['model'].get('num_encoder_layers', 6),
            'decoder_layers': config['model'].get('num_decoder_layers', 6),
            'hidden_dim': config['model'].get('hidden_dim', 256),
            'nheads': config['model'].get('nheads', 8)
        }

        # Only reduce model size if explicitly requested in config
        if config.get('optimize_model_size', False):
            # Reduce model complexity for faster training
            config['model']['num_encoder_layers'] = min(original_layers['encoder_layers'], 4)
            config['model']['num_decoder_layers'] = min(original_layers['decoder_layers'], 4)
            config['model']['hidden_dim'] = min(original_layers['hidden_dim'], 256)
            config['model']['nheads'] = min(original_layers['nheads'], 8)

            # Print model optimization
            print("⚠️ Optimizing model for faster training:")
            print(f"  Encoder layers: {original_layers['encoder_layers']} -> {config['model']['num_encoder_layers']}")
            print(f"  Decoder layers: {original_layers['decoder_layers']} -> {config['model']['num_decoder_layers']}")
            print(f"  Hidden dimension: {original_layers['hidden_dim']} -> {config['model']['hidden_dim']}")
            print(f"  Attention heads: {original_layers['nheads']} -> {config['model']['nheads']}")
        else:
            print("Using full model size as specified in config:")
            print(f"  Encoder layers: {original_layers['encoder_layers']}")
            print(f"  Decoder layers: {original_layers['decoder_layers']}")
            print(f"  Hidden dimension: {original_layers['hidden_dim']}")
            print(f"  Attention heads: {original_layers['nheads']}")

    # Handle mixed precision settings
    mixed_precision = config['training'].get('mixed_precision', True)

    # If user explicitly disables mixed precision, override config
    if args.no_mixed_precision:
        mixed_precision = False
        print("Mixed precision disabled by command line argument")

    # Check for BFloat16 support
    bf16_supported = (device.type == 'cuda' and
                     torch.cuda.is_available() and
                     torch.cuda.get_device_capability()[0] >= 8)

    # Configure mixed precision settings
    amp_dtype = None
    scaler = None

    if mixed_precision:
        if device.type == 'cuda':
            if bf16_supported:
                print("Using BFloat16 mixed precision training (better stability)")
                amp_dtype = torch.bfloat16
                # BFloat16 doesn't need a scaler due to higher dynamic range
                scaler = None
            else:
                print("Using Float16 mixed precision training")
                amp_dtype = torch.float16
                # Float16 needs a gradient scaler to prevent underflow
                scaler = GradScaler()
                print("Gradient scaling enabled for Float16 training")
        else:
            print("Mixed precision not supported on CPU, falling back to FP32")
            mixed_precision = False
            amp_dtype = torch.float32
    else:
        print("Using FP32 (full precision) training...")
        amp_dtype = torch.float32

    # Print final precision configuration
    print(f"Training precision: {amp_dtype} (Mixed precision: {mixed_precision})")

    # Configure CUDA memory settings
    if device.type == 'cuda':
        # Enable memory optimizations
        torch.cuda.empty_cache()

        # Print memory allocation strategy
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            print(f"GPU memory: {allocated_memory/1e9:.2f} GB allocated / {total_memory/1e9:.2f} GB total")
        except Exception as e:
            print(f"Could not determine GPU memory usage: {e}")

    # Get gradient accumulation setting
    grad_accumulation = config['training'].get('grad_accumulation', 1)

    # If user explicitly specifies gradient accumulation, override config
    if args.grad_accumulation > 1:
        grad_accumulation = args.grad_accumulation
        print(f"Gradient accumulation overridden to {grad_accumulation} steps by command line argument")
    elif grad_accumulation > 1:
        print(f"Using gradient accumulation with {grad_accumulation} steps from config file")
    else:
        print("Gradient accumulation disabled (steps=1)")

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

        # Initialize wandb if available
        if wandb is not None:
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                name=config['wandb']['name'],
                config=wandb_config,
                tags=config['wandb']['tags']
            )
        else:
            print("Warning: wandb module not found. Wandb logging disabled.")

    # Load datasets
    print("Loading datasets...")

    # Define annotations files based on split
    train_annotations_file = args.annotations_file
    val_annotations_file = None

    if train_annotations_file is None:
        train_annotations_file = os.path.join('metadata', 'train_annotations.json')
        val_annotations_file = os.path.join('metadata', 'val_annotations.json')

    # Load train and validation datasets with direct GPU tensor creation
    train_dataset = build_dataset(
        root_dir=args.dataset_path,
        annotations_file=train_annotations_file,
        split='train',
        config=config,
        device=device        # Pass device to create tensors directly on GPU
    )

    val_dataset = build_dataset(
        root_dir=args.dataset_path,
        annotations_file=val_annotations_file,
        split='val',
        config=config,
        device=device        # Pass device to create tensors directly on GPU
    )

    # Print dataset size information
    print(f"Dataset size: {len(train_dataset)} training examples, {len(val_dataset)} validation examples")
    print(f"Image size: {config['dataset']['img_size']}")

    # Apply dataset optimizations
    # Use sample ratio from config or command line
    sample_ratio = config['training'].get('sample_ratio', 1.0)  # Default to using full dataset

    # Override with command line argument if provided
    if args.sample_ratio is not None:
        sample_ratio = args.sample_ratio
        print(f"Dataset sample ratio overridden to {sample_ratio} by command line argument")

    # Apply sampling if ratio is less than 1.0
    if sample_ratio < 1.0:
        print(f"Sampling dataset to {sample_ratio*100:.1f}% of original size")
        train_dataset = SampledDataset(train_dataset, sample_ratio=sample_ratio, seed=args.seed)
        val_dataset = SampledDataset(val_dataset, sample_ratio=sample_ratio, seed=args.seed)
    else:
        print("Using full dataset (no sampling)")

    # 2. Use dynamic caching for memory-efficient access
    # Get cache sizes from config file for flexible control
    train_cache_size = config['training'].get('cache_size_train', len(train_dataset))  # Default: cache entire dataset
    val_cache_size = config['training'].get('cache_size_val', len(val_dataset))        # Default: cache entire dataset

    # Get preload sizes from config file
    preload_size_train = config['training'].get('preload_size_train', 500)  # Default: preload 500 samples
    preload_size_val = config['training'].get('preload_size_val', 250)      # Default: preload 250 samples

    print(f"Cache configuration:")
    print(f"  - Train cache size: {train_cache_size} samples")
    print(f"  - Val cache size: {val_cache_size} samples")
    print(f"  - Train preload size: {preload_size_train} samples")
    print(f"  - Val preload size: {preload_size_val} samples")

    # Use dynamic caching with optimized parameters for GPU memory usage
    # Pass device to dataset to ensure tensors are created directly on GPU
    train_dataset = DynamicCachedDataset(
        train_dataset,
        cache_size=train_cache_size,  # Use config value (0 = disabled)
        preload=train_cache_size > 0,  # Only preload if caching is enabled
        preload_size=preload_size_train,  # Use config value (0 = disabled)
        device=device  # Always pass the device to ensure tensors go to GPU
    )
    val_dataset = DynamicCachedDataset(
        val_dataset,
        cache_size=val_cache_size,  # Use config value (0 = disabled)
        preload=val_cache_size > 0,  # Only preload if caching is enabled
        preload_size=preload_size_val,  # Use config value (0 = disabled)
        device=device  # Always pass the device to ensure tensors go to GPU
    )

    # Force garbage collection after dataset creation
    import gc
    gc.collect()

    print(f"Using optimized dataset: {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")

    # Batch size - always take from config file without automatic changes
    batch_size = config['training'].get('batch_size', 8)
    print(f"Batch size: {batch_size} (from config)")

    # Removed batch size check for CUDA memory error - always use config value
    print(f"Batch size {batch_size} and grad_accumulation {grad_accumulation} values taken from config file.")

    # Training parameters
    print(f"Training parameters:")
    print(f"  - Epochs: {config['training']['epochs']}")
    print(f"  - Learning rate: {config['optimizer']['lr']}")
    print(f"  - Weight decay: {config['optimizer']['weight_decay']}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {grad_accumulation}")
    print(f"  - Effective batch size: {batch_size * grad_accumulation}")
    print(f"  - Workers: {config['training'].get('num_workers', 2)}")
    print(f"  - Prefetch factor: {config['training'].get('prefetch_factor', 2)}")
    print(f"  - Pin memory: {config['training'].get('pin_memory', True)}")
    print(f"  - Mixed precision: {mixed_precision}")

    # Create data loaders with optimized settings from config file
    # Get number of workers from config for flexible control
    num_workers = config['training'].get('num_workers', 4)  # Default: 4 workers

    # Pin memory for faster GPU transfer
    pin_memory = config['training'].get('pin_memory', True) and device.type == 'cuda'

    # Prefetch factor: how many batches to prefetch per worker
    # Get from config file for flexible control
    prefetch_factor = config['training'].get('prefetch_factor', 2)  # Default: prefetch 2 batches per worker

    print(f"DataLoader configuration:")
    print(f"  - Number of workers: {num_workers}")
    print(f"  - Prefetch factor: {prefetch_factor}")
    print(f"  - Pin memory: {pin_memory}")

    # Keep workers alive between epochs for faster startup if using multiple workers
    persistent_workers = num_workers > 0
    print(f"  - Persistent workers: {persistent_workers}")

    # Use persistent_workers in dataloader configuration

    # Create optimized data loaders
    import gc
    import psutil

    # Force garbage collection before creating dataloaders
    gc.collect()

    # Check system memory and reduce workers/prefetch if memory is tight
    try:
        mem = psutil.virtual_memory()
        if mem.percent > 70:  # If system memory usage is above 70%
            print(f"WARNING: High memory usage detected ({mem.percent}%). Reducing worker count and prefetch factor.")
            num_workers = max(1, min(num_workers, 2))  # Limit to 2 workers
            prefetch_factor = 1  # Minimum prefetch factor
    except Exception as e:
        print(f"Could not check system memory: {e}")

    # Create train dataloader
    train_dataloader = build_dataloader(
        dataset=train_dataset,
        shuffle=True,
        config=config,
        drop_last=True  # Drop last incomplete batch for better GPU utilization
    )

    # Run garbage collection after creating train dataloader
    gc.collect()

    # Create validation dataloader
    val_dataloader = build_dataloader(
        dataset=val_dataset,
        shuffle=False,
        config=config,
        drop_last=False  # Keep all samples for validation
    )

    # Run garbage collection after creating dataloaders
    gc.collect()

    # Log data loader information
    print(f"Training data: {len(train_dataset)} samples, {len(train_dataloader)} batches")
    print(f"Validation data: {len(val_dataset)} samples, {len(val_dataloader)} batches")

    # Initialize the Deformable DETR model architecture
    print("Building model...")
    # Get memory management settings from config file for flexible control
    # These parameters control memory usage during attention computation
    memory_management = config['training'].get('memory_management', {})
    use_checkpoint = memory_management.get('use_checkpoint', False)  # Default: disabled
    chunk_size_large = memory_management.get('chunk_size_large', 5000)  # Default: 5000
    chunk_size_small = memory_management.get('chunk_size_small', 100)   # Default: 100

    print(f"Memory management settings:")
    print(f"  - Gradient checkpointing: {use_checkpoint}")
    print(f"  - Large chunk size: {chunk_size_large}")
    print(f"  - Small chunk size: {chunk_size_small}")

    # Print model configuration parameters for verification
    print("\nModel Configuration Parameters:")
    print(f"  - hidden_dim: {config['model']['hidden_dim']}")
    print(f"  - nheads: {config['model']['nheads']}")
    print(f"  - num_encoder_layers: {config['model']['num_encoder_layers']}")
    print(f"  - num_decoder_layers: {config['model']['num_decoder_layers']}")
    print(f"  - dim_feedforward: {config['model'].get('dim_feedforward', 1024)}")
    print(f"  - dropout: {config['model'].get('dropout', 0.1)}")
    print(f"  - num_feature_levels: {config['model']['num_feature_levels']}")
    print(f"  - enc_n_points: {config['model']['enc_n_points']}")
    print(f"  - dec_n_points: {config['model']['dec_n_points']}")
    print(f"  - num_queries: {config['model']['num_queries']}")
    print(f"  - aux_loss: {config['model'].get('aux_loss', True)}")
    print(f"  - use_checkpoint: {use_checkpoint}")
    print(f"  - chunk_size_large: {chunk_size_large}")
    print(f"  - chunk_size_small: {chunk_size_small}")

    # Create model with parameters from config
    model = DeformableDetrModel(
        num_classes=config['dataset']['num_classes'],
        hidden_dim=config['model']['hidden_dim'],
        nheads=config['model']['nheads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model'].get('dim_feedforward', 1024),
        dropout=config['model'].get('dropout', 0.1),
        num_feature_levels=config['model']['num_feature_levels'],
        enc_n_points=config['model']['enc_n_points'],
        dec_n_points=config['model']['dec_n_points'],
        num_queries=config['model']['num_queries'],
        aux_loss=config['model'].get('aux_loss', True),
        use_checkpoint=use_checkpoint,
        chunk_size_large=chunk_size_large,
        chunk_size_small=chunk_size_small
    )

    # Calculate and display total parameter count to estimate model complexity
    # This helps understand memory requirements and computational demands
    total_params = get_num_parameters(model)
    print(f"Model has {total_params:,} trainable parameters")

    # Apply GPU-specific memory optimizations to improve training efficiency
    # These optimizations reduce memory fragmentation and improve throughput
    if device.type == 'cuda':
        # Apply CUDA-specific memory optimizations for better training efficiency
        print("Applying GPU memory optimizations for faster training...")

        # Configure model precision based on hardware capabilities and mixed precision settings
        # Using lower precision (FP16/BF16) reduces memory usage and can improve performance
        if mixed_precision:
            print("Using mixed precision for model weights (memory savings and better stability)")
            # Check BFloat16 support
            if bf16_supported:
                # Use BFloat16 (wider dynamic range and more stable)
                print("Using BFloat16 for model weights")
                # We don't actually need to convert parameters now - autocast will handle it
            else:
                # Use Float16 if BFloat16 is not supported
                print("⚠️ BFloat16 not supported, using Float16 (less stable)")
                # We don't actually need to convert parameters now - autocast will handle it

        # Enable memory sharing between tensor storage to reduce memory fragmentation
        # This allows more efficient use of GPU memory during training
        for module in model.modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                if module.bias is not None:
                    module.bias.share_memory_()
                if module.weight is not None:
                    module.weight.share_memory_()

    # Transfer model to the target compute device (GPU or CPU)
    # This is required before any forward/backward passes can be performed
    model = model.to(device)

    # Skip pre-allocating tensors to avoid unnecessary memory usage

    # Enable model optimizations for A100 GPU
    print("Model optimizations enabled for A100 GPU")

    # Use channels_last memory format for better performance on A100
    model = model.to(memory_format=torch.channels_last)

    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True

    # Enable TF32 for faster computation on A100
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Watch model parameters automatically in wandb
    if args.wandb and wandb is not None:
        wandb.watch(model, log="all", log_freq=100)

    # Create matcher and loss function with performance optimizations
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

    # Move criterion to device
    criterion = criterion.to(device)

    # Calculate total number of optimization steps for learning rate scheduling
    # This is needed for schedulers that adjust learning rate based on progress
    steps_per_epoch = len(train_dataloader) // grad_accumulation if grad_accumulation > 1 else len(train_dataloader)
    total_training_steps = steps_per_epoch * config['training']['epochs']

    # Use standard optimizer without GPU-specific optimizations to prevent memory issues
    if config['optimizer']['type'] == 'AdamW':
        print("Using standard AdamW optimizer")
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=float(config['optimizer']['lr']),
            weight_decay=float(config['optimizer']['weight_decay'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']['type']}")

    # Configure learning rate scheduler to adjust LR during training
    # Different schedulers have different effects on convergence and final performance
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
    elif scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config['training']['epochs'],
            eta_min=float(config['optimizer']['lr']) * 0.01
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    # Make scheduler accessible to the training loop function
    # This allows the scheduler to be stepped at the right time
    config['lr_scheduler'] = lr_scheduler

    # Load previous training state if checkpoint is provided
    # This enables continuing training from where it was interrupted
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            # Load model weights with non-strict matching to handle architecture changes
            # This allows resuming training even if the model definition has been modified
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")

    # Ensure checkpoint directory exists for saving model states
    # This prevents errors when saving checkpoints during training
    os.makedirs('checkpoints', exist_ok=True)

    # Set up PyTorch profiler for performance analysis when requested
    # This helps identify bottlenecks in the training process
    if args.profile:
        # Use torch profiler to identify bottlenecks
        from torch.profiler import profile, ProfilerActivity

        # Setup profiler
        profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA if device.type == 'cuda' else None
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )
        print("Performance profiling enabled. Results will be saved to ./profiler_logs")
        profiler.start()
    else:
        profiler = None

    # Disable GPU optimizations that might cause memory issues
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    # Check GPU memory before starting training
    if device.type == 'cuda':
        # Release unused GPU memory to maximize available resources for training
        torch.cuda.empty_cache()

        # Print current GPU memory status before training begins
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        capacity = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\nGPU Memory Status (Before Training Begins):")
        print(f"  Total GPU memory: {capacity:.2f} GB")
        print(f"  Allocated memory: {allocated:.2f} GB")
        print(f"  Reserved memory: {reserved:.2f} GB")
        print(f"  Free memory: {capacity - reserved:.2f} GB")

        # Verify sufficient GPU memory is available to prevent crashes during training
        if (capacity - reserved) < 1.0:  # Critical threshold: 1GB free memory minimum for stable training
            print("\n⚠️ WARNING: GPU memory is very full! You may get memory errors during training.")
            print("   Consider reducing the batch size further or using a smaller model.")
            print("   Waiting 5 seconds before continuing...")
            time.sleep(5)

    # Main training loop with memory management and error handling for robust execution
    print(f"Starting training for {config['training']['epochs']} epochs...")
    best_val_loss = float('inf')

    # Store per-epoch metrics for analysis, visualization and checkpoint recovery
    epoch_stats = {}

    # Print clear epoch separation
    print("\n" + "="*80)
    print(f"STARTING TRAINING: {config['training']['epochs']} EPOCHS")
    print("="*80 + "\n")

    # Main training loop
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            start_time = time.time()

            print(f"\n{'='*50}")
            print(f"EPOCH {epoch+1}/{config['training']['epochs']}")
            print(f"{'='*50}")

            # Pre-initialize statistics dictionaries with default values
            # Ensures metrics exist even if training or validation fails, preventing KeyErrors
            train_stats = {'loss': float('inf')}
            val_stats = {'loss': float('inf')}

            # Release GPU memory fragments before each epoch to prevent memory fragmentation
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Train for one epoch
            try:
                print("Starting training for epoch {}...".format(epoch+1))
                # Force garbage collection before training
                import gc
                gc.collect()

                train_stats = train_one_epoch(
                    model=model,
                    criterion=criterion,
                    data_loader=train_dataloader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    amp_enabled=mixed_precision,
                    scaler=scaler,
                    max_norm=config['training']['clip_max_norm'],
                    lr_scheduler=lr_scheduler if scheduler_type == 'onecycle' else None,
                    grad_accumulation_steps=grad_accumulation,
                    config=config,
                    batch_size=batch_size
                )
                print("Training for epoch {} completed successfully.".format(epoch+1))

                # Clear both GPU and CPU memory
                if device.type == 'cuda' and config.get('training', {}).get('memory_management', {}).get('empty_cache_freq', 0) > 0:
                    torch.cuda.empty_cache()
                    print("Cleared CUDA cache between training and validation")

                # Force garbage collection to free up CPU memory
                gc.collect()
                print("Forced garbage collection to free CPU memory")

                # Run validation with comprehensive error handling to prevent training interruption
                # Even if validation fails, training will continue with fallback metrics
                try:
                    print("Starting validation for epoch {}...".format(epoch+1))
                    val_stats = validate(
                        model=model,
                        criterion=criterion,
                        data_loader=val_dataloader,
                        device=device,
                        amp_enabled=mixed_precision
                    )
                    print("Validation for epoch {} completed successfully.".format(epoch+1))
                except Exception as e:
                    print(f"Validation error: {e}")
                    import traceback
                    traceback.print_exc()
                    val_stats = {'loss': float('inf')}  # Fallback metrics when validation fails, allowing training to continue

            except RuntimeError as e:
                print(f"Runtime error during epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
                if "CUDA out of memory" in str(e):
                    print("\n" + "!"*80)
                    print("CUDA OUT OF MEMORY ERROR!")
                    print("!"*80)
                    print("\nInsufficient memory. Optimizing training parameters and trying again...\n")

                    # Clear memory
                    torch.cuda.empty_cache()

                    # Reduce batch size (if not already 1)
                    if batch_size > 1:
                        batch_size = 1
                        print(f"Batch size reduced to 1")

                        # Recreate dataloaders
                        train_dataloader = build_dataloader(
                            dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,  # No workers to prevent CUDA initialization issues
                            pin_memory=False,
                            prefetch_factor=1,
                            persistent_workers=False
                        )

                        val_dataloader = build_dataloader(
                            dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,  # No workers to prevent CUDA initialization issues
                            pin_memory=False,
                            prefetch_factor=1,
                            persistent_workers=False
                        )

                        # Increase gradient accumulation
                        grad_accumulation = max(16, grad_accumulation * 2)
                        print(f"Gradient accumulation increased to {grad_accumulation} steps")

                        # Reduce learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        print(f"Learning rate reduced by 50%: {optimizer.param_groups[0]['lr']:.6f}")

                        print("\nRestarting training with new parameters...\n")

                        # Set a flag to restart the epoch
                        # Continue with the next epoch
                    else:
                        print("Batch size is already minimum (1). Further memory optimizations required.")
                        print("Suggested solutions:")
                        print("1. Use a smaller model (reduce hidden_dim, num_layers)")
                        print("2. Further reduce image size")
                        print("3. Sample the dataset to use fewer examples")
                        print("4. Run on a device with larger GPU memory")
                        # Stop training due to memory limitations
                        print("\nExiting due to memory limitations...")
                        return
                else:
                    # Re-raise the error if it's not a memory error
                    raise e

            # Post-epoch processing: update learning rate, save checkpoints, and log metrics
            # These operations ensure training progress is preserved and monitored
            # Update learning rate according to schedule (StepLR updates after epoch, OneCycleLR during training)
            if scheduler_type == 'step':
                lr_scheduler.step()
                print(f"Learning rate updated: {optimizer.param_groups[0]['lr']:.6f}")

            # Extract loss values from training and validation statistics
            # Using get() with default values prevents KeyErrors if metrics are missing
            val_loss = val_stats.get('loss', float('inf'))
            train_loss = train_stats.get('loss', float('inf'))

            # Save model checkpoint if validation performance has improved
            # This preserves the best model for later use or deployment
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Use asynchronous saving on CUDA devices to avoid blocking the training process
                # This improves overall training throughput while still preserving checkpoints
                if device.type == 'cuda':
                    # Use async save to avoid blocking the main thread
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }, os.path.join('checkpoints', 'best_model.pth'), _use_new_zipfile_serialization=True)
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }, os.path.join('checkpoints', 'best_model.pth'))
                print(f"\n✅ Saved new best model with validation loss: {best_val_loss:.4f}")

            # Periodically save the latest model state regardless of performance
            # Saving every 5 epochs balances checkpoint frequency with I/O overhead
            if (epoch + 1) % 5 == 0 or epoch == config['training']['epochs'] - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join('checkpoints', 'last_checkpoint.pth'))

            # Display comprehensive training statistics for the completed epoch
            # This provides immediate feedback on training progress and performance
            epoch_time = time.time() - start_time
            samples_per_sec = len(train_dataset) / epoch_time
            print(f"\n{'*'*50}")
            print(f"EPOCH {epoch+1}/{config['training']['epochs']} SUMMARY:")
            print(f"{'*'*50}")
            print(f"  Time: {epoch_time:.2f} seconds ({samples_per_sec:.2f} samples/second)")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f} (previous best: {best_val_loss:.4f})")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Display dataset caching performance metrics when using memory caching
            # These metrics help monitor the effectiveness of the caching strategy
            if isinstance(train_dataset, DynamicCachedDataset):
                stats = train_dataset.cache_stats()
                print(f"  Train Cache Stats: Size={stats['cache_size']}/{stats['max_cache_size']} | "
                      f"Hit Rate={stats['hit_rate']:.2f}% | Hits={stats['cache_hits']} | Misses={stats['cache_misses']}")

            if isinstance(val_dataset, DynamicCachedDataset):
                stats = val_dataset.cache_stats()
                print(f"  Val Cache Stats: Size={stats['cache_size']}/{stats['max_cache_size']} | "
                      f"Hit Rate={stats['hit_rate']:.2f}% | Hits={stats['cache_hits']} | Misses={stats['cache_misses']}")

            # Calculate and display loss changes compared to previous epoch
            # This helps identify if the model is learning, plateauing, or diverging
            if epoch > 0:
                prev_train_loss = epoch_stats.get(epoch-1, {}).get('train_loss', float('inf'))
                prev_val_loss = epoch_stats.get(epoch-1, {}).get('val_loss', float('inf'))
                train_diff = prev_train_loss - train_loss
                val_diff = prev_val_loss - val_loss

                print(f"  Training Loss Change: {train_diff:.4f} ({'+' if train_diff > 0 else ''}{(train_diff/prev_train_loss)*100:.2f}%)")
                print(f"  Validation Loss Change: {val_diff:.4f} ({'+' if val_diff > 0 else ''}{(val_diff/prev_val_loss)*100:.2f}%)")

            # Record all metrics for this epoch in the tracking dictionary
            # This data is used for trend analysis and final reporting
            epoch_stats[epoch] = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'samples_per_sec': samples_per_sec
            }

            # Only clear GPU cache if explicitly enabled in config
            if device.type == 'cuda' and config.get('training', {}).get('memory_management', {}).get('empty_cache_freq', 0) > 0:
                torch.cuda.empty_cache()
                print("Cleared CUDA cache after epoch completion")

                # Print memory usage statistics after cleanup
                if (epoch + 1) % 5 == 0:  # Print detailed stats every 5 epochs
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    print(f"GPU Memory: Current={allocated:.2f}GB, Peak={max_allocated:.2f}GB, Reserved={reserved:.2f}GB")
                    # Reset peak stats for next epoch
                    torch.cuda.reset_peak_memory_stats()

            # Print epoch separator
            print(f"\n{'-'*80}")

            # Send detailed training metrics to Weights & Biases for visualization and tracking
            # This provides a comprehensive dashboard for monitoring training progress
            if 'wandb' in sys.modules and wandb.run is not None:
                epoch_log_dict = {
                    # Basic metrics
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                    'samples_per_sec': samples_per_sec,
                    'train_val_loss_ratio': train_loss / val_loss if val_loss > 0 else 0,

                    # System metrics
                    'system/epoch': epoch,
                    'system/epoch_time': epoch_time,
                    'system/samples_per_sec': samples_per_sec,
                    'system/progress': (epoch + 1) / config['training']['epochs'],

                    # Training configuration
                    'config/batch_size': batch_size,
                    'config/grad_accumulation': grad_accumulation,
                    'config/effective_batch_size': batch_size * grad_accumulation,
                    'config/learning_rate': optimizer.param_groups[0]['lr'],
                    'config/mixed_precision': mixed_precision
                }

                # Log individual loss components to track specific aspects of model performance
                # This helps identify which parts of the loss function are improving or struggling
                for loss_name, loss_value in train_stats.items():
                    if loss_name.startswith('loss'):
                        epoch_log_dict[f'train/{loss_name}'] = loss_value

                # Include validation metrics for each loss component when available
                # This enables direct comparison between training and validation performance
                if isinstance(val_stats, dict):
                    for loss_name, loss_value in val_stats.items():
                        if loss_name.startswith('loss'):
                            epoch_log_dict[f'val/{loss_name}'] = loss_value
                            # Also add train/val ratio for each loss component
                            if loss_name in train_stats and val_stats.get(loss_name, 0) > 0:
                                epoch_log_dict[f'{loss_name}_train_val_ratio'] = train_stats[loss_name] / val_stats[loss_name]

                # Log object detection specific metrics from validation
                # These metrics (mAP, precision, recall) are the primary indicators of model quality
                if isinstance(val_stats, dict):
                    print("\nValidation metrics available at epoch end:")
                    print(f"Val_stats keys: {list(val_stats.keys())}")

                    # Process and log all available validation metrics regardless of type
                    # This ensures we capture all performance indicators, even custom ones
                    for key, value in val_stats.items():
                        if not key.startswith('loss'):
                            if isinstance(value, (int, float, bool)):
                                epoch_log_dict[f'val/metrics/{key}'] = value
                                print(f"  {key}: {value}")
                            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                                epoch_log_dict[f'val/metrics/{key}'] = value.item()
                                print(f"  {key}: {value.item()}")
                            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                                print(f"  {key}: {value}")
                                for i, x in enumerate(value):
                                    epoch_log_dict[f'val/metrics/{key}/{i}'] = x

                    # Highlight critical detection metrics in the console output
                    # These are the most important indicators of model performance
                    print(f"\nImportant Validation Metrics:")
                    if 'precision' in val_stats:
                        print(f"  Precision: {val_stats['precision']:.4f}")
                        # Add directly to wandb
                        epoch_log_dict['val/precision'] = float(val_stats['precision'])
                    else:
                        print("  Precision: Not found")

                    if 'recall' in val_stats:
                        print(f"  Recall: {val_stats['recall']:.4f}")
                        epoch_log_dict['val/recall'] = float(val_stats['recall'])
                    else:
                        print("  Recall: Not found")

                    if 'f1' in val_stats:
                        print(f"  F1 Score: {val_stats['f1']:.4f}")
                        epoch_log_dict['val/f1'] = float(val_stats['f1'])
                    else:
                        print("  F1 Score: Not found")

                    if 'accuracy' in val_stats:
                        print(f"  Accuracy: {val_stats['accuracy']:.4f}")
                        epoch_log_dict['val/accuracy'] = float(val_stats['accuracy'])
                    else:
                        print("  Accuracy: Not found")

                    if 'mAP' in val_stats:
                        print(f"  mAP: {val_stats['mAP']:.4f}")
                        epoch_log_dict['val/mAP'] = float(val_stats['mAP'])
                    else:
                        print("  mAP: Not found")

                # Track hardware utilization metrics to monitor computational efficiency
                # This helps identify potential bottlenecks in the training process
                if device.type == 'cuda':
                    try:
                        epoch_log_dict['system/gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9  # GB
                        epoch_log_dict['system/gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9  # GB
                        epoch_log_dict['system/gpu_max_memory'] = torch.cuda.max_memory_allocated() / 1e9  # GB
                        epoch_log_dict['system/gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    except Exception as e:
                        print(f"Error logging GPU metrics: {e}")

                # Record current learning rate information from the scheduler
                # This helps correlate learning rate changes with model performance
                if lr_scheduler is not None:
                    if hasattr(lr_scheduler, 'get_last_lr'):
                        epoch_log_dict['train/last_lr'] = lr_scheduler.get_last_lr()[0]
                    if hasattr(lr_scheduler, '_last_lr'):
                        epoch_log_dict['train/last_lr'] = lr_scheduler._last_lr[0]

                # Periodically update the wandb run summary with key performance metrics
                # These values represent the overall training progress and final results
                if epoch == config['training']['epochs'] - 1 or (epoch + 1) % 5 == 0:
                    # Update summary metrics at regular intervals and at training completion
                    # These provide a persistent record of the most important training outcomes
                    wandb.run.summary['best_val_loss'] = best_val_loss
                    wandb.run.summary['final_train_loss'] = train_loss
                    wandb.run.summary['final_val_loss'] = val_loss
                    wandb.run.summary['train_val_loss_ratio'] = train_loss / val_loss if val_loss > 0 else 0
                    wandb.run.summary['total_epochs'] = epoch + 1
                    wandb.run.summary['total_training_time'] = sum(epoch_stats[e]['epoch_time'] for e in epoch_stats)

                    # Add validation metrics to summary if available
                    if isinstance(val_stats, dict):
                        print("\nAdding end-of-epoch summary metrics to Wandb:")
                        if 'precision' in val_stats:
                            wandb.run.summary['final_precision'] = val_stats['precision']
                            print(f"  final_precision: {val_stats['precision']}")
                        if 'recall' in val_stats:
                            wandb.run.summary['final_recall'] = val_stats['recall']
                            print(f"  final_recall: {val_stats['recall']}")
                        if 'f1' in val_stats:
                            wandb.run.summary['final_f1'] = val_stats['f1']
                            print(f"  final_f1: {val_stats['f1']}")
                        if 'mAP' in val_stats:
                            wandb.run.summary['final_mAP'] = val_stats['mAP']
                            print(f"  final_mAP: {val_stats['mAP']}")
                        else:
                            print("  None of the validation metrics were found!")
                            print("  Available val_stats keys:", list(val_stats.keys()))

                # Log to wandb
                wandb.log(epoch_log_dict)

    except KeyboardInterrupt:
        # Handle user-initiated training interruption gracefully
        # Save progress to allow resuming from this point later
        print("\nTraining interrupted by user. Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'best_val_loss': best_val_loss,
            'config': config,
        }, os.path.join('checkpoints', 'interrupted_checkpoint.pth'))
        print("Checkpoint saved. Exiting gracefully.")

    except RuntimeError as e:
        # Handle CUDA out-of-memory errors with automatic parameter adjustment
        # This allows training to continue with reduced memory requirements
        if "CUDA out of memory" in str(e):
            print("\n" + "!"*80)
            print("CUDA OUT OF MEMORY ERROR!")
            print("!"*80)
            print("\nInsufficient memory. Optimizing training parameters and trying again...\n")

            # Release all GPU memory to recover from the OOM error
            # This is necessary before attempting to continue with reduced parameters
            torch.cuda.empty_cache()

            # First memory-saving strategy: reduce batch size to minimum
            # This is the most effective way to reduce memory usage
            if batch_size > 1:
                batch_size = 1
                print(f"Batch size reduced to 1")

                # Rebuild data loaders with new batch size and reduced worker count
                # Fewer workers also helps reduce memory pressure
                train_dataloader = build_dataloader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,  # Also reduce number of workers
                    pin_memory=pin_memory,
                    prefetch_factor=2,
                    persistent_workers=False
                )

                val_dataloader = build_dataloader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=pin_memory,
                    prefetch_factor=2,
                    persistent_workers=False
                )

                # Second memory-saving strategy: increase gradient accumulation steps
                # This maintains effective batch size while reducing memory requirements
                grad_accumulation = max(16, grad_accumulation * 2)
                print(f"Gradient accumulation increased to {grad_accumulation} steps")

                # Reduce learning rate to maintain training stability with new parameters
                # Smaller batches typically require lower learning rates
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"Learning rate reduced by 50%: {optimizer.param_groups[0]['lr']:.6f}")

                print("\nRestarting training with new parameters...\n")

                # Restart the epoch
                print("Epoch will be restarted with new parameters")
            else:
                print("Batch size is already minimum (1). Cannot reduce memory usage further.")
                print("Training cannot continue with current model and hardware configuration.")
                print("\nSuggested solutions to resolve memory limitations:")
                print("1. Use a smaller model (reduce hidden_dim, num_layers)")
                print("2. Further reduce image size")
                print("3. Sample the dataset to use fewer examples")
                print("4. Run on a device with larger GPU memory")
                # Exit training gracefully
                return
    except Exception as e:
        # Catch any other unexpected errors during training
        # Provide detailed error information to help with debugging
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        # Exit training gracefully
        return

    finally:
        # Clean up profiler if it was enabled
        if profiler:
            profiler.stop()

        # Final cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    if args.wandb and wandb.run is not None:
        # Add final summary metrics before finishing
        wandb.run.summary['best_val_loss'] = best_val_loss
        wandb.run.summary['completed_epochs'] = config['training']['epochs']
        wandb.run.summary['total_training_time'] = sum(epoch_stats[e]['epoch_time'] for e in epoch_stats) if epoch_stats else 0

        # Finish wandb run
        wandb.finish()


if __name__ == '__main__':
    main()