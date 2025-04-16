"""
Deformable DETR Model Evaluation File

This file contains the necessary functions for evaluating the trained Deformable DETR model
on the AU-AIR dataset. Metrics such as mAP (mean Average Precision) and AP (Average Precision)
are calculated to measure the model's performance, and the results are visualized.

Usage:
    python eval.py --checkpoint <model_path> --config <config_path> --dataset-path <data_path> 
                   --annotations-file <anno_path> --split test --device cuda --visualize --wandb

Parameters:
    --checkpoint: Model file to be evaluated
    --config: YAML file containing model configuration
    --dataset-path: Location of the dataset
    --annotations-file: Location of the annotation file
    --split: Dataset split to be used for evaluation (test, val)
    --device: Device to be used for evaluation (cuda, cpu)
    --visualize: Visualize predictions
    --wandb: Enables Weights & Biases integration

© 2023 AU-AIR Dataset and Deformable DETR Implementation
"""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou
from torch.cuda.amp import autocast
import json
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from torch.utils.data import Subset

from model.deformable_detr import DeformableDetrModel
from model.utils import box_cxcywh_to_xyxy
from model.dataset import build_dataset, build_dataloader


def parse_args():
    """
    Parses and returns command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate Deformable DETR on AU-AIR dataset")
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/deformable_detr_config.yaml',
                      help='Path to config file')
    parser.add_argument('--dataset-path', type=str, default='dataset',
                      help='Path to dataset directory')
    parser.add_argument('--annotations-file', type=str, default=None,
                      help='Path to annotations file (default: metadata/{split}_annotations.json)')
    parser.add_argument('--split', type=str, default='test',
                      help='Dataset split to evaluate on (train, val, test)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize predictions')
    parser.add_argument('--max-vis-samples', type=int, default=10,
                      help='Maximum number of samples to visualize')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable wandb logging')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                      help='Fraction of dataset to evaluate (0-1)')
    parser.add_argument('--mixed-precision', action='store_true',
                      help='Enable mixed precision evaluation for faster speed')
    return parser.parse_args()


def box_iou(boxes1, boxes2):
    """
    Calculates IoU (Intersection over Union) between two sets of bounding boxes.
    
    Parameters:
        boxes1: First set of bounding boxes [N, 4]
        boxes2: Second set of bounding boxes [M, 4]
    
    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - intersection
    
    return intersection / union


def compute_ap(recall, precision):
    """
    Calculates Average Precision (AP) from the precision-recall curve.
    
    Parameters:
        recall: Array of recall values
        precision: Array of precision values
    
    Returns:
        AP value
    """
    # Make sure precision and recall arrays start with 0 and end with 0
    mrec = [0] + list(recall) + [1]
    mpre = [0] + list(precision) + [0]
    
    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Look for points where recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    
    # Calculate area under the curve
    ap = 0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    
    return ap


def evaluate(model, dataloader, device, class_names, confidence_threshold=0.5, mixed_precision=False):
    """
    Evaluates the model on the dataset and calculates performance metrics.
    
    Parameters:
        model: Model to be evaluated
        dataloader: Data loader
        device: Processing device (CPU/GPU)
        class_names: List of class names
        confidence_threshold: Confidence threshold for predictions
        mixed_precision: Enable evaluation with mixed precision
    
    Returns:
        metrics: Evaluation metrics (mAP, AP values, etc.)
        predictions: Model predictions
        targets: Ground truth values
    """
    model.eval()
    num_classes = len(class_names)
    
    # Initialize metrics
    class_metrics = {
        cls_id: {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'gt_count': 0,
            'predictions': []
        }
        for cls_id in range(num_classes)
    }
    
    confusion_matrix_data = np.zeros((num_classes, num_classes + 1))  # +1 for no detection
    
    # Store some examples for visualization
    all_predictions = []
    all_targets = []
    vis_count = 0
    max_vis_samples = 10  # Number of samples to store for visualization
    
    # Process dataset
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Mixed precision evaluation if enabled
            if mixed_precision:
                with autocast():
                    # Forward pass
                    outputs = model(images)
            else:
                # Forward pass
                outputs = model(images)
            
            # Process batch output
            for i, (image, target, output) in enumerate(zip(images, targets, outputs['pred_logits'])):
                # Store for visualization (if max_vis_samples not reached)
                if vis_count < max_vis_samples:
                    # Calculate image scale to bring normalized boxes to original pixel space
                    W, H = image.shape[1:]
                    
                    all_predictions.append({
                        'image': image.cpu(),
                        'logits': output.cpu(),
                        'boxes': outputs['pred_boxes'][i].cpu(),
                        'width': W,
                        'height': H,
                    })
                    
                    all_targets.append({
                        'boxes': target['boxes'].cpu(),
                        'labels': target['labels'].cpu(),
                        'width': W,
                        'height': H,
                    })
                    
                    vis_count += 1
                
                # Get predictions
                logits = output  # [num_queries, num_classes + 1]
                boxes = outputs['pred_boxes'][i]  # [num_queries, 4] (cxcywh format, normalized)
                
                # Get scores (remove the background class)
                scores = torch.nn.functional.softmax(logits, dim=-1)[:, :-1]  # [num_queries, num_classes]
                
                # Find best score and class for each query
                max_scores, max_classes = scores.max(dim=1)
                
                # Apply confidence threshold
                keep = max_scores > confidence_threshold
                max_scores = max_scores[keep]
                max_classes = max_classes[keep]
                boxes = boxes[keep]
                
                # Skip if no predictions
                if len(max_classes) == 0:
                    # Count false negatives (all ground truth are missed)
                    for label in target['labels']:
                        label_id = label.item()
                        if label_id < num_classes:
                            class_metrics[label_id]['fn'] += 1
                            class_metrics[label_id]['gt_count'] += 1
                            
                            # Update confusion matrix for FN
                            confusion_matrix_data[label_id, num_classes] += 1
                    continue
                
                # Convert predictions to xyxy format
                pred_boxes = box_cxcywh_to_xyxy(boxes)
                
                # Get ground truth boxes and labels
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                # Count ground truth objects
                for label in gt_labels:
                    label_id = label.item()
                    if label_id < num_classes:
                        class_metrics[label_id]['gt_count'] += 1
                
                # Skip if no ground truth
                if len(gt_boxes) == 0:
                    # Count all predictions as false positives
                    for cls_id in max_classes:
                        cls_id = cls_id.item()
                        if cls_id < num_classes:
                            class_metrics[cls_id]['fp'] += 1
                            class_metrics[cls_id]['predictions'].append((0, 0))  # score, iou
                    continue
                
                # Convert ground truth to xyxy format
                gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
                
                # Compute IoU between predictions and ground truth
                iou_matrix = box_iou(pred_boxes, gt_boxes)  # [num_preds, num_gts]
                
                # For each prediction, check if it's a TP or FP
                for pred_idx, (pred_class, pred_score, ious) in enumerate(zip(max_classes, max_scores, iou_matrix)):
                    pred_class_id = pred_class.item()
                    
                    if pred_class_id >= num_classes:
                        continue
                    
                    # Find ground truth objects of the same class
                    matching_gt_indices = torch.where(gt_labels == pred_class_id)[0]
                    
                    if len(matching_gt_indices) == 0:
                        # False positive - no matching ground truth of this class
                        class_metrics[pred_class_id]['fp'] += 1
                        class_metrics[pred_class_id]['predictions'].append((pred_score.item(), 0))
                        
                        # Update confusion matrix for FP
                        confusion_matrix_data[pred_class_id, pred_class_id] += 1
                        continue
                    
                    # Find the best matching ground truth (highest IoU)
                    ious_for_class = ious[matching_gt_indices]
                    max_iou, max_idx = ious_for_class.max(dim=0)
                    max_gt_idx = matching_gt_indices[max_idx]
                    
                    if max_iou >= 0.5:
                        # True positive
                        class_metrics[pred_class_id]['tp'] += 1
                        
                        # Update confusion matrix for TP
                        confusion_matrix_data[pred_class_id, pred_class_id] += 1
                    else:
                        # False positive - IoU too low
                        class_metrics[pred_class_id]['fp'] += 1
                        
                        # Get the true class id for confusion matrix
                        true_class_id = gt_labels[max_gt_idx].item()
                        if true_class_id < num_classes:
                            confusion_matrix_data[true_class_id, pred_class_id] += 1
                    
                    class_metrics[pred_class_id]['predictions'].append((pred_score.item(), max_iou.item()))
                
                # Count false negatives - ground truth objects with no matching predictions
                gt_matched = torch.zeros(len(gt_labels), dtype=torch.bool)
                
                for gt_idx, gt_label in enumerate(gt_labels):
                    gt_class_id = gt_label.item()
                    if gt_class_id >= num_classes:
                        continue
                    
                    # Find predictions of the same class
                    matching_pred_indices = torch.where(max_classes == gt_class_id)[0]
                    
                    if len(matching_pred_indices) == 0:
                        # False negative - no predictions of this class
                        class_metrics[gt_class_id]['fn'] += 1
                        continue
                    
                    # Find the best matching prediction (highest IoU)
                    ious_for_gt = iou_matrix[matching_pred_indices, gt_idx]
                    max_iou, _ = ious_for_gt.max(dim=0)
                    
                    if max_iou < 0.5:
                        # False negative - IoU too low
                        class_metrics[gt_class_id]['fn'] += 1
                        
                        # Update confusion matrix for FN
                        confusion_matrix_data[gt_class_id, num_classes] += 1
    
    # Compute mAP and other metrics
    ap_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    class_ap = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    
    for class_id, metrics in class_metrics.items():
        # Skip classes with no ground truth
        if metrics['gt_count'] == 0:
            continue
        
        # Sort predictions by confidence score
        predictions = sorted(metrics['predictions'], key=lambda x: x[0], reverse=True)
        
        if not predictions:
            ap_values.append(0)
            precision_values.append(0)
            recall_values.append(0)
            f1_values.append(0)
            class_ap[class_id] = 0
            class_precision[class_id] = 0
            class_recall[class_id] = 0
            class_f1[class_id] = 0
            continue
        
        # Compute precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precision = []
        recall = []
        
        for idx, (score, iou) in enumerate(predictions):
            if iou >= 0.5:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recall.append(tp_cumsum / metrics['gt_count'])
        
        # Compute average precision
        ap = compute_ap(recall, precision)
        ap_values.append(ap)
        class_ap[class_id] = ap
        
        # Compute overall precision and recall
        TP = metrics['tp']
        FP = metrics['fp']
        FN = metrics['fn']
        
        class_precision[class_id] = TP / (TP + FP) if (TP + FP) > 0 else 0
        class_recall[class_id] = TP / (TP + FN) if (TP + FN) > 0 else 0
        class_f1[class_id] = 2 * (class_precision[class_id] * class_recall[class_id]) / (class_precision[class_id] + class_recall[class_id]) if (class_precision[class_id] + class_recall[class_id]) > 0 else 0
        
        precision_values.append(class_precision[class_id])
        recall_values.append(class_recall[class_id])
        f1_values.append(class_f1[class_id])
    
    # Compute mAP and other average metrics
    mAP = np.mean(ap_values) if ap_values else 0
    avg_precision = np.mean(precision_values) if precision_values else 0
    avg_recall = np.mean(recall_values) if recall_values else 0
    avg_f1 = np.mean(f1_values) if f1_values else 0
    
    # Combine all metrics
    metrics = {
        'mAP': mAP,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'class_ap': class_ap,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion_matrix': confusion_matrix_data
    }
    
    return metrics, all_predictions, all_targets


def visualize_predictions(predictions, targets, output_dir, class_names, confidence_threshold=0.5, max_samples=5):
    """
    Visualizes model predictions and saves them to the specified directory.
    
    Parameters:
        predictions: List of predictions
        targets: List of ground truth values
        output_dir: Directory to save the images
        class_names: List of class names
        confidence_threshold: Confidence threshold for predictions
        max_samples: Maximum number of images to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Only visualize up to max_samples
    predictions = predictions[:max_samples]
    targets = targets[:max_samples]
    
    for idx, (prediction, target) in enumerate(zip(predictions, targets)):
        # Get image
        image = prediction['image']
        # Convert to numpy and denormalize
        img_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Get image dimensions
        W, H = image.shape[1:]
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_np)
        
        # Get predictions
        logits = prediction['logits']  # [num_queries, num_classes + 1]
        pred_boxes = prediction['boxes']  # [num_queries, 4] (cxcywh format, normalized)
        
        # Get scores (remove the background class)
        scores = torch.nn.functional.softmax(logits, dim=-1)[:, :-1]  # [num_queries, num_classes]
        
        # Find best score and class for each query
        max_scores, max_classes = scores.max(dim=1)
        
        # Apply confidence threshold
        keep = max_scores > confidence_threshold
        max_scores = max_scores[keep]
        max_classes = max_classes[keep]
        pred_boxes = pred_boxes[keep]
        
        # Convert predictions to xyxy format
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        
        # Draw predicted boxes
        for box, cls_id, score in zip(pred_boxes, max_classes, max_scores):
            if cls_id >= len(class_names):
                continue
                
            # Denormalize box coordinates
            x1, y1, x2, y2 = box.tolist()
            x1 *= W
            y1 *= H
            x2 *= W
            y2 *= H
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, 
                edgecolor='r', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = class_names[cls_id]
            ax.text(
                x1, 
                y1 - 5, 
                f'{class_name}: {score:.2f}',
                color='r',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # Get ground truth boxes and labels
        gt_boxes = target['boxes']  # [num_boxes, 4] (cxcywh format, normalized)
        gt_labels = target['labels']  # [num_boxes]
        
        # Convert ground truth to xyxy format
        gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        
        # Draw ground truth boxes
        for box, label in zip(gt_boxes, gt_labels):
            if label >= len(class_names):
                continue
                
            # Denormalize box coordinates
            x1, y1, x2, y2 = box.tolist()
            x1 *= W
            y1 *= H
            x2 *= W
            y2 *= H
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, 
                edgecolor='g', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = class_names[label]
            ax.text(
                x1, 
                y1 - 5, 
                class_name,
                color='g',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        red_patch = patches.Patch(color='red', label='Predictions')
        green_patch = patches.Patch(color='green', label='Ground Truth')
        ax.legend(handles=[red_patch, green_patch], loc='upper right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_{idx}.png'), dpi=100)
        plt.close()
    
    print(f"Saved {len(predictions)} visualization images to {output_dir}")


def sample_dataset(dataset, fraction):
    """
    Select a portion of the dataset with random sampling
    
    Args:
        dataset: Dataset to be sampled
        fraction: Ratio of data to be used (between 0-1)
        
    Returns:
        Subset: Sampled dataset
    """
    if fraction >= 1.0:
        return dataset
    
    # Select random indices from the dataset
    dataset_size = len(dataset)
    num_samples = int(dataset_size * fraction)
    indices = random.sample(range(dataset_size), num_samples)
    
    # Create subset
    return Subset(dataset, indices)


def main():
    """
    Main evaluation function. Processes arguments, loads the model and
    manages the evaluation process.
    """
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device used: {device}")
    
    # Define annotations file based on split
    annotations_file = args.annotations_file
    if annotations_file is None:
        annotations_file = os.path.join('metadata', f'{args.split}_annotations.json')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset for '{args.split}' split...")
    dataset = build_dataset(
        root_dir=args.dataset_path,
        annotations_file=annotations_file,
        split=args.split,
        img_size=config['dataset']['img_size']
    )
    
    # Apply dataset sampling if enabled
    if args.sample_fraction < 1.0:
        original_size = len(dataset)
        dataset = sample_dataset(dataset, args.sample_fraction)
        print(f"Sampled dataset from {original_size} to {len(dataset)} examples ({args.sample_fraction:.1%})")
    
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Create data loader with optimized settings
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Load model and checkpoint
    print(f"Loading model from checkpoint file: {args.checkpoint}")
    print("Loading model...")
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Move model to device
    model = model.to(device)
    
    # Use channels last memory format for better performance on CUDA
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("Using channels_last memory format for better GPU performance")
    
    # Initialize wandb if enabled
    if args.wandb:
        print("Sending results to Weights & Biases...")
        
        wandb_config = {
            # Model parameters
            "model_name": config['model']['name'],
            "hidden_dim": config['model']['hidden_dim'],
            "nheads": config['model']['nheads'],
            "num_encoder_layers": config['model']['num_encoder_layers'],
            "num_decoder_layers": config['model']['num_decoder_layers'],
            "num_queries": config['model']['num_queries'],
            
            # Evaluation parameters
            "batch_size": config['training']['batch_size'],
            "confidence_threshold": 0.5,
            "sample_fraction": args.sample_fraction,
            "split": args.split,
            
            # Dataset information
            "dataset_name": config['dataset']['name'],
            "num_classes": config['dataset']['num_classes'],
            "class_names": config['dataset']['class_names'],
            "img_size": config['dataset']['img_size'],
        }
        
        # Initialize wandb
        run = wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=f"{config['wandb']['name']}_eval_{args.split}",
            config=wandb_config,
            tags=config['wandb']['tags'] + [f"split_{args.split}"]
        )
    
    # Evaluate model
    print(f"Evaluating model...")
    metrics, predictions, targets = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=config['dataset']['class_names'],
        confidence_threshold=0.5,
        mixed_precision=args.mixed_precision
    )
    
    # Print metrics
    print("\nEvaluation results:")
    print(f"mAP@0.5: {metrics['mAP']:.4f}")
    print("\nClass-wise AP@0.5 values:")
    
    # Print per-class metrics
    for class_id, ap in metrics['class_ap'].items():
        class_name = config['dataset']['class_names'][class_id] if class_id < len(config['dataset']['class_names']) else 'Unknown'
        prec = metrics['class_precision'].get(class_id, 0)
        rec = metrics['class_recall'].get(class_id, 0)
        f1 = metrics['class_f1'].get(class_id, 0)
        print(f"{class_name}: AP={ap:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    # Save metrics to JSON
    json_metrics = {
        'mAP': float(metrics['mAP']),
        'avg_precision': float(metrics['avg_precision']),
        'avg_recall': float(metrics['avg_recall']),
        'avg_f1': float(metrics['avg_f1']),
        'class_metrics': {
            config['dataset']['class_names'][class_id] if class_id < len(config['dataset']['class_names']) else f'Unknown_{class_id}': {
                'ap': float(ap),
                'precision': float(metrics['class_precision'].get(class_id, 0)),
                'recall': float(metrics['class_recall'].get(class_id, 0)),
                'f1': float(metrics['class_f1'].get(class_id, 0))
            }
            for class_id, ap in metrics['class_ap'].items()
        }
    }
    
    with open(os.path.join(args.output_dir, f'metrics_{args.split}.json'), 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    # Create and save confusion matrix
    class_names = config['dataset']['class_names']
    conf_matrix = metrics['confusion_matrix']
    
    # Normalize by row (true classes)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    norm_conf_matrix = conf_matrix / row_sums
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Add "No detection" to class names
    all_classes = class_names + ["No detection"]
    
    # Only include classes with ground truth examples
    active_classes = [i for i, sum_val in enumerate(conf_matrix.sum(axis=1)) if sum_val > 0]
    
    # Handle case with no active classes
    if not active_classes:
        active_classes = list(range(min(len(class_names), 5)))
    
    norm_conf_matrix_subset = norm_conf_matrix[active_classes][:, active_classes + [len(class_names)]]
    class_names_subset = [all_classes[i] for i in active_classes] + ["No detection"]
    
    sns.heatmap(
        norm_conf_matrix_subset,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names_subset,
        yticklabels=[all_classes[i] for i in active_classes],
        ax=ax
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png'), dpi=100)
    plt.close()
    
    # Create and save AP bar chart
    plt.figure(figsize=(12, 6))
    
    # Get class AP values
    class_aps = [(class_id, ap) for class_id, ap in metrics['class_ap'].items()]
    sorted_class_aps = sorted(class_aps, key=lambda x: x[1], reverse=True)
    
    # Only show classes with ground truth examples
    class_ids = [class_id for class_id, _ in sorted_class_aps]
    ap_values = [ap for _, ap in sorted_class_aps]
    class_names_to_show = [class_names[i] if i < len(class_names) else f'Unknown_{i}' for i in class_ids]
    
    # Plot horizontal bar chart
    plt.barh(range(len(class_ids)), ap_values, align='center')
    plt.yticks(range(len(class_ids)), class_names_to_show)
    plt.xlabel('Average Precision (AP)')
    plt.title(f'Per-class AP values (mAP: {metrics["mAP"]:.4f})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'ap_values_{args.split}.png'), dpi=100)
    plt.close()
    
    # Visualize predictions
    if args.visualize:
        print(f"Visualizing predictions and saving to directory {args.output_dir}...")
        
        visualize_predictions(
            predictions=predictions,
            targets=targets,
            output_dir=os.path.join(args.output_dir, 'visualizations'),
            class_names=config['dataset']['class_names'],
            confidence_threshold=0.5,
            max_samples=args.max_vis_samples
        )
    
    # Log to wandb
    if args.wandb:
        # Log metrics
        wandb.log({
            'mAP': metrics['mAP'],
            'avg_precision': metrics['avg_precision'],
            'avg_recall': metrics['avg_recall'],
            'avg_f1': metrics['avg_f1'],
        })
        
        # Log per-class metrics as a table
        class_metrics_table = wandb.Table(columns=['Class', 'AP', 'Precision', 'Recall', 'F1'])
        
        for class_id, ap in metrics['class_ap'].items():
            class_name = config['dataset']['class_names'][class_id] if class_id < len(config['dataset']['class_names']) else f'Unknown_{class_id}'
            prec = metrics['class_precision'].get(class_id, 0)
            rec = metrics['class_recall'].get(class_id, 0)
            f1 = metrics['class_f1'].get(class_id, 0)
            
            class_metrics_table.add_data(class_name, ap, prec, rec, f1)
        
        wandb.log({'class_metrics': class_metrics_table})
        
        # Log confusion matrix
        wandb.log({
            'confusion_matrix': wandb.Image(os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png')),
            'ap_values': wandb.Image(os.path.join(args.output_dir, f'ap_values_{args.split}.png'))
        })
        
        # Log visualizations
        if args.visualize:
            vis_images = []
            for i in range(min(len(predictions), args.max_vis_samples)):
                vis_images.append(wandb.Image(
                    os.path.join(args.output_dir, 'visualizations', f'prediction_{i}.png'),
                    caption=f"Prediction {i}"
                ))
            
            if vis_images:
                wandb.log({'visualizations': vis_images})
        
        # Finish wandb
        wandb.finish()
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main() 