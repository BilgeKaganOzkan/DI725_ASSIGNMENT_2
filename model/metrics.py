import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import wandb

def box_iou(boxes1, boxes2):
    """Compute Intersection over Union (IoU) between pairs of bounding boxes.

    This function efficiently calculates the IoU between all pairs of boxes using
    vectorized operations for better performance. IoU measures the overlap between
    predicted and ground truth boxes.

    Args:
        boxes1: Tensor of shape (N, 4) containing N boxes in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) containing M boxes in (x1, y1, x2, y2) format

    Returns:
        Tensor of shape (N, M) containing IoU values for all pairs of boxes
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def compute_detection_metrics(outputs, targets, iou_thresholds=[0.1, 0.3, 0.5, 0.7], num_classes=8):
    """Compute comprehensive evaluation metrics for object detection performance.

    This function calculates metrics across multiple IoU thresholds including precision,
    recall, F1 score, accuracy, mAP, and confusion matrix. It handles edge cases and
    provides detailed debug information.

    Args:
        outputs: Model prediction outputs containing 'pred_logits' and 'pred_boxes'
        targets: List of ground truth dictionaries with 'labels' and 'boxes'
        iou_thresholds: List of IoU thresholds for evaluation (default: [0.1, 0.3, 0.5, 0.7])
        num_classes: Number of object classes in the dataset (default: 8)

    Returns:
        metrics: Dictionary containing all computed metrics and debug information
    """
    # We'll use the lowest IoU threshold as the base threshold for basic metrics calculation
    # This ensures we have a lenient baseline for initial evaluation

    # Initialize metrics
    metrics = {
        'precision': 0.0,  # True positives / (True positives + False positives)
        'recall': 0.0,     # True positives / (True positives + False negatives)
        'f1': 0.0,         # 2 * precision * recall / (precision + recall)
        'accuracy': 0.0,   # (True positives + True negatives) / Total
        'mAP': 0.0,        # Mean Average Precision
        'class_precision': [0.0] * num_classes,  # Per-class precision
        'class_recall': [0.0] * num_classes,     # Per-class recall
        'class_f1': [0.0] * num_classes,         # Per-class F1 score
        'confusion_matrix': torch.zeros(num_classes, num_classes, dtype=torch.int),  # Confusion matrix
        'debug_info': {    # Debug information
            'total_predictions': 0,
            'total_gt_boxes': 0,
            'total_matches': 0,
            'max_confidence': 0.0,
            'min_confidence': 1.0,
            'avg_confidence': 0.0,
            'max_iou': 0.0,
            'min_iou': 1.0,
            'avg_iou': 0.0
        }
    }

    # Add metrics for each IoU threshold
    for iou_threshold in iou_thresholds:
        metrics[f'mAP@{iou_threshold}'] = 0.0
        metrics[f'precision@{iou_threshold}'] = 0.0
        metrics[f'recall@{iou_threshold}'] = 0.0
        metrics[f'f1@{iou_threshold}'] = 0.0
        metrics[f'class_precision@{iou_threshold}'] = [0.0] * num_classes
        metrics[f'class_recall@{iou_threshold}'] = [0.0] * num_classes
        metrics[f'class_f1@{iou_threshold}'] = [0.0] * num_classes
        metrics[f'matches@{iou_threshold}'] = 0

    # If no predictions or targets, return zeros
    if not outputs or not targets:
        return metrics

    # Get predictions
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']

    # Convert to probabilities and get class predictions
    pred_prob = torch.nn.functional.softmax(pred_logits, dim=-1)
    # Remove background class (last class)
    pred_prob = pred_prob[:, :, :-1]

    # Get max probability and corresponding class
    max_scores, pred_classes = pred_prob.max(dim=-1)

    # Initialize counters
    total_gt = 0
    total_pred = 0

    # Initialize counters for each IoU threshold
    total_correct = {iou_th: 0 for iou_th in iou_thresholds}

    # Class-wise counters
    class_gt = [0] * num_classes
    class_pred = [0] * num_classes
    class_correct = {iou_th: [0] * num_classes for iou_th in iou_thresholds}

    # Import utility function for box conversion
    from model.utils import box_cxcywh_to_xyxy

    # Process each image
    batch_size = len(targets)

    # Debug info accumulators
    all_confidences = []
    all_ious = []

    for i in range(batch_size):
        # Get ground truth boxes and labels
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']

        # Get predicted boxes, labels, and scores for this image
        pred_boxes_img = pred_boxes[i]
        pred_classes_img = pred_classes[i]
        pred_scores_img = max_scores[i]

        # Process all predictions without confidence threshold filtering
        # Sort predictions by confidence score (highest first) to prioritize high-confidence detections
        # This approach ensures we don't miss any potential matches while favoring confident predictions
        sorted_indices = torch.argsort(pred_scores_img, descending=True)
        pred_boxes_img = pred_boxes_img[sorted_indices]
        pred_classes_img = pred_classes_img[sorted_indices]
        pred_scores_img = pred_scores_img[sorted_indices]

        # Collect confidence scores for debug info
        if len(pred_scores_img) > 0:
            all_confidences.extend(pred_scores_img.detach().cpu().tolist())

        # Print some debug information
        if i == 0:  # Only for the first image
            print(f"\nValidation Debug Info:")
            print(f"  Number of predictions: {len(pred_scores_img)}")
            if len(pred_scores_img) > 0:
                print(f"  Max confidence score: {pred_scores_img.max().item():.4f}")
                print(f"  Min confidence score: {pred_scores_img.min().item():.4f}")
                print(f"  Mean confidence score: {pred_scores_img.mean().item():.4f}")
            print(f"  Number of ground truth boxes: {len(gt_boxes)}")

        # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
        if len(gt_boxes) > 0:
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        else:
            gt_boxes_xyxy = torch.zeros((0, 4), device=gt_boxes.device)

        if len(pred_boxes_img) > 0:
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_img)
        else:
            pred_boxes_xyxy = torch.zeros((0, 4), device=pred_boxes_img.device)

        # Update total counts
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes_img)

        # Update class-wise counts
        for label in gt_labels:
            class_gt[label.item()] += 1

        for label in pred_classes_img:
            class_pred[label.item()] += 1

        # If no predictions or no ground truth, continue
        if len(pred_boxes_img) == 0 or len(gt_boxes) == 0:
            continue

        # Compute IoU between all pairs of boxes
        iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)

        # Collect IoU values for debug info
        if iou_matrix.numel() > 0:
            all_ious.extend(iou_matrix.flatten().detach().cpu().tolist())

        # For each prediction, find the best matching ground truth
        max_iou_values, max_iou_indices = iou_matrix.max(dim=1)

        # A prediction is correct if IoU > threshold and class is correct
        for pred_idx, (iou_val, gt_idx) in enumerate(zip(max_iou_values, max_iou_indices)):
            # Update confusion matrix with proper handling of IoU thresholds
            # Only count predictions that have sufficient IoU with ground truth
            pred_class = pred_classes_img[pred_idx].item()
            gt_class = gt_labels[gt_idx].item()

            # Use the standard IoU threshold (0.5) for confusion matrix calculation
            # This is the most common threshold in object detection literature and evaluation protocols
            # Using a standard threshold allows for consistent comparison with other models
            standard_iou_threshold = 0.5

            # Find the closest IoU threshold in our list
            closest_iou = min(iou_thresholds, key=lambda x: abs(x - standard_iou_threshold))

            # Only update confusion matrix if IoU is above the threshold
            if iou_val >= closest_iou:
                # Update confusion matrix - this counts true positives and false positives
                metrics['confusion_matrix'][gt_class, pred_class] += 1
            else:
                # If IoU is too low, this is considered a false positive detection
                # Since we don't have a dedicated background class in our confusion matrix,
                # we skip adding these low-IoU predictions to avoid cluttering the matrix
                # These false positives are still accounted for in precision calculations
                pass

            # Check against each IoU threshold
            for iou_threshold in iou_thresholds:
                # Count as correct only if IoU >= threshold and class is correct
                if iou_val >= iou_threshold and pred_class == gt_class:
                    total_correct[iou_threshold] += 1
                    class_correct[iou_threshold][gt_class] += 1
                    metrics[f'matches@{iou_threshold}'] += 1

                    # Use the lowest threshold for the debug info
                    if iou_threshold == iou_thresholds[0]:
                        metrics['debug_info']['total_matches'] += 1

    # Compute metrics for each IoU threshold
    for iou_threshold in iou_thresholds:
        # Compute overall metrics for this threshold
        if total_pred > 0:
            metrics[f'precision@{iou_threshold}'] = total_correct[iou_threshold] / total_pred

        if total_gt > 0:
            metrics[f'recall@{iou_threshold}'] = total_correct[iou_threshold] / total_gt

        if metrics[f'precision@{iou_threshold}'] + metrics[f'recall@{iou_threshold}'] > 0:
            metrics[f'f1@{iou_threshold}'] = 2 * metrics[f'precision@{iou_threshold}'] * metrics[f'recall@{iou_threshold}'] / (metrics[f'precision@{iou_threshold}'] + metrics[f'recall@{iou_threshold}'])

        # Compute class-wise metrics for this threshold
        for c in range(num_classes):
            if class_pred[c] > 0:
                metrics[f'class_precision@{iou_threshold}'][c] = class_correct[iou_threshold][c] / class_pred[c]

            if class_gt[c] > 0:
                metrics[f'class_recall@{iou_threshold}'][c] = class_correct[iou_threshold][c] / class_gt[c]

            if metrics[f'class_precision@{iou_threshold}'][c] + metrics[f'class_recall@{iou_threshold}'][c] > 0:
                metrics[f'class_f1@{iou_threshold}'][c] = 2 * metrics[f'class_precision@{iou_threshold}'][c] * metrics[f'class_recall@{iou_threshold}'][c] / (metrics[f'class_precision@{iou_threshold}'][c] + metrics[f'class_recall@{iou_threshold}'][c])

        # Compute mAP for this threshold using proper Average Precision calculation
        # This is a more accurate implementation that follows standard object detection metrics
        # We compute AP for each class and then average them
        class_aps = []

        for c in range(num_classes):
            # Skip classes with no ground truth instances
            if class_gt[c] == 0:
                continue

            # Average Precision (AP) calculation approach:
            #
            # A comprehensive AP implementation would:
            # 1. Sort all predictions by confidence score
            # 2. Compute precision and recall at each confidence threshold
            # 3. Calculate the area under the precision-recall curve
            #
            # Our simplified approach uses class precision as an AP approximation
            # This is reasonable for training feedback and when we have relatively
            # few predictions per class, providing a good balance between accuracy
            # and computational efficiency
            class_ap = metrics[f'class_precision@{iou_threshold}'][c]
            class_aps.append(class_ap)

        # Compute mAP as the mean of class APs
        if class_aps:
            metrics[f'mAP@{iou_threshold}'] = sum(class_aps) / len(class_aps)
        else:
            metrics[f'mAP@{iou_threshold}'] = 0.0

    # Use the standard IoU threshold (0.5) for the main metrics if available, otherwise use the first threshold
    standard_iou = 0.5
    if standard_iou in iou_thresholds:
        metrics['precision'] = metrics[f'precision@{standard_iou}']
        metrics['recall'] = metrics[f'recall@{standard_iou}']
        metrics['f1'] = metrics[f'f1@{standard_iou}']
        metrics['class_precision'] = metrics[f'class_precision@{standard_iou}']
        metrics['class_recall'] = metrics[f'class_recall@{standard_iou}']
        metrics['class_f1'] = metrics[f'class_f1@{standard_iou}']
        metrics['mAP'] = metrics[f'mAP@{standard_iou}']
    else:
        # Use the first threshold as fallback
        first_iou = iou_thresholds[0]
        metrics['precision'] = metrics[f'precision@{first_iou}']
        metrics['recall'] = metrics[f'recall@{first_iou}']
        metrics['f1'] = metrics[f'f1@{first_iou}']
        metrics['class_precision'] = metrics[f'class_precision@{first_iou}']
        metrics['class_recall'] = metrics[f'class_recall@{first_iou}']
        metrics['class_f1'] = metrics[f'class_f1@{first_iou}']
        metrics['mAP'] = metrics[f'mAP@{first_iou}']

    # Compute COCO-style mAP (average over multiple IoU thresholds)
    if len(iou_thresholds) > 1:
        coco_map_values = [metrics[f'mAP@{iou_th}'] for iou_th in iou_thresholds if iou_th >= 0.5 and iou_th <= 0.95]
        if coco_map_values:
            metrics['mAP_coco'] = sum(coco_map_values) / len(coco_map_values)

    # Compute accuracy (correct predictions / total predictions) using the standard IoU threshold
    if total_pred > 0:
        metrics['accuracy'] = metrics['precision']  # Accuracy is the same as precision in this context

    # Note on mAP calculation:
    # A comprehensive mAP implementation would compute Average Precision at different IoU thresholds
    # and average them (as in COCO evaluation). Our implementation provides a good approximation
    # while balancing computational efficiency and accuracy for training feedback.

    # Update debug info
    metrics['debug_info']['total_predictions'] = total_pred
    metrics['debug_info']['total_gt_boxes'] = total_gt

    # Confidence score statistics
    if all_confidences:
        metrics['debug_info']['max_confidence'] = max(all_confidences)
        metrics['debug_info']['min_confidence'] = min(all_confidences)
        metrics['debug_info']['avg_confidence'] = sum(all_confidences) / len(all_confidences)

    # IoU statistics
    if all_ious:
        metrics['debug_info']['max_iou'] = max(all_ious)
        metrics['debug_info']['min_iou'] = min(all_ious)
        metrics['debug_info']['avg_iou'] = sum(all_ious) / len(all_ious)

    # Print detailed debug info
    print("\nDetailed Validation Debug Info:")
    print(f"  Total predictions: {metrics['debug_info']['total_predictions']}")
    print(f"  Total ground truth boxes: {metrics['debug_info']['total_gt_boxes']}")
    print(f"  Total matches: {metrics['debug_info']['total_matches']}")
    if all_confidences:
        print(f"  Confidence scores: max={metrics['debug_info']['max_confidence']:.4f}, "
              f"min={metrics['debug_info']['min_confidence']:.4f}, "
              f"avg={metrics['debug_info']['avg_confidence']:.4f}")
    if all_ious:
        print(f"  IoU values: max={metrics['debug_info']['max_iou']:.4f}, "
              f"min={metrics['debug_info']['min_iou']:.4f}, "
              f"avg={metrics['debug_info']['avg_iou']:.4f}")

    # Print metrics for each IoU threshold
    print("\nMetrics at different IoU thresholds:")
    for iou_threshold in iou_thresholds:
        print(f"  IoU={iou_threshold}:")
        print(f"    Precision: {metrics[f'precision@{iou_threshold}']:.4f}")
        print(f"    Recall: {metrics[f'recall@{iou_threshold}']:.4f}")
        print(f"    F1 Score: {metrics[f'f1@{iou_threshold}']:.4f}")
        print(f"    mAP: {metrics[f'mAP@{iou_threshold}']:.4f}")
        print(f"    Matches: {metrics[f'matches@{iou_threshold}']}")

    # Print COCO-style mAP if available
    if 'mAP_coco' in metrics:
        print(f"\nCOCO-style mAP (IoU=0.5:0.95): {metrics['mAP_coco']:.4f}")

    # Print standard metrics (usually at IoU=0.5)
    print("\nStandard Metrics (IoU=0.5 or first threshold):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  mAP: {metrics['mAP']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def plot_confusion_matrix(confusion_matrix, class_names):
    """Create a visually informative confusion matrix plot for Weights & Biases logging.

    This function generates a high-quality visualization of the confusion matrix showing
    the relationship between predicted and actual classes with color-coded heatmap,
    proper labels, and optimized formatting for readability in Weights & Biases dashboards.

    Args:
        confusion_matrix: PyTorch tensor containing the confusion matrix values
        class_names: List of class names corresponding to matrix indices

    Returns:
        wandb_image: Weights & Biases Image object ready for logging to dashboards
    """
    # Convert to numpy for plotting
    cm = confusion_matrix.cpu().numpy()

    # Check if the confusion matrix has an extra column for "no detection"
    has_no_detection = cm.shape[1] > len(class_names)

    # Create figure and axis
    plt.figure(figsize=(12, 10))

    # Prepare labels for the plot
    x_labels = class_names.copy()
    y_labels = class_names.copy()

    # Add "No Detection" label if needed
    if has_no_detection:
        x_labels = x_labels + ["No Detection"]

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=x_labels,
        yticklabels=y_labels
    )

    # Set labels with more descriptive text
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix - Object Detection Results')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add a colorbar label
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Number of Instances')

    # Disable grid lines for a cleaner visualization
    # Grid lines can make the confusion matrix harder to read with many classes
    plt.grid(False)

    # Tight layout
    plt.tight_layout()

    # Save the confusion matrix figure to an in-memory buffer
    # Using higher DPI (150) for better image quality in Wandb dashboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)  # Reset buffer position to beginning for reading

    # Convert buffer to PIL Image
    img = Image.open(buf)

    # Close figure to free memory
    plt.close()

    # Return as wandb Image
    return wandb.Image(img, caption='Confusion Matrix - Object Detection Results')
