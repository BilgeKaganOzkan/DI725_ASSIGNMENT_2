"""
Utility functions for object detection models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any, Iterable
import matplotlib.pyplot as plt

# Try to import seaborn, but don't fail if it's not available
try:
    import seaborn as sns
except ImportError:
    # Define a simple replacement for sns.heatmap if seaborn is not available
    class SeabornReplacement:
        def heatmap(self, data, annot=True, fmt='d', cmap='Blues', xticklabels=None, yticklabels=None):
            fig, ax = plt.subplots()
            im = ax.imshow(data, cmap=cmap)
            if annot:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(j, i, format(data[i, j], fmt), ha="center", va="center")
            if xticklabels is not None:
                ax.set_xticks(range(len(xticklabels)))
                ax.set_xticklabels(xticklabels)
            if yticklabels is not None:
                ax.set_yticks(range(len(yticklabels)))
                ax.set_yticklabels(yticklabels)
            return im
    sns = SeabornReplacement()
import time
import datetime
import math
from collections import defaultdict, deque


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from center-based to corner-based format

    This function transforms bounding box coordinates from center-based format
    (center_x, center_y, width, height) to corner-based format (x1, y1, x2, y2),
    which is required for many operations like IoU calculation and visualization.

    The function includes robust handling for various edge cases:
    - Non-tensor inputs are automatically converted to tensors
    - Empty tensors are handled gracefully
    - Width and height values are forced to be positive

    Args:
        x: Tensor or array-like of shape (..., 4) containing bounding boxes in (cx, cy, w, h) format

    Returns:
        Tensor of same shape containing bounding boxes in (x1, y1, x2, y2) format
    """
    # Handle non-tensor inputs
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Handle empty tensors
    if x.numel() == 0:
        return x.reshape(-1, 4)

    # Extract components
    x_c, y_c, w, h = x.unbind(-1)

    # Ensure width and height are positive
    w = torch.abs(w)
    h = torch.abs(h)

    # Compute corners
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h

    # Stack and return
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from corner-based to center-based format

    This function transforms bounding box coordinates from corner-based format
    (x1, y1, x2, y2) to center-based format (center_x, center_y, width, height),
    which is the preferred format for the Deformable DETR model.

    The function includes robust handling for various edge cases:
    - Non-tensor inputs are automatically converted to tensors
    - Empty tensors are handled gracefully
    - Coordinates are automatically sorted to ensure x1 ≤ x2 and y1 ≤ y2

    Args:
        x: Tensor or array-like of shape (..., 4) containing bounding boxes in (x1, y1, x2, y2) format

    Returns:
        Tensor of same shape containing bounding boxes in (cx, cy, w, h) format
    """
    # Handle non-tensor inputs
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Handle empty tensors
    if x.numel() == 0:
        return x.reshape(-1, 4)

    # Extract components
    x0, y0, x1, y1 = x.unbind(-1)

    # Ensure correct order (x1 <= x2, y1 <= y2)
    x0_new = torch.min(x0, x1)
    y0_new = torch.min(y0, y1)
    x1_new = torch.max(x0, x1)
    y1_new = torch.max(y0, y1)

    # Compute center and dimensions
    cx = (x0_new + x1_new) / 2
    cy = (y0_new + y1_new) / 2
    w = x1_new - x0_new
    h = y1_new - y0_new

    # Stack and return
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between all pairs of bounding boxes

    This function efficiently calculates the IoU between all pairs of boxes from two sets,
    using vectorized operations for better performance. IoU is a critical metric for
    object detection that measures the overlap between predicted and ground truth boxes.

    The calculation follows these steps:
    1. Compute areas of all boxes in both sets
    2. Find the coordinates of intersection rectangles
    3. Calculate intersection areas
    4. Calculate union areas (sum of individual areas minus intersection)
    5. Compute IoU as intersection / union

    A small epsilon (1e-6) is added to the denominator to prevent division by zero.

    Args:
        boxes1: Tensor of shape (N, 4) containing N boxes in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) containing M boxes in (x1, y1, x2, y2) format

    Returns:
        Tensor of shape (N, M) containing IoU values for all pairs of boxes
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou


def box_area(boxes):
    """
    Compute the area of bounding boxes efficiently

    This function calculates the area of each bounding box in a batch by multiplying
    the width and height. It works with batches of any shape as long as the last
    dimension contains the box coordinates in (x1, y1, x2, y2) format.

    The implementation uses PyTorch's broadcasting capabilities to efficiently
    handle batches of boxes in a single operation.

    Args:
        boxes: Tensor of shape (..., 4) containing boxes in (x1, y1, x2, y2) format

    Returns:
        Tensor of shape (...) containing the area of each box
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU (GIoU) between pairs of bounding boxes

    Generalized IoU is an improved version of the standard IoU metric that better
    handles non-overlapping boxes. It penalizes predictions that are far from the
    ground truth even when there is no overlap, making it a more effective loss
    function for object detection training.

    The GIoU is calculated as:
    GIoU = IoU - (area_of_enclosing_box - union) / area_of_enclosing_box

    This results in a value between -1 and 1, where:
    - 1 indicates perfect overlap (same as IoU=1)
    - 0 indicates no overlap with boxes touching
    - Negative values indicate the degree of separation between non-overlapping boxes

    Args:
        boxes1: Tensor of shape (N, 4) containing N boxes in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) containing M boxes in (x1, y1, x2, y2) format

    Returns:
        Tensor of shape (N, M) containing generalized IoU values for each pair of boxes
    """
    # Standard IoU
    iou = box_iou(boxes1, boxes2)

    # Get the coordinates of bounding boxes
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Compute union
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - iou * (area1[:, None] + area2 - area1[:, None] * area2)

    # Compute GIoU
    giou = iou - (area - union) / (area + 1e-6)

    return giou


class HungarianMatcher:
    """
    Optimal bipartite matching between predictions and ground truth using Hungarian algorithm

    This class implements the core matching logic for Deformable DETR, which assigns
    each ground truth box to exactly one prediction. The matching is optimal in the sense
    that it minimizes the total assignment cost across all possible assignments.

    The matching cost combines three components:
    1. Classification cost: Negative log-likelihood between predicted class probabilities
       and target classes
    2. L1 distance cost: L1 distance between predicted and target box coordinates
    3. GIoU cost: Negative generalized IoU between predicted and target boxes

    These costs can be weighted differently using the cost_class, cost_bbox, and cost_giou
    parameters to emphasize different aspects of the prediction quality.
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        """
        Initialize the matcher with configurable cost weights

        The weights control the relative importance of different components in the
        matching cost calculation. Higher weights make that component more important
        in determining the final matching.

        Optimized values for better detection performance:
        - cost_class=1.0: Standard weight for classification cost
        - cost_bbox=5.0: Higher weight for box coordinate accuracy
        - cost_giou=2.0: Medium weight for box overlap quality

        Args:
            cost_class: Weight for classification cost (negative log-likelihood)
            cost_bbox: Weight for bounding box L1 distance cost
            cost_giou: Weight for generalized IoU cost
        """
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(self, outputs, targets):
        """
        Match predictions to targets using the Hungarian algorithm

        This method performs the following steps:
        1. Extracts prediction logits and boxes from the model outputs
        2. Computes classification cost using negative log-likelihood
        3. Computes L1 distance between predicted and target boxes
        4. Computes GIoU between predicted and target boxes
        5. Combines these costs using the configured weights
        6. Solves the resulting assignment problem using the Hungarian algorithm

        The matching is performed independently for each image in the batch.

        Args:
            outputs: Dictionary containing model outputs with keys:
                     - 'pred_logits': Classification logits [batch_size, num_queries, num_classes+1]
                     - 'pred_boxes': Predicted boxes [batch_size, num_queries, 4] in (cx, cy, w, h) format
            targets: List of dictionaries (one per image) with keys:
                     - 'labels': Ground truth class indices [num_objects]
                     - 'boxes': Ground truth boxes [num_objects, 4] in (cx, cy, w, h) format

        Returns:
            List of tuples (pred_indices, target_indices) for each image in the batch,
            where pred_indices and target_indices are tensors of indices that form the
            optimal bipartite matching
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # Apply temperature scaling for better probability distribution
        temperature = 1.5  # Temperature > 1 makes distribution softer
        out_prob = (outputs["pred_logits"] / temperature).flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost
        # Negative log likelihood between predictions and targets
        # Apply focal-like weighting to focus more on hard examples
        alpha = 0.25
        gamma = 2.0
        probs = out_prob[:, tgt_ids]
        focal_weight = alpha * (1 - probs) ** gamma
        cost_class = -focal_weight * torch.log(probs + 1e-8)

        # Compute the L1 cost between boxes
        # Scale the cost to be more balanced with classification cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Apply scaling to prevent extremely high values
        cost_bbox = torch.clamp(cost_bbox, max=10.0)

        # Compute the GIoU cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix with balanced weights
        C = (
            self.cost_bbox * cost_bbox +
            self.cost_class * cost_class +
            self.cost_giou * cost_giou
        )

        # Apply additional penalty for matching with predictions that have low confidence
        # This helps reduce false positives
        confidence_penalty = -torch.log(out_prob.max(dim=1)[0]).unsqueeze(1).expand_as(C)
        C = C + 0.1 * confidence_penalty

        # Reshape cost matrix to have batch dimension
        C = C.view(batch_size, num_queries, -1)

        # Get number of targets for each batch element
        sizes = [len(v["boxes"]) for v in targets]

        # Split the cost matrix by batch element
        indices = [
            linear_sum_assignment(c[i].cpu().detach().numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]

        # Convert numpy arrays to torch tensors
        return [
            (
                torch.as_tensor(i, dtype=torch.int64, device=out_prob.device),
                torch.as_tensor(j, dtype=torch.int64, device=out_prob.device)
            )
            for i, j in indices
        ]


def linear_sum_assignment(cost_matrix):
    """
    Solve the linear sum assignment problem with multiple fallback implementations

    This function implements a robust approach to solving the assignment problem by
    trying multiple implementations in order of efficiency:

    1. First attempts to use scipy's highly optimized implementation
    2. If scipy is unavailable, tries to use networkx for medium to large matrices
    3. For small matrices (≤10x10), uses a simple greedy approach for speed
    4. For larger matrices without scipy/networkx, uses greedy with a warning

    This multi-tiered approach ensures the function works in various environments
    while maintaining reasonable performance characteristics.

    Args:
        cost_matrix: 2D numpy array where cost_matrix[i,j] is the cost of assigning
                     row i to column j

    Returns:
        Tuple of (row_indices, col_indices) giving the optimal assignment that
        minimizes the total cost
    """
    try:
        # Always try to use scipy's highly optimized implementation first
        from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
        return scipy_linear_sum_assignment(cost_matrix)
    except ImportError:
        # If scipy is not available, use our own implementation
        import numpy as np
        x = np.array(cost_matrix)

        # Handle empty matrices
        if x.shape[0] == 0 or x.shape[1] == 0:
            return np.array([]), np.array([])

        # For very small matrices, use a simple greedy approach
        if x.shape[0] <= 10 and x.shape[1] <= 10:
            return _greedy_assignment(x)

        # For larger matrices, use a more efficient algorithm
        # Import networkx if available (much faster than our implementation)
        try:
            import networkx as nx
            return _networkx_assignment(x)
        except ImportError:
            # Fall back to greedy for medium-sized matrices
            if x.shape[0] <= 50 and x.shape[1] <= 50:
                return _greedy_assignment(x)
            else:
                # For larger matrices, print a warning and use greedy anyway
                print("Warning: Using slow assignment algorithm for large matrix. "
                      "Install scipy or networkx for better performance.")
                return _greedy_assignment(x)


def _greedy_assignment(cost_matrix):
    """Simple greedy assignment algorithm for small matrices."""
    import numpy as np
    x = np.array(cost_matrix)
    row_ind, col_ind = [], []
    mask = np.ones(x.shape, dtype=bool)

    # Assign each row to its minimum cost column that's still available
    for _ in range(min(x.shape[0], x.shape[1])):
        i, j = np.unravel_index(np.argmin(np.where(mask, x, np.inf)), x.shape)
        row_ind.append(i)
        col_ind.append(j)
        mask[i, :] = False
        mask[:, j] = False

    return np.array(row_ind), np.array(col_ind)


def _networkx_assignment(cost_matrix):
    """Use networkx's implementation of the Hungarian algorithm."""
    import numpy as np
    import networkx as nx

    # Create a bipartite graph
    n, m = cost_matrix.shape
    G = nx.Graph()

    # Add nodes with bipartite attribute
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n+m), bipartite=1)

    # Add edges with weights from cost matrix
    for i in range(n):
        for j in range(m):
            G.add_edge(i, n+j, weight=cost_matrix[i, j])

    # Find minimum weight matching
    matching = nx.algorithms.matching.min_weight_matching(G)

    # Extract row and column indices
    row_ind, col_ind = [], []
    for i, j in matching:
        if i < n:  # i is a row index
            row_ind.append(i)
            col_ind.append(j - n)
        else:  # j is a row index
            row_ind.append(j)
            col_ind.append(i - n)

    return np.array(row_ind), np.array(col_ind)


def plot_confusion_matrix(confusion_matrix, class_names):
    """Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: Tensor of shape (num_classes, num_classes)
        class_names: List of class names

    Returns:
        Matplotlib figure object for visualization
    """
    # Convert to numpy array
    cm = confusion_matrix.cpu().numpy()

    # Create figure and axis
    plt.figure(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Close figure to avoid memory leaks
    plt.close()

    # Return image
    return buf


# Add missing import
import io
import os
import sys


class SmoothedValue:
    """
    Track and smooth a series of values for stable metric reporting

    This class maintains a running history of values in a fixed-size window and
    provides various statistical aggregations (median, mean, global average).
    It's particularly useful for training metrics that can be noisy, providing
    smoothed values that give a better indication of trends.

    Key features:
    - Maintains a deque of recent values with configurable window size
    - Tracks global sum and count for accurate long-term averages
    - Provides multiple statistical views of the data (median, mean, max, etc.)
    - Supports custom formatting of output values
    - Optional distributed training synchronization
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque!"""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger:
    """
    Comprehensive metric tracking and logging system for training and validation

    This class provides a centralized way to track multiple metrics during model training
    and validation. It maintains a collection of SmoothedValue objects for each metric,
    allowing for consistent handling and reporting of various statistics.

    Key features:
    - Tracks multiple metrics simultaneously with automatic smoothing
    - Handles tensor values by automatically extracting scalar values
    - Provides formatted string representation of all tracked metrics
    - Supports distributed training with proper metric synchronization
    - Allows dynamic addition of new metrics during training
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_num_parameters(model):
    """Get number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
