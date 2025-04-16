import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.utils import box_cxcywh_to_xyxy, generalized_box_iou, HungarianMatcher


class DeformableDETRLoss(nn.Module):
    """
    Loss function for Deformable DETR
    
    Args:
        num_classes: number of object classes
        matcher: module to match predictions to ground truth
        weight_dict: dict containing weights for different losses
        eos_coef: weight for background class (no-object)
        losses: list of losses to compute
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, losses=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses if losses is not None else ['labels', 'boxes', 'cardinality']
        
        # Parameters for Focal Loss to address class imbalance
        # Focal Loss focuses more on hard-to-predict examples
        self.alpha = 0.25  # Balance between positive/negative examples
        self.gamma = 2.0   # Focus factor for easy/hard examples
        
        # Weights based on the class distribution in the AU-AIR dataset
        # Category counts from dataset_exploration/category_distribution.csv
        class_counts = {
            0: 5158,    # Human
            1: 102619,  # Car
            2: 9545,    # Truck
            3: 9995,    # Van
            4: 319,     # Motorbike
            5: 1128,    # Bicycle
            6: 729,     # Bus
            7: 2538     # Trailer
        }
        
        # Effective number of samples calculation for class balanced loss
        # Formula: (1 - beta^n) / (1 - beta), beta=0.9999
        beta = 0.9999
        effective_nums = {k: (1.0 - beta**v) / (1.0 - beta) for k, v in class_counts.items()}
        
        # Inverse frequency weighting
        total_count = sum(class_counts.values())
        inv_freq = {k: total_count / v for k, v in class_counts.items()}
        
        # Normalize weights
        max_inv_freq = max(inv_freq.values())
        norm_inv_freq = {k: v / max_inv_freq for k, v in inv_freq.items()}
        
        # Calculate class weights (Effective Number + Inverse Frequency Combined)
        class_weight_dict = {}
        for k in range(num_classes):
            if k in effective_nums:
                # 10x weight for the rarest class (Motorbike - 4),
                # 1x weight for the most common class (Car - 1)
                # Proportional weights for other classes
                class_weight_dict[k] = norm_inv_freq[k] * (effective_nums[4] / effective_nums[k])
            else:
                class_weight_dict[k] = 1.0
        
        # Convert class weights to tensor
        empty_weight = torch.ones(self.num_classes + 1)
        for k, v in class_weight_dict.items():
            empty_weight[k] = v
        
        # Set the weight for the background class
        empty_weight[-1] = self.eos_coef
        
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (using focal loss with class balancing)
        """
        assert 'pred_logits' in outputs
        device = outputs['pred_logits'].device
        
        # Get matched indices
        src_idx = self._get_src_permutation_idx(indices, device)
        tgt_idx = self._get_tgt_permutation_idx(indices, device)
        
        # Check if we have any matches
        if src_idx[0].numel() == 0:
            # No matches, create a zero tensor that requires grad
            return {'loss_ce': outputs['pred_logits'].sum() * 0.0}
        
        # Extract target classes
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices) if len(J) > 0])
        
        # Initialize all targets as background class (num_classes)
        target_classes = torch.full(outputs['pred_logits'].shape[:2], self.num_classes,
                                   dtype=torch.int64, device=device)
        
        # Assign matched target classes to source indices
        if len(target_classes_o) > 0:
            target_classes[src_idx] = target_classes_o
        
        # Fix any negative indices (replacing -1 with background class)
        target_classes = torch.clamp(target_classes, min=0)
        
        # Implement focal loss with class balanced weights
        
        # One-hot encoding of target classes
        target_classes_onehot = torch.zeros(outputs['pred_logits'].shape, 
                                           dtype=outputs['pred_logits'].dtype,
                                           device=device)
        
        # Extract actual number of classes (without background)
        num_classes = outputs['pred_logits'].shape[-1]
        
        # Safely fill the one-hot tensor
        batch_size, num_queries = target_classes.shape
        for b in range(batch_size):
            for q in range(num_queries):
                cls_idx = target_classes[b, q].item()
                if cls_idx < num_classes:  # Ensure valid class index
                    target_classes_onehot[b, q, cls_idx] = 1.0
        
        # Compute class-balanced focal loss with weights from init
        # Directly use logits (BEFORE sigmoid) for mixed precision compatibility
        pred_logits = outputs['pred_logits']
        
        # Compute focal loss with class balancing (pass logits directly)
        focal_loss = self._focal_loss(pred_logits, target_classes_onehot, 
                                     alpha=self.alpha, gamma=self.gamma)
        
        # Apply class weights
        target_classes_for_weights = torch.clamp(target_classes, max=self.empty_weight.shape[0]-1)
        
        # Ensure device compatibility: empty_weight and target_classes_for_weights should be on the same device
        empty_weight_device = self.empty_weight.device
        if target_classes_for_weights.device != empty_weight_device:
            target_classes_for_weights = target_classes_for_weights.to(empty_weight_device)
            
        class_weights = self.empty_weight[target_classes_for_weights]
        
        # Move class_weights to the same device as focal_loss (if different)
        if class_weights.device != focal_loss.device:
            class_weights = class_weights.to(focal_loss.device)
        
        # Weight the loss by class weights
        weighted_loss = focal_loss * class_weights.unsqueeze(-1)
        
        # Normalize by number of boxes
        class_loss = weighted_loss.sum() / (num_boxes + 1e-8)  # Avoid division by zero
        
        losses = {'loss_ce': class_loss}
        
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Box regression loss (L1 and GIoU)
        """
        assert 'pred_boxes' in outputs
        device = outputs['pred_boxes'].device
        
        # Get matched indices
        idx = self._get_src_permutation_idx(indices, device)
        
        # Check if we have any matches or no boxes
        if idx[0].numel() == 0 or num_boxes.item() <= 0:
            # Return zero losses that require grad
            return {
                'loss_bbox': outputs['pred_boxes'].sum() * 0.0,
                'loss_giou': outputs['pred_boxes'].sum() * 0.0
            }
        
        # Extract target boxes - only from indices with matches
        matched_targets = [t['boxes'][i] for t, (_, i) in zip(targets, indices) if len(i) > 0]
        
        # If no matched targets, return zero losses
        if not matched_targets:
            return {
                'loss_bbox': outputs['pred_boxes'].sum() * 0.0,
                'loss_giou': outputs['pred_boxes'].sum() * 0.0
            }
        
        target_boxes = torch.cat(matched_targets, dim=0)
        
        # Get predicted boxes for matching elements
        src_boxes = outputs['pred_boxes'][idx]
        
        # Get target classes (for class-aware box loss scaling)
        target_classes = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices) if len(i) > 0])
        
        # Compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        # Compute class-specific box loss scaling based on object sizes
        # Scale box loss by inverse of class frequency for better detection of small objects
        class_scales = torch.ones_like(target_classes, dtype=torch.float, device=device)
        
        # Scale up loss for rare classes that often have smaller objects
        # Motorbike (4), Bicycle (5), Bus (6) often have smaller boxes
        rare_class_indices = torch.tensor([4, 5, 6], device=device)
        scale_factor = 2.0  # Boost the loss for rare classes
        
        for class_idx in rare_class_indices:
            # Check device compatibility
            if target_classes.device != class_idx.device:
                class_idx = class_idx.to(target_classes.device)
            class_scales[target_classes == class_idx] = scale_factor
            
        # Apply class scales to each box dimension
        box_loss_scaling = class_scales.unsqueeze(1).expand_as(loss_bbox)
        loss_bbox = loss_bbox * box_loss_scaling
        
        # Normalize by box dimension
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / (num_boxes + 1e-8)  # Avoid division by zero
        
        # Compute GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        
        # Apply same class scaling to GIoU loss
        loss_giou = loss_giou * class_scales
        
        losses['loss_giou'] = loss_giou.sum() / (num_boxes + 1e-8)  # Avoid division by zero
        
        return losses
    
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute cardinality error - prediction count vs target count
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        # Count number of target boxes in each image
        target_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        
        # Shape: [batch_size, num_queries, num_classes] -> [batch_size]
        # Sigmoid output: 0-1 probability per class
        # First apply sigmoid and find the most likely class for each query
        probs = pred_logits.sigmoid()
        
        # Find the highest class probability for each query
        max_probs, _ = probs.max(dim=-1)  # [batch_size, num_queries]
        
        # Count queries above threshold (as positive predictions)
        card_pred = (max_probs > 0.5).sum(dim=1).float()  # [batch_size]
        
        # Ensure target_lengths is float
        target_lengths = target_lengths.float()
        
        # L1 loss between predicted and target cardinality
        # In this line, card_pred and target_lengths should have the same shape: [batch_size]
        card_err = F.l1_loss(card_pred, target_lengths)
        
        losses = {'loss_cardinality': card_err}
        
        return losses
    
    def _focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """
        Enhanced Focal Loss calculation (Mixed precision compatible)
        
        Args:
            inputs: Model predictions (LOGITS - values BEFORE sigmoid)
            targets: Target classes (one-hot encoded)
            alpha: Positive/negative sample balance
            gamma: Focus factor - higher values focus more on hard examples
            
        Returns:
            Focal loss value
        """
        # Use Binary Cross Entropy with Logits to ensure mixed precision compatibility
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply sigmoid (mixed precision-friendly)
        pred_prob = torch.sigmoid(inputs)
        
        # Calculate pt (probability of correct prediction)
        pt = torch.where(targets == 1, pred_prob, 1 - pred_prob)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** gamma
        
        # Apply alpha weight (class balance)
        alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)
        
        # Total weight
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss
    
    def _get_src_permutation_idx(self, indices, device=None):
        """
        Get permutation indices for the source and target indices.
        """
        # Handle empty src indices
        batch_idx = []
        src_idx = []
        
        for i, (src, _) in enumerate(indices):
            if len(src) > 0:  # Check if src is not empty
                # Convert src to tensor if it's not already
                if not isinstance(src, torch.Tensor):
                    src = torch.tensor(src, dtype=torch.long, device=device)
                elif device is not None:
                    src = src.to(device)
                
                batch_idx.append(torch.full_like(src, i))
                src_idx.append(src)
        
        if not batch_idx:  # If all src indices are empty, return empty tensors
            return (torch.tensor([], device=device, dtype=torch.long),
                    torch.tensor([], device=device, dtype=torch.long))
        
        batch_idx = torch.cat(batch_idx)
        src_idx = torch.cat(src_idx)
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices, device=None):
        """
        Get permutation indices for the source and target indices.
        """
        # Handle empty tgt indices
        batch_idx = []
        tgt_idx = []
        
        for i, (_, tgt) in enumerate(indices):
            if len(tgt) > 0:  # Check if tgt is not empty
                # Convert tgt to tensor if it's not already
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.tensor(tgt, dtype=torch.long, device=device)
                elif device is not None:
                    tgt = tgt.to(device)
                
                batch_idx.append(torch.full_like(tgt, i))
                tgt_idx.append(tgt)
        
        if not batch_idx:  # If all tgt indices are empty, return empty tensors
            return (torch.tensor([], device=device, dtype=torch.long),
                    torch.tensor([], device=device, dtype=torch.long))
        
        batch_idx = torch.cat(batch_idx)
        tgt_idx = torch.cat(tgt_idx)
        return batch_idx, tgt_idx
    
    def forward(self, outputs, targets):
        """
        Main forward function for calculating all losses
        
        Args:
            outputs: dict of model outputs
            targets: list of dicts containing ground truth
            
        Returns:
            losses: dict of losses
        """
        # Get device from outputs
        device = outputs['pred_logits'].device
        
        # Match predictions to targets
        indices = self.matcher(outputs, targets)
        
        # Count number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        
        # Compute all loss components
        losses = {}
        for loss in self.losses:
            loss_func = getattr(self, f'loss_{loss}')
            loss_dict = loss_func(outputs, targets, indices, num_boxes)
            losses.update(loss_dict)
        
        # Ensure all losses require gradients by adding a small operation with model outputs
        for k, v in losses.items():
            if not v.requires_grad:
                losses[k] = v + outputs['pred_logits'].sum() * 0.0
        
        # Weight losses
        weighted_losses = {k: self.weight_dict[k] * v if k in self.weight_dict else v for k, v in losses.items()}
        
        # Ensure final sum for backward pass has gradient information
        total_loss = sum(weighted_losses.values())
        if not total_loss.requires_grad:
            dummy_loss = outputs['pred_logits'].sum() * 0.0
            total_loss = total_loss + dummy_loss
        
        # Return weighted losses
        return weighted_losses 