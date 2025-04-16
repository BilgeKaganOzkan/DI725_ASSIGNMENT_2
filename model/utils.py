import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import time
from collections import defaultdict, deque
import datetime
import torch.distributed as dist


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from (center_x, center_y, width, height) format to (min_x, min_y, max_x, max_y) format.
    
    Args:
        x (torch.Tensor): Bounding boxes in (cx, cy, w, h) format, shape (N, 4)
        
    Returns:
        torch.Tensor: Bounding boxes in (min_x, min_y, max_x, max_y) format, shape (N, 4)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from (min_x, min_y, max_x, max_y) format to (center_x, center_y, width, height) format.
    
    Args:
        x (torch.Tensor): Bounding boxes in (min_x, min_y, max_x, max_y) format, shape (N, 4)
        
    Returns:
        torch.Tensor: Bounding boxes in (center_x, center_y, width, height) format, shape (N, 4)
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Compute the generalized IoU between two sets of boxes.
    
    Args:
        boxes1: first set of boxes (N, 4)
        boxes2: second set of boxes (M, 4)
        
    Returns:
        giou: generalized IoU (N, M)
    """
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    # Calculate IoU
    area1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=1)
    area2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=1)
    
    # Get the coordinates of the intersection rectangle
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]
    
    # Check if there's an intersection
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Calculate union
    union = area1[:, None] + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    # Calculate the coordinates of the smallest enclosing box
    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # left-top of enclosing box
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom of enclosing box
    
    # Calculate area of the enclosing box
    wh_c = (rb_c - lt_c).clamp(min=0)  # [N,M,2]
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]  # [N,M]
    
    # Calculate GIoU
    giou = iou - (area_c - union) / area_c
    
    return giou


class NestedTensor(object):
    """
    Utility class for handling tensors with padding.
    """
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # Move tensors to the specified device
        return NestedTensor(self.tensors.to(device), self.mask.to(device))

    def decompose(self):
        return self.tensors, self.mask


class Preprocessor:
    """
    Preprocessing for Deformable DETR model
    """
    def __init__(self, 
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 max_size=1333):
        self.mean = mean
        self.std = std
        self.max_size = max_size
        
        # Transforms for training
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(sizes=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], max_size=max_size),
            T.Normalize(mean=mean, std=std)
        ])
        
        # Transforms for validation/testing
        self.val_transforms = T.Compose([
            T.RandomResize(sizes=[800], max_size=max_size),
            T.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, img, target=None, is_train=True):
        """
        Apply transformation to image and targets
        
        Args:
            img: PIL image
            target: dict with fields 'boxes', 'labels'
            is_train: whether to apply training transforms or validation transforms
            
        Returns:
            img: tensor image
            target: transformed target
        """
        transforms = self.train_transforms if is_train else self.val_transforms
        
        if target is not None:
            # Apply transformations to both image and target
            img, target = transforms(img, target)
            return img, target
        else:
            # Apply transformations to image only
            img = transforms(img)
            return img


def create_position_embedding(hidden_dim, spatial_shape, device=None):
    """
    Create 2D sine-cosine positional embeddings.
    
    Args:
        hidden_dim: embedding dimension
        spatial_shape: tuple of spatial dimensions (height, width)
        device: device to create the embeddings on
        
    Returns:
        pos_embed: positional embeddings of shape (hidden_dim, height, width)
    """
    height, width = spatial_shape
    
    # Generate grid of positions
    y_embed, x_embed = torch.meshgrid(torch.arange(height, device=device),
                                     torch.arange(width, device=device),
                                     indexing='ij')
    
    if hidden_dim % 4 != 0:
        raise ValueError("Hidden dimension must be divisible by 4 for 2D position embedding")
    
    temperature = 10000.0
    dim_t = torch.arange(hidden_dim // 4, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / (hidden_dim // 4))
    
    # Position embeddings for x and y dimensions
    pos_x = x_embed.flatten()[..., None] / dim_t
    pos_y = y_embed.flatten()[..., None] / dim_t
    
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(-2)
    
    # Combine embeddings
    pos = torch.cat((pos_y, pos_x), dim=-1)
    pos = pos.reshape(height, width, hidden_dim).permute(2, 0, 1)
    
    return pos


def get_num_parameters(model):
    """
    Get the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


class HungarianMatcher(nn.Module):
    """
    Matcher based on Hungarian algorithm for Deformable DETR
    
    This matches predicted boxes to target boxes using a weighted sum of class prediction loss,
    L1 box distance, and Generalized IoU.
    
    Args:
        cost_class: weight for class prediction loss
        cost_bbox: weight for L1 box distance
        cost_giou: weight for Generalized IoU loss
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing:
                'pred_logits': torch.Tensor of shape (batch_size, num_queries, num_classes)
                'pred_boxes': torch.Tensor of shape (batch_size, num_queries, 4)
            targets: list of dicts containing:
                'labels': torch.Tensor of shape (num_target_boxes,)
                'boxes': torch.Tensor of shape (num_target_boxes, 4)
                
        Returns:
            indices: list of tuples (pred_idx, tgt_idx) containing the indices of matched predictions and targets
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # List of indices for each batch element
        indices = []
        
        # Ensure targets list is at least as long as the batch size
        if len(targets) < bs:
            # Pad targets list with empty targets if needed
            device = outputs["pred_logits"].device
            for _ in range(bs - len(targets)):
                # Create an empty target with the right device
                empty_target = {
                    "labels": torch.tensor([], dtype=torch.int64, device=device),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device)
                }
                targets.append(empty_target)
        
        # Process each batch element separately
        for b in range(bs):
            try:
                # Skip if target is missing or invalid
                if b >= len(targets) or "labels" not in targets[b] or "boxes" not in targets[b]:
                    empty_inds = ([], [])
                    indices.append(empty_inds)
                    continue
                
                tgt_ids = targets[b]["labels"]
                tgt_bbox = targets[b]["boxes"]
                
                # Skip if no targets for this batch element
                if tgt_ids.shape[0] == 0:
                    indices.append(([], []))
                    continue
                
                # Class cost: -log(p) where p is the probability of the correct class
                # Shape: [num_queries, num_target_boxes]
                cost_class = -out_prob[b * num_queries: (b + 1) * num_queries, tgt_ids]
                
                # Compute L1 cost between boxes
                # Shape: [num_queries, num_target_boxes]
                cost_bbox = torch.cdist(out_bbox[b * num_queries: (b + 1) * num_queries], tgt_bbox, p=1)
                
                # Compute GIoU cost between boxes
                # Shape: [num_queries, num_target_boxes]
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox[b * num_queries: (b + 1) * num_queries]),
                    box_cxcywh_to_xyxy(tgt_bbox)
                )
                
                # Final cost matrix
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                
                # Use Hungarian algorithm to find optimal assignment
                from scipy.optimize import linear_sum_assignment
                src_idx, tgt_idx = linear_sum_assignment(C.cpu().numpy())
                
                # Convert numpy arrays to torch tensors
                device = outputs["pred_logits"].device
                src_idx = torch.tensor(src_idx, dtype=torch.long, device=device)
                tgt_idx = torch.tensor(tgt_idx, dtype=torch.long, device=device)
                
                # Add batch element index to the indices
                indices.append((src_idx, tgt_idx))
            
            except Exception as e:
                print(f"Error processing batch element {b}: {e}")
                # Return empty indices for this batch element
                indices.append(([], []))
            
        return indices


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
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
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
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
        return self.deque[-1] if len(self.deque) > 0 else None

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger:
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
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
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
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict 