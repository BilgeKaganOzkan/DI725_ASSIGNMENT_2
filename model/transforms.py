import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

from model.utils import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                width, _ = image.size
                target["boxes"][:, 0] = width - target["boxes"][:, 0]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = random.choice(self.sizes)
        return resize(image, target, size, self.max_size)


class RandomSelect:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class RandomSizeCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        w = random.randint(self.min_size, min(image.width, self.max_size))
        h = random.randint(self.min_size, min(image.height, self.max_size))
        region = T.RandomCrop.get_params(image, [h, w])
        return crop(image, target, region)


class Resize:
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, image, target=None):
        return resize(image, target, self.size, self.max_size)


def resize(image, target, size, max_size=None):
    # Size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size
        else:
            # Size is a scalar
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        if len(boxes) > 0:
            # Convert to xyxy
            x_c, y_c, w, h = boxes.unbind(1)
            boxes = torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=1)
            
            # Resize boxes
            boxes = boxes * torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32)
            
            # Convert back to cxcywh
            x0, y0, x1, y1 = boxes.unbind(1)
            boxes = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=1)
            
            target["boxes"] = boxes

    if "area" in target:
        area = target["area"]
        target["area"] = area * ratio_width * ratio_height

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def crop(image, target, region):
    i, j, h, w = region
    cropped_image = F.crop(image, i, j, h, w)

    if target is None:
        return cropped_image, None

    target = target.copy()
    # Crop boxes
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        
        # Keep only boxes with positive area
        keep = torch.where(area > 0)[0]
        if len(keep) < len(area):
            for key in ["boxes", "labels", "area"]:
                if key in target:
                    target[key] = target[key][keep]

    cropped_image_size = cropped_image.size
    target["size"] = torch.tensor([cropped_image_size[1], cropped_image_size[0]])
    
    return cropped_image, target


def create_training_transforms(scales, max_size):
    """
    Create training transforms with more consistent sizing
    """
    return Compose([
        RandomHorizontalFlip(),
        RandomSelect(
            RandomResize(scales, max_size=max_size),
            Compose([
                RandomResize([400, 500, 600]),
                RandomSizeCrop(384, 600),
                RandomResize([800], max_size=max_size),  # Fix the final size to be more consistent
            ])
        ),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def create_validation_transforms(size, max_size):
    """
    Create validation transforms with fixed sizing
    """
    return Compose([
        RandomResize([size], max_size=max_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 