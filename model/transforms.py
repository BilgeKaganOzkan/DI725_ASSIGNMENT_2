import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    """
    Sequentially applies a list of transforms to both image and target

    This class chains multiple transforms together, ensuring that both the image
    and its corresponding target dictionary (containing bounding boxes and labels)
    are consistently transformed together.

    Args:
        transforms: List of transform objects to apply in sequence
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        """
        Apply all transforms sequentially

        Args:
            image: PIL Image to transform
            target: Dictionary containing bounding boxes and labels

        Returns:
            Tuple of (transformed_image, transformed_target)
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize:
    """
    Normalizes image tensor with mean and standard deviation

    This transform normalizes a tensor image with mean and standard deviation.
    This is a common preprocessing step that helps models converge faster by
    standardizing the input data distribution.

    Args:
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        """
        Normalize the image tensor and pass through the target unchanged

        Args:
            image: Tensor image to normalize
            target: Target dictionary (unchanged by this transform)

        Returns:
            Tuple of (normalized_image, target)
        """
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    """
    Converts a PIL Image to a PyTorch tensor

    This transform converts a PIL Image to a PyTorch tensor and scales the values
    to [0, 1] range. It's a necessary preprocessing step for neural networks that
    operate on tensor inputs.
    """
    def __call__(self, image, target=None):
        """
        Convert PIL Image to tensor and pass through the target unchanged

        Args:
            image: PIL Image to convert
            target: Target dictionary (unchanged by this transform)

        Returns:
            Tuple of (tensor_image, target)
        """
        image = F.to_tensor(image)
        return image, target


# All random augmentation classes removed - using only basic transforms


class Resize:
    """
    Resizes images while maintaining aspect ratio

    This transform resizes images to a specified size while preserving the aspect ratio.
    It also handles the corresponding transformation of bounding box coordinates in the target.

    Args:
        size: Target size (can be a single integer for the shorter side, or a tuple (h, w))
        max_size: Maximum allowed size for the longer side (to limit memory usage)
    """
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, image, target=None):
        """
        Resize the image and update target bounding boxes accordingly

        Args:
            image: PIL Image to resize
            target: Target dictionary with bounding boxes to update

        Returns:
            Tuple of (resized_image, updated_target)
        """
        return resize(image, target, self.size, self.max_size)


def resize(image, target, size, max_size=None):
    """Resize an image and update corresponding target annotations.

    This function resizes both the image and its bounding box annotations while maintaining
    aspect ratio. It supports both fixed size (w,h) and dynamic sizing where only the
    shorter side is specified.

    Args:
        image: PIL Image to resize
        target: Dictionary containing bounding boxes and other annotations
        size: Target size (scalar for shorter side or tuple for exact dimensions)
        max_size: Maximum allowed size for the longer dimension

    Returns:
        Tuple of (resized_image, updated_target)
    """
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


# Crop function removed - not needed without data augmentation

def create_training_transforms(size=320, max_size=480, config=None):
    """Create a transform pipeline for training images.

    This function builds a minimal transform pipeline that includes resizing, tensor conversion,
    and normalization. Data augmentation is intentionally excluded to maintain dataset consistency.

    Args:
        size: Target size for the shorter dimension
        max_size: Maximum allowed size for the longer dimension
        config: Configuration dictionary that may contain normalization parameters

    Returns:
        Compose object with the transform pipeline
    """
    # Get image size from config if available
    if config is not None and 'dataset' in config and 'img_size' in config['dataset']:
        if isinstance(config['dataset']['img_size'], (list, tuple)) and len(config['dataset']['img_size']) > 0:
            size = config['dataset']['img_size'][0]  # Use height
            if len(config['dataset']['img_size']) > 1:
                max_size = config['dataset']['img_size'][1]  # Use width

    # Get normalization parameters from config if available
    if config is not None and 'dataset' in config and 'normalization' in config['dataset']:
        mean = config['dataset']['normalization'].get('mean', [0.485, 0.456, 0.406])
        std = config['dataset']['normalization'].get('std', [0.229, 0.224, 0.225])
    else:
        # Default ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return Compose([
        Resize(size, max_size=max_size),  # Use config values or defaults
        ToTensor(),
        Normalize(mean, std)  # Use config values or ImageNet statistics
    ])


def create_validation_transforms(size=320, max_size=480, config=None):
    """Create a transform pipeline for validation images.

    This function builds a minimal transform pipeline identical to training transforms
    to ensure consistent evaluation metrics without any discrepancies.

    Args:
        size: Target size for the shorter dimension
        max_size: Maximum allowed size for the longer dimension
        config: Configuration dictionary that may contain normalization parameters

    Returns:
        Compose object with the validation transform pipeline
    """
    # Get normalization parameters from config if available
    if config is not None and 'dataset' in config and 'normalization' in config['dataset']:
        mean = config['dataset']['normalization'].get('mean', [0.485, 0.456, 0.406])
        std = config['dataset']['normalization'].get('std', [0.229, 0.224, 0.225])
    else:
        # Default ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return Compose([
        Resize(size, max_size=max_size),  # Consistent size for reliable evaluation
        ToTensor(),
        Normalize(mean, std)  # Use config values or ImageNet statistics
    ])