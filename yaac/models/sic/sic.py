import os
import torchvision
import torch
from transformers import AutoModel
from yaac.common.trainable_model import TrainableModel
from yaac.models.sic.classification_heads import SingleFCClassificationHead
from functools import partial
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Union


class SIC(TrainableModel):
    """Simple Image Classifier using composition pattern.
    
    This class uses composition to separate concerns between the backbone,
    predictions head, loss function, and postprocessing function.
    """
    
    def __init__(
        self,
        backbone: torch.nn.Module,
        predictions_head: torch.nn.Module,
        loss_function: torch.nn.Module,
        postprocess_function: callable,
        input_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ):
        """Initialize SIC with composed components.
        
        Args:
            backbone: Feature extraction backbone network
            predictions_head: Final classification head
            loss_function: Loss function for training
            postprocess_function: Function to postprocess raw outputs
            input_size: Expected input image size (height, width). Defaults to (224, 224).
            mean: Normalization mean values for RGB channels. Defaults to ImageNet values.
            std: Normalization std values for RGB channels. Defaults to ImageNet values.
        """
        super().__init__()
        self._backbone = backbone
        self._predictions_head = predictions_head
        self._loss_function = loss_function
        self._postprocess_function = postprocess_function
        self._input_size = input_size
        # ImageNet normalization stats (default)
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
        self._mean = torch.tensor(mean)
        self._std = torch.tensor(std)

    def preprocess(self, inputs: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess input data for model inference.
        
        Handles PIL Image, torch.Tensor, or numpy.ndarray inputs. Applies only
        necessary transforms: resize if needed, convert to tensor if needed,
        normalize if input is in [0,255] or [0,1] range.
        
        Args:
            inputs: Input data in PIL Image, torch.Tensor, or numpy.ndarray format
            
        Returns:
            Preprocessed tensor in format (B, C, H, W) ready for model forward pass
        """
        target_h, target_w = self._input_size
        target_h = int(target_h)
        target_w = int(target_w)
        
        # Convert to tensor if needed
        if isinstance(inputs, Image.Image):
            # PIL Image: convert to RGB if needed
            if inputs.mode != "RGB":
                inputs = inputs.convert("RGB")
            
            # Resize if needed before converting to tensor
            if inputs.size != (target_w, target_h):
                inputs = inputs.resize((target_w, target_h), Image.LANCZOS)
            
            tensor = transforms.ToTensor()(inputs)
        elif isinstance(inputs, np.ndarray):
            # Numpy array: handle different shapes
            if inputs.dtype != np.float32 and inputs.dtype != np.float64:
                inputs = inputs.astype(np.float32)
            
            # Handle different array shapes
            if len(inputs.shape) == 2:
                # Grayscale (H, W) -> (1, H, W)
                tensor = torch.from_numpy(inputs).unsqueeze(0)
            elif len(inputs.shape) == 3:
                if inputs.shape[2] == 3:
                    # (H, W, C) -> (C, H, W)
                    tensor = torch.from_numpy(inputs).permute(2, 0, 1)
                else:
                    # (C, H, W)
                    tensor = torch.from_numpy(inputs)
            elif len(inputs.shape) == 4:
                # (B, C, H, W) or (B, H, W, C)
                if inputs.shape[3] == 3:
                    tensor = torch.from_numpy(inputs).permute(0, 3, 1, 2)
                else:
                    tensor = torch.from_numpy(inputs)
            else:
                raise ValueError(f"Unsupported numpy array shape: {inputs.shape}")
        elif isinstance(inputs, torch.Tensor):
            tensor = inputs
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        # Ensure tensor is float
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        # Handle batch dimension
        has_batch = len(tensor.shape) == 4
        if not has_batch:
            tensor = tensor.unsqueeze(0)
        
        # Get current dimensions (convert to Python ints)
        _, num_channels, current_h, current_w = tensor.shape
        current_h = int(current_h)
        current_w = int(current_w)
        
        # Ensure we have 3 channels (RGB)
        if num_channels == 1:
            # Grayscale -> RGB by repeating channels
            tensor = tensor.repeat(1, 3, 1, 1)
        elif num_channels != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {num_channels}")
        
        # Resize if needed (for tensors/numpy arrays that weren't PIL)
        if current_h != target_h or current_w != target_w:
            tensor = transforms.functional.resize(
                tensor, size=(target_h, target_w), antialias=True
            )
        
        # Check value range and normalize if needed
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Determine if we should normalize
        should_normalize = False
        
        if max_val > 1.0 and min_val >= 0.0:
            # Values are in [0, 255] range
            tensor = tensor / 255.0
            should_normalize = True
        elif max_val <= 1.0 and min_val >= 0.0:
            # Values are in [0, 1] range
            should_normalize = True
        # Otherwise, values are already normalized or outside expected range, skip normalization
        
        # Apply ImageNet normalization if needed
        if should_normalize:
            mean = self._mean.view(1, 3, 1, 1).to(tensor.device)
            std = self._std.view(1, 3, 1, 1).to(tensor.device)
            tensor = (tensor - mean) / std
        
        return tensor
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            image: Input image tensor
            
        Returns:
            Raw logits from the predictions head
        """
        features = self._backbone(image)
        logits = self._predictions_head(features)
        return logits

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute loss between outputs and targets.
        
        Args:
            outputs: Raw model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary containing the loss value
        """
        loss = self._loss_function(outputs, targets)
        return {"loss": loss}

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        """Postprocess raw outputs into final predictions.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed predictions (probabilities or confidences)
        """
        confidences = self._postprocess_function(outputs)
        return confidences


def make_model(
    num_classes: int,
    backbone_type: str = "resnet18",
    head_type: str = "basic_001",
    loss_type: str = "auto",
    postprocess_type: str = "auto",
    input_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> SIC:
    """Factory function to create a SIC model with configurable components.
    
    This function creates a SIC model using configurable backbone, head,
    loss function, and postprocessing components.
    
    Note: Models are created with random weights. Customers will load their
    own trained weights using load_model_from_checkpoint().
    
    Args:
        num_classes: Number of output classes
        backbone_type: Type of backbone to use ("resnet18" or "convnext_tiny_dinov3")
        head_type: Type of predictions head to use ("basic_001" or "basic_convnext_tiny")
        loss_type: Type of loss function to use ("auto", "bce", "ce")
        postprocess_type: Type of postprocess function to use ("auto", "sigmoid", "softmax")
        input_size: Expected input image size (height, width). Defaults to (224, 224).
        mean: Normalization mean values for RGB channels. Defaults to ImageNet values.
        std: Normalization std values for RGB channels. Defaults to ImageNet values.
        
    Returns:
        Configured SIC model instance
    """
    backbone = _build_backbone(backbone_type)
    predictions_head = _build_head(head_type, num_classes)
    loss_function = _build_loss_function(loss_type, num_classes)
    postprocess_function = _build_postprocess_function(postprocess_type, num_classes)
    
    return SIC(
        backbone=backbone,
        predictions_head=predictions_head,
        loss_function=loss_function,
        postprocess_function=postprocess_function,
        input_size=input_size,
        mean=mean,
        std=std,
    )


def _build_backbone(backbone_type: str) -> torch.nn.Module:
    """Build backbone network based on type.
    
    Args:
        backbone_type: Type of backbone to build ("resnet18" or "convnext_tiny_dinov3")
        
    Returns:
        Backbone network module that outputs (batch, features, H, W) or (batch, features, 1, 1)
    """
    if backbone_type == "resnet18":
        # Create ResNet18 without pretrained weights (customers load their own)
        backbone = torchvision.models.resnet18(weights=None)
        # Remove avgpool and fc layers, keep only the feature extraction part
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        return backbone
    elif backbone_type == "convnext_tiny_dinov3":
        return _build_convnext_tiny_dinov3_backbone()
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def _build_convnext_tiny_dinov3_backbone() -> torch.nn.Module:
    """Build ConvNeXt-Tiny backbone from DINOv3.
    
    Loads the DINOv3 ConvNeXt-Tiny architecture from HuggingFace.
    Weights are loaded separately via load_model_from_checkpoint().
    
    Returns:
        Backbone module that takes (batch, 3, H, W) and returns (batch, 768, H', W')
    """
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Required for loading gated DINOv3 model from HuggingFace. "
            "Get a token from: https://huggingface.co/settings/tokens"
        )
    hf_model = AutoModel.from_pretrained(
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        token=hf_token,
    )
    
    class DINOv3Backbone(torch.nn.Module):
        """Simple wrapper around DINOv3 ConvNeXt stages."""
        
        def __init__(self, stages: torch.nn.ModuleList, layer_norm: torch.nn.Module):
            super().__init__()
            self.stages = stages
            self.layer_norm = layer_norm
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for stage in self.stages:
                x = stage(x)
            # Layer norm: (B,C,H,W) -> (B,H,W,C) -> norm -> (B,C,H,W)
            x = x.permute(0, 2, 3, 1)
            x = self.layer_norm(x)
            x = x.permute(0, 3, 1, 2)
            return x
    
    return DINOv3Backbone(hf_model.stages, hf_model.layer_norm)


def _build_head(head_type: str, num_classes: int) -> torch.nn.Module:
    """Build predictions head based on type.
    
    Args:
        head_type: Type of head to build ("basic_001" or "basic_convnext_tiny")
        num_classes: Number of output classes
        n
    Returns:
        Predictions head module
    """
    # For binary classification (num_classes=2), use 1 output logit
    # This matches yaac_internal's behavior: binary classification uses BCEWithLogitsLoss
    # with a single logit output (sigmoid activation)
    head_output_dim = 1 if num_classes == 2 else num_classes
    
    if head_type == "basic_001":
        # Basic head for ResNet18 (feature dimension is 512)
        return SingleFCClassificationHead(
            in_features=512,
            num_classes=head_output_dim,
            bias=True,
        )
    elif head_type == "basic_convnext_tiny":
        # Basic head for ConvNeXt-Tiny (feature dimension is 768)
        return SingleFCClassificationHead(
            in_features=768,
            num_classes=head_output_dim,
            bias=True,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def _build_loss_function(loss_type: str, num_classes: int) -> torch.nn.Module:
    """Build loss function based on type and number of classes.
    
    Args:
        loss_type: Type of loss function to build
        num_classes: Number of output classes
        
    Returns:
        Loss function module
    """
    if loss_type == "auto":
        # Automatically choose based on number of classes
        if num_classes == 1 or num_classes == 2:
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
    elif loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == "ce":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def _build_postprocess_function(postprocess_type: str, num_classes: int) -> callable:
    """Build postprocess function based on type and number of classes.
    
    Args:
        postprocess_type: Type of postprocess function to build
        num_classes: Number of output classes
        
    Returns:
        Postprocess function
    """
    if postprocess_type == "auto":
        # Automatically choose based on number of classes
        if num_classes == 1 or num_classes == 2:
            return torch.sigmoid
        else:
            return partial(torch.nn.functional.softmax, dim=1)
    elif postprocess_type == "sigmoid":
        return torch.sigmoid
    elif postprocess_type == "softmax":
        return partial(torch.nn.functional.softmax, dim=1)
    else:
        raise ValueError(f"Unknown postprocess type: {postprocess_type}")


if __name__ == "__main__":
    # Example 1: Create a model from scratch
    print("Example 1: Creating a model from scratch")
    model = make_model(num_classes=2, backbone_type="resnet18")
    dummy_image = torch.randn(1, 3, 224, 224)
    outputs = model(dummy_image)
    print(f"  Model output shape: {outputs.shape}")
    
    # Example 2: Load model from checkpoint (commented out - requires checkpoint files)
    # print("\nExample 2: Loading model from checkpoint")
    # from yaac.common.model_loader import load_model_from_checkpoint
    # 
    # checkpoint_dir = Path("path/to/checkpoint")
    # loaded_model, config = load_model_from_checkpoint(checkpoint_dir, device="cpu")
    # print(f"  Loaded model type: {config['model_type']}")
    # print(f"  Backbone: {config['backbone_type']}")
    # print(f"  Classes: {config['classes']}")
    # 
    # # Run inference
    # predictions = loaded_model(dummy_image)
    # print(f"  Predictions shape: {predictions.shape}")