"""Abstract base class for trainable models.

This module defines the TrainableModel class, which serves as an abstract base class for all
trainable models in the system. It provides a standardized interface for model training,
inference, and post-processing across different domains (images, video, text, multimodal, etc.).
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class TrainableModel(ABC, torch.nn.Module):
    """Abstract base class for trainable models.

    This class defines the interface that all trainable models must implement. It combines
    PyTorch's nn.Module with Python's ABC to create an abstract base class that enforces a
    consistent interface across different model types and domains.

    The interface is domain-agnostic and supports models for images, video, text, multimodal,
    or any other input/output types. Subclasses define the specific input/output formats
    appropriate for their domain.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Perform the forward pass of the model.

        This method executes the core model computation. The input and output types are
        domain-specific and determined by the subclass implementation.

        Args:
            inputs: Preprocessed input data. The format depends on the model domain:
                - Image models: typically torch.Tensor (B, C, H, W)
                - Video models: typically torch.Tensor (B, T, C, H, W) or sequences
                - Text models: typically tokenized sequences or embeddings
                - Multimodal: combinations of the above

        Returns:
            Raw model outputs. The format depends on the task:
                - Classification: logits (B, num_classes)
                - Detection: bounding boxes, scores, classes
                - Segmentation: pixel-wise predictions
                - Text generation: token logits or sequences
                The exact type and shape depend on the specific model implementation.
        """
        pass

    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """Preprocess raw inputs for model inference.

        This method handles domain-specific preprocessing to convert raw inputs into the
        format expected by the forward() method. Subclasses implement preprocessing appropriate
        for their domain (e.g., image resizing/normalization, text tokenization, video frame
        extraction).

        The method should apply only necessary transforms based on the current state of
        the input (e.g., skip resizing if already the correct size, skip normalization if
        already normalized).

        Args:
            inputs: Raw input data in any format appropriate for the domain:
                - Image models: PIL.Image, numpy.ndarray, torch.Tensor, file paths
                - Video models: video files, frame sequences, torch.Tensor
                - Text models: strings, tokenized sequences, file paths
                - Multimodal: combinations of the above

        Returns:
            Preprocessed inputs ready for forward(). Typically torch.Tensor, but the exact
            format is determined by the subclass implementation.
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """Convert raw model outputs into final predictions.

        This method handles domain-specific post-processing to convert raw model outputs
        into a format suitable for evaluation, deployment, or user consumption.

        Args:
            outputs: Raw outputs from the forward() method.

        Returns:
            Processed predictions in a task-appropriate format. Examples:
                - Classification: class probabilities, confidence scores, or class labels
                - Detection: filtered bounding boxes with scores and class labels
                - Segmentation: pixel-wise class predictions or masks
                - Text generation: decoded text strings
                The exact format depends on the specific model and task.
        """
        pass

    @abstractmethod
    def loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """Compute the loss between model outputs and ground truth targets.

        This method computes the training loss by comparing model predictions with ground
        truth labels. The loss computation is task-specific and determined by the subclass.

        Args:
            outputs: Raw outputs from the forward() method.
            targets: Ground truth data corresponding to the inputs. The format depends on
                the task (e.g., class labels, bounding boxes, segmentation masks).

        Returns:
            Dictionary mapping loss names to their corresponding tensor values. Multiple
            losses can be returned for multi-task learning, auxiliary objectives, or
            regularization terms. The dictionary must contain at least one loss value.
        """
        pass
