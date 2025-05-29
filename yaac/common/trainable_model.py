"""Abstract base class for trainable models."""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class TrainableModel(ABC, torch.nn.Module):
    """Abstract base class for trainable models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass. Returns raw outputs (logits, bounding boxes, etc.)."""
        pass

    @abstractmethod
    def loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """
        Compute loss from forward outputs and targets.
        
        Args:
            outputs: Raw outputs from forward()
            targets: Ground-truth data
        Returns:
            A dict of named losses.
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """
        Postprocess raw outputs into final predictions.

        For example:
            - Classifier: logits → softmax scores or class labels
            - Detector: bbox offsets → bboxes, then apply NMS
        """
        pass
