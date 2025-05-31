"""Yet Another Detector"""

import torch
from typing import Dict, Tuple
import timm


class YAD(torch.nn.Module):
    """Yet Another Detector


    img --> backbone --> neck   --> bbox_head --------> bbox_out
                                --> cls_head  --------> cls_out
                                --> objectness_head --> objectness_out

    Args:
        backbone: Backbone network
        neck: Neck network
        bbox_head: Bbox head network
        cls_head: Classification head network
        objectness_head: Objectness head network
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        neck: torch.nn.Module,
        bbox_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        objectness_head: torch.nn.Module,
    ):
        super().__init__()
        self._backbone = backbone
        self._neck = neck
        self._bbox_head = bbox_head
        self._cls_head = cls_head
        self._objectness_head = objectness_head

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            img: Input image
        Returns:
            bbox_out: Bbox output
            cls_out: Classification output
            objectness_out: Objectness output
        """
        backbone_features = self._backbone(img)
        neck_features = self._neck(backbone_features)
        bbox_out = self._bbox_head(neck_features)
        cls_out = self._cls_head(neck_features)
        objectness_out = self._objectness_head(neck_features)
        return bbox_out, cls_out, objectness_out

    def loss(
        self,
        bbox_out: torch.Tensor,
        cls_out: torch.Tensor,
        objectness_out: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Loss function

        Args:
            bbox_out: Bbox output
            cls_out: Classification output
            objectness_out: Objectness output
        Returns:
            Keys are loss names, values are loss values (scalar tensors)
        """
        return {
            "bbox_loss": torch.tensor(0.0, device=bbox_out.device),
            "cls_loss": torch.tensor(0.0, device=cls_out.device),
            "objectness_loss": torch.tensor(0.0, device=objectness_out.device),
        }

    def postprocess(
        self, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Postprocess outputs

        Args:
            outputs: Outputs from forward pass

        Returns:
            The detected bboxes. Shape (B, N, 6).
            - B is the batch size.
            - N is the number of detected bboxes.
            - 6 is the bounding box format: (min_row, min_col, max_row, max_col, score, class_id).
        """
        pass



def make_yad(num_classes: int = 80) -> YAD:
    """Make a YAD model.

    This function creates a YAD model with the following components:

    Args:
        num_classes: Number of classes for classification head. Defaults to 80 (COCO).

    Returns:
        A YAD model.
    """
    backbone = timm.create_model("resnet18", pretrained=True)
    input_normalization_kwargs = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    # Break up the backbone into a list of modules, then wrap in a Sequential
    if hasattr(backbone, "children"):
        backbone_modules = list(backbone.children())
        chosen_modules = backbone_modules[:-4]
        backbone = torch.nn.Sequential(*chosen_modules)
    else:
        raise ValueError("Backbone must have children")

    backbone_out_channels = 128
    
    # The typical order is: Conv -> BatchNorm -> Activation
    neck = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
    )

    # BBox head: two convs
    bbox_head = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
    )

    # Objectness head: two convs, one output channel
    objectness_head = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=1,  # One channel for objectness
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
    )

    # Classification head: two convs, num_classes output channels
    cls_head = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=num_classes,  # One channel per class
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
    )

    return YAD(
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        cls_head=cls_head,
        objectness_head=objectness_head,
    )


if __name__ == "__main__":
    make_yad()
    
    