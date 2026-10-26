"""
ResNet50 Architecture for Pulmonary Classification with Grad-CAM Explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np
import cv2


class ResNet50PulmonaryClassifier(nn.Module):
    """
    Fine-tuned ResNet-50 model tailored for pulmonary disease classification.
    Incorporates custom dense head with Dropout and Batch Normalization.
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.4):
        super(ResNet50PulmonaryClassifier, self).__init__()
        
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # In_features from the penultimate layer of ResNet-50 (2048)
        in_features = self.resnet.fc.in_features
        
        # Replace the final FC layer with our custom classification head
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

    def freeze_feature_extractor(self):
        """Freezes all layers except the final classification head."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self):
        """Unfreezes Layer 4 (last residual block) and the FC head for fine-tuning."""
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes entire network for end-to-end training."""
        for param in self.parameters():
            param.requires_grad = True


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes regions in the radiographic scan that most strongly influence
    the model's classification decision.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else self.model.resnet.layer4[-1]
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM heatmap for a given input tensor (1, C, H, W).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass for the score of the target class
        target_score = output[:, target_class]
        target_score.backward()

        # Global average pooling of gradients: weight alpha_k
        alpha = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Linear combination of forward activations weighted by alpha
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # ReLU to capture positive evidence only

        # Normalize heatmap to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam, target_class


def overlay_cam_on_image(original_img_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a normalized Grad-CAM heatmap onto the original PIL image.
    """
    orig_np = np.array(original_img_pil.convert("RGB"))
    h, w, _ = orig_np.shape
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_map = cv2.applyColorMap(heatmap_uint8, colormap)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    blended = np.uint8(alpha * color_map + (1 - alpha) * orig_np)
    return blended, color_map
