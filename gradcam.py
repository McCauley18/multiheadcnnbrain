import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import CLASSES, DEVICE


def disable_inplace_activations(module):
    for child in module.children():
        if isinstance(child, (nn.ReLU, nn.SiLU)):
            if hasattr(child, 'inplace'):
                child.inplace = False
        else:
            disable_inplace_activations(child)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output, _ = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(torch.abs(gradients), dim=[1, 2], keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def get_gradcam_target_layer(model):
    try:
        return model.backbone.features[-1].block[-1][0]
    except Exception:
        return model.backbone.features[-1]


def visualize_gradcam(image_path, model, grad_cam, transform, alpha=0.55):
    image = Image.open(image_path).convert('RGB')
    img_original = np.array(image) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _ = model(input_tensor)
        pred = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred].item()

    cam = grad_cam.generate_cam(input_tensor, pred)

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        image.size, Image.BICUBIC
    )
    cam_resized = np.array(cam_resized)

    heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
    overlay = heatmap * alpha + img_original * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_original)
    plt.title('Original MRI')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cam_resized, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(overlay)
    plt.title(f'Overlay\n{CLASSES[pred]} ({confidence:.2%})')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_original)
    plt.contour(
        cam_resized,
        levels=[80, 140, 200],
        colors=['cyan', 'yellow', 'red'],
        linewidths=1.5
    )
    plt.title('Tumor Contours')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return pred, confidence, overlay, cam_resized