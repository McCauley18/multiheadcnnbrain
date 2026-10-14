import torch

from config import DEVICE, DATA_DIR
from dataset import build_dataloaders
from model import BrainTumorModel
from train import progressive_training
from evaluate import (
    get_predictions, print_classification_report,
    plot_confusion_matrix, plot_training_curves, load_best_model
)
from gradcam import (
    disable_inplace_activations, GradCAM,
    get_gradcam_target_layer, visualize_gradcam
)


def main():
    print(f"Device: {DEVICE}")
    print(f"GPUs available: {torch.cuda.device_count()}")

    train_loader, val_loader, test_loader, full_dataset = build_dataloaders(DATA_DIR)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    model = BrainTumorModel().to(DEVICE)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    history = progressive_training(model, train_loader, val_loader, DEVICE)
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")

    plot_training_curves(history, save_path='training_curves.png')

    model = load_best_model(device=DEVICE)

    labels, preds = get_predictions(model, test_loader)
    print_classification_report(labels, preds)
    plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png')

    disable_inplace_activations(model)
    target_layer = get_gradcam_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    sample_img = full_dataset.images[0]
    visualize_gradcam(
        sample_img, model, grad_cam,
        transform=__import__('dataset').get_transforms()[1],
        alpha=0.55
    )


if __name__ == '__main__':
    main()