import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    PHASE1_EPOCHS, PHASE2_EPOCHS, PHASE3_EPOCHS,
    PHASE1_LR, PHASE2_LR_BACKBONE, PHASE2_LR_HEAD,
    PHASE3_LR_BACKBONE, PHASE3_LR_HEAD,
    WEIGHT_DECAY, LABEL_SMOOTHING, MODEL_PATH, DEVICE
)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


def get_optimizer(model, phase):
    if phase == 1:
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
        )
    elif phase == 2:
        return optim.AdamW([
            {'params': [p for n, p in model.named_parameters()
                        if 'backbone.features' in n and p.requires_grad],
             'lr': PHASE2_LR_BACKBONE},
            {'params': [p for n, p in model.named_parameters()
                        if 'backbone.features' not in n],
             'lr': PHASE2_LR_HEAD}
        ], weight_decay=WEIGHT_DECAY)
    else:
        return optim.AdamW([
            {'params': [p for n, p in model.named_parameters()
                        if 'backbone.features' in n and p.requires_grad],
             'lr': PHASE3_LR_BACKBONE},
            {'params': [p for n, p in model.named_parameters()
                        if 'backbone.features' not in n],
             'lr': PHASE3_LR_HEAD}
        ], weight_decay=WEIGHT_DECAY)


def progressive_training(model, train_loader, val_loader, device=DEVICE):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    phases = [
        (1, PHASE1_EPOCHS, 0, 'CosineAnnealing'),
        (2, PHASE2_EPOCHS, 3, 'ReduceLROnPlateau'),
        (3, PHASE3_EPOCHS, 6, 'ReduceLROnPlateau'),
    ]

    for phase, epochs, unfreeze_n, sched_type in phases:
        model.freeze_backbone(unfreeze_last_n=unfreeze_n)
        optimizer = get_optimizer(model, phase=phase)

        if sched_type == 'CosineAnnealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            factor = 0.5 if phase == 2 else 0.3
            patience = 3 if phase == 2 else 2
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=factor, patience=patience
            )

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if sched_type == 'CosineAnnealing':
                scheduler.step()
            else:
                scheduler.step(val_acc)

            print(f"Phase {phase} | Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)

    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }