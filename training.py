import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

# ─────────────────────────────────────────────
#  0.  Config
# ─────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 10
BATCH_SIZE   = 128
NUM_EPOCHS   = 30
LR_HEAD      = 1e-3       # learning rate for the FC head
LR_LAYER4    = 1e-4       # lower LR for the unfrozen backbone block
PATIENCE     = 5          # early stopping patience
DATA_DIR     = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\data"
SAVE_PATH    = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\resnet18_cifar10.pth"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck"
]

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
#  1.  Data  (CIFAR-10 — auto-downloads)
# ─────────────────────────────────────────────
# ResNet expects 224×224 — upsample from 32×32
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=16),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True,  download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"Train samples: {len(train_dataset):,}  |  Test samples: {len(test_dataset):,}")


# ─────────────────────────────────────────────
#  2.  Model
#      — Freeze all backbone layers
#      — Unfreeze layer4 for richer CIFAR-10 features  (Week-1 addition)
#      — Replace FC head for 10 classes
# ─────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze the entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 — gives better feature adaptation to CIFAR-10
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final FC layer (512 → 10)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    # model.fc is unfrozen by default (new layer)

    return model.to(DEVICE)

model = build_model()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params : {trainable:,}  /  Total: {total:,}")


# ─────────────────────────────────────────────
#  3.  Loss, Optimizer, Scheduler
#      — Separate LR for layer4 vs FC head
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": LR_LAYER4},
    {"params": model.fc.parameters(),     "lr": LR_HEAD},
])

# Decay LR by 0.5 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# ─────────────────────────────────────────────
#  4.  Evaluate helper  (standalone — NOT inside the training loop)
# ─────────────────────────────────────────────
def evaluate(loader):
    """Return accuracy (%) on the given DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs      = model(imgs)
            _, preds     = outputs.max(1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────────────
#  5.  Training loop with Early Stopping  (Week-1 addition)
# ─────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "test_acc": []}

# Early-stopping state — defined BEFORE the loop
best_acc         = 0.0
best_state       = None
no_improve_count = 0

print("\n── Training ──────────────────────────────────────────────")
for epoch in range(1, NUM_EPOCHS + 1):

    # ── Train for one epoch ───────────────────
    model.train()
    running_loss = correct = total = 0

    for imgs, labels in tqdm(train_loader,
                             desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}",
                             leave=False, ncols=80):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds      = outputs.max(1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    scheduler.step()

    # ── End-of-epoch metrics ──────────────────
    train_loss = running_loss / total
    train_acc  = 100.0 * correct / total
    test_acc   = evaluate(test_loader)          # call the standalone function

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)

    print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # ── Early stopping check ──────────────────
    if test_acc > best_acc:
        best_acc         = test_acc
        best_state       = copy.deepcopy(model.state_dict())
        no_improve_count = 0
        print(f"           ✔ New best: {best_acc:.2f}%  (model saved in memory)")
    else:
        no_improve_count += 1
        print(f"           No improvement ({no_improve_count}/{PATIENCE})")
        if no_improve_count >= PATIENCE:
            print(f"\n⏹  Early stopping triggered at epoch {epoch}.")
            break

# Restore the best weights before saving
model.load_state_dict(best_state)
torch.save(best_state, SAVE_PATH)
print(f"\n✅ Best Test Accuracy : {best_acc:.2f}%")
print(f"Model saved          → {SAVE_PATH}")


# ─────────────────────────────────────────────
#  6.  Training curves
# ─────────────────────────────────────────────
epochs_ran = len(history["train_loss"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, epochs_ran + 1), history["train_loss"],
         label="Train Loss", color="#e63946")
ax1.set_title("Loss over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(range(1, epochs_ran + 1), history["train_acc"],
         label="Train Acc", color="#457b9d")
ax2.plot(range(1, epochs_ran + 1), history["test_acc"],
         label="Test Acc",  color="#2a9d8f")
ax2.axhline(y=best_acc, color="gray", linestyle="--", alpha=0.6,
            label=f"Best: {best_acc:.2f}%")
ax2.set_title("Accuracy over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved → training_curves.png")


# ─────────────────────────────────────────────
#  7.  Per-class accuracy breakdown
#      (useful baseline before XAI — Week 2 prep)
# ─────────────────────────────────────────────
print("\n── Per-Class Accuracy ────────────────────────────────────")
class_correct = [0] * NUM_CLASSES
class_total   = [0] * NUM_CLASSES

model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs      = model(imgs)
        _, preds     = outputs.max(1)
        for label, pred in zip(labels, preds):
            class_total[label]  += 1
            if label == pred:
                class_correct[label] += 1

for i, cls in enumerate(CIFAR10_CLASSES):
    acc = 100.0 * class_correct[i] / class_total[i]
    bar = "█" * int(acc / 5)
    print(f"  {cls:<12} {acc:5.1f}%  {bar}")
