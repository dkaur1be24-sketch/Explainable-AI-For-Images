import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import shap

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 10
DATA_DIR      = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\data"
MODEL_PATH    = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\resnet18_cifar10.pth"
OUT_DIR       = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\xai_outputs\linkedin"
CLASS_TO_SHOW = "airplane"   # ← change this to any class
N_BACKGROUND  = 50
DISPLAY_SIZE  = 672

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck"
]

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using device : {DEVICE}")


# ─────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
raw_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


# ─────────────────────────────────────────────
#  Load datasets
# ─────────────────────────────────────────────
test_raw   = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                 download=True,  transform=raw_transform)
test_norm  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                 download=False, transform=test_transform)
train_norm = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                 download=False, transform=test_transform)

target_cls = CIFAR10_CLASSES.index(CLASS_TO_SHOW)
chosen_idx = next(i for i, (_, lbl) in enumerate(test_raw)
                  if lbl == target_cls)
print(f"Image index {chosen_idx} — class '{CLASS_TO_SHOW}'\n")


# ─────────────────────────────────────────────
#  Load model  (used only for inference — never touched by SHAP)
# ─────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                 weights_only=True))
model.to(DEVICE).eval()
print(f"Model loaded ← {MODEL_PATH}\n")


# ─────────────────────────────────────────────
#  THE ROOT CAUSE & FIX
#
#  shap.DeepExplainer wraps the entire backward pass inside a custom
#  PyTorch autograd Function. PyTorch then forbids ANY inplace operation
#  on tensors that flow through that Function.
#
#  ResNet-18 has inplace ops in TWO places:
#
#    1.  nn.ReLU(inplace=True)  →  relu_(x)   overwrites tensor in memory
#    2.  out += identity        →  adds residual connection in-place
#
#  Fixing only ReLU still crashes on the += residual (that is exactly
#  what line 102 of resnet.py is). We must fix BOTH.
#
#  Solution: rewrite BasicBlock.forward() so that:
#    - ReLU uses inplace=False
#    - residual addition uses  out = out + identity  (new tensor each time)
#
#  Weights are copied exactly — output values are numerically identical.
#  Only the memory behaviour changes.
# ─────────────────────────────────────────────
class SafeBasicBlock(nn.Module):
    """
    Identical to torchvision BasicBlock but with ALL inplace ops removed.
    Safe to use with shap.DeepExplainer.
    """
    expansion = 1

    def __init__(self, src):
        super().__init__()
        self.conv1      = src.conv1
        self.bn1        = src.bn1
        self.relu       = nn.ReLU(inplace=False)   # fix 1: no inplace ReLU
        self.conv2      = src.conv2
        self.bn2        = src.bn2
        self.downsample = src.downsample
        self.stride     = src.stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity    # fix 2: out + identity  NOT  out += identity
        out = self.relu(out)
        return out


def make_shap_safe_model(trained_model):
    """
    Deep-copy the trained model. Replace every BasicBlock with
    SafeBasicBlock. Also fix the stem ReLU.
    All weights are preserved — only inplace behaviour changes.
    """
    safe = copy.deepcopy(trained_model)

    # Replace all four layer groups
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        old_layer = getattr(safe, layer_name)
        new_layer = nn.Sequential(
            *[SafeBasicBlock(block) for block in old_layer]
        )
        setattr(safe, layer_name, new_layer)

    # Fix the stem ReLU (after the very first conv)
    safe.relu = nn.ReLU(inplace=False)

    safe.eval()
    return safe


# ─────────────────────────────────────────────
#  Build SHAP-safe model + explainer
# ─────────────────────────────────────────────
print("Building SHAP-safe model...")
shap_model = make_shap_safe_model(model)
print("Safe model ready — all inplace operations removed.\n")

print(f"Loading {N_BACKGROUND} background images from training set...")
bg_loader  = DataLoader(train_norm, batch_size=N_BACKGROUND,
                        shuffle=True, num_workers=0)
background = next(iter(bg_loader))[0].to(DEVICE)

explainer  = shap.DeepExplainer(shap_model, background)
print("SHAP explainer ready.\n")


# ─────────────────────────────────────────────
#  Inference on chosen image
# ─────────────────────────────────────────────
raw_t,  true_lbl = test_raw[chosen_idx]
norm_t, _        = test_norm[chosen_idx]
raw_np   = (raw_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
norm_inp = norm_t.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = torch.softmax(model(norm_inp), dim=1)[0]
pred_cls    = probs.argmax().item()
conf        = probs[pred_cls].item()
correct_sym = "✔" if pred_cls == true_lbl else "✘"
print(f"Predicted: {CIFAR10_CLASSES[pred_cls]} {correct_sym}  ({conf*100:.1f}%)")


# ─────────────────────────────────────────────
#  Compute SHAP values
# ─────────────────────────────────────────────
print("Computing SHAP values — please wait...")
shap_vals = explainer.shap_values(norm_inp.detach().clone(), check_additivity=False)
print("SHAP done.\n")

# ── Extract sv from shap_vals ────────────────────────────────────────
# Your SHAP version returns shape (1, 3, H, W, NUM_CLASSES)
# We need (3, H, W) for the predicted class only.
raw_sv = np.array(shap_vals)               # ensure numpy array
print(f"  shap_vals shape : {raw_sv.shape}")

if raw_sv.ndim == 5:
    # Shape: (1, 3, H, W, NUM_CLASSES)  ← your version
    sv = raw_sv[0, :, :, :, pred_cls]     # → (3, H, W)

elif raw_sv.ndim == 4 and raw_sv.shape[-1] == NUM_CLASSES:
    # Shape: (1, H, W, NUM_CLASSES)
    sv = raw_sv[0, :, :, pred_cls]        # → (H, W)
    sv = np.stack([sv, sv, sv], axis=0)   # → (3, H, W)

elif raw_sv.ndim == 4:
    # Shape: (1, 3, H, W)
    sv = raw_sv[0]                        # → (3, H, W)

elif isinstance(shap_vals, list):
    # List of NUM_CLASSES arrays, each (1, 3, H, W)
    sv = np.array(shap_vals[pred_cls])[0] # → (3, H, W)

else:
    sv = raw_sv[0]

print(f"  sv final shape  : {sv.shape}\n")
assert sv.ndim == 3 and sv.shape[0] == 3, (
    f"Unexpected sv shape {sv.shape} — expected (3, H, W)")

# Panel 2 — positive SHAP (supports prediction)
pos_map     = np.maximum(sv, 0).mean(0)
pos_map     = (pos_map - pos_map.min()) / (pos_map.max() - pos_map.min() + 1e-8)

# Panel 3 — absolute SHAP (overall importance)
abs_map     = np.abs(sv).mean(0)
abs_map     = (abs_map - abs_map.min()) / (abs_map.max() - abs_map.min() + 1e-8)

# Panel 4 — signed SHAP (red=supports, blue=opposes)
signed      = sv.mean(0)
vmax        = np.abs(signed).max() + 1e-8
signed_norm = (signed / vmax + 1) / 2     # [0,1] for RdBu_r colourmap

def blend(raw, cmap, val, alpha=0.55):
    heat = (plt.get_cmap(cmap)(val)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * raw).astype(np.uint8)

pos_ov  = blend(raw_np, "hot",    pos_map)
abs_ov  = blend(raw_np, "YlOrRd", abs_map)
sign_ov = blend(raw_np, "RdBu_r", signed_norm)


# ─────────────────────────────────────────────
#  LANCZOS upscale to DISPLAY_SIZE
# ─────────────────────────────────────────────
def upscale(arr):
    return np.array(
        Image.fromarray(arr).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS))

raw_up  = upscale(raw_np)
pos_up  = upscale(pos_ov)
abs_up  = upscale(abs_ov)
sign_up = upscale(sign_ov)


# ─────────────────────────────────────────────
#  LinkedIn figure — dark theme, 4 panels
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7.5), facecolor="#111111")

fig.text(0.5, 0.97, "SHAP  ·  ResNet-18 on CIFAR-10",
         ha="center", va="top", fontsize=18,
         fontweight="bold", color="white")
fig.text(0.5, 0.90,
         f"Class: {CLASS_TO_SHOW.upper()}   |   "
         f"Predicted: {CIFAR10_CLASSES[pred_cls]} {correct_sym}   |   "
         f"Confidence: {conf*100:.1f}%",
         ha="center", va="top", fontsize=12, color="#999999")

panels = [
    (0.02,  0.10, 0.22, 0.72),
    (0.265, 0.10, 0.22, 0.72),
    (0.51,  0.10, 0.22, 0.72),
    (0.755, 0.10, 0.22, 0.72),
]
labels = [
    "Original image",
    "Positive SHAP\n(what supports prediction)",
    "Absolute SHAP\n(overall importance)",
    "Signed SHAP\n(red = for  ·  blue = against)",
]
images = [raw_up, pos_up, abs_up, sign_up]

for (l, b, w, h), img, lbl in zip(panels, images, labels):
    ax = fig.add_axes([l, b, w, h])
    ax.imshow(img); ax.axis("off")
    ax.text(0.5, -0.04, lbl, transform=ax.transAxes,
            ha="center", va="top", fontsize=10,
            color="#cccccc", linespacing=1.5)

# Colourbar under signed panel
cb = fig.add_axes([0.755, 0.04, 0.22, 0.025])
cb.imshow(np.linspace(0, 1, 256).reshape(1, -1),
          aspect="auto", cmap="RdBu_r", extent=[0, 1, 0, 1])
cb.set_yticks([])
cb.set_xticks([0, 0.5, 1])
cb.set_xticklabels(["Opposes class", "Neutral", "Supports class"],
                    color="#aaaaaa", fontsize=8)
cb.tick_params(colors="#555555", length=3)
for sp in cb.spines.values(): sp.set_edgecolor("#333333")

fig.text(0.5, 0.02,
         "SHAP: Shapley values assign each pixel a fair contribution "
         "score based on cooperative game theory",
         ha="center", va="bottom",
         fontsize=9, color="#666666", style="italic")

# ─────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────
save_path = os.path.join(OUT_DIR, f"linkedin_shap_{CLASS_TO_SHOW}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"✅  Saved  →  {save_path}")
print(f"   300 DPI  |  Open from File Explorer to view")
