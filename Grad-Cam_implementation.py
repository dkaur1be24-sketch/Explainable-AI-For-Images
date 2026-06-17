import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# ─────────────────────────────────────────────
#  CONFIG — change CLASS_TO_SHOW to any of:
#  "airplane" "automobile" "bird" "cat" "deer"
#  "dog" "frog" "horse" "ship" "truck"
# ─────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 10
DATA_DIR      = r"D:\ALL_PROJECTS\XAI_PROJECT\data"
MODEL_PATH    = r"D:\ALL_PROJECTS\XAI_PROJECT\resnet18_cifar10.pth"
OUT_DIR       = r"D:\ALL_PROJECTS\XAI_PROJECT\xai_outputs\linkedin"
CLASS_TO_SHOW = "dog"   

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck"
]

os.makedirs(OUT_DIR, exist_ok=True)


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
#  Load data
# ─────────────────────────────────────────────
test_raw  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                download=True, transform=raw_transform)
test_norm = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                download=True, transform=test_transform)

target_class = CIFAR10_CLASSES.index(CLASS_TO_SHOW)
chosen_idx   = next(i for i, (_, lbl) in enumerate(test_raw)
                    if lbl == target_class)
print(f"Using image index {chosen_idx} for class '{CLASS_TO_SHOW}'")


# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                 weights_only=True))
model.to(DEVICE).eval()


# ─────────────────────────────────────────────
#  Grad-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, layer):
        self.act = self.grad = None
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "act", o.detach()))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def generate(self, inp, cls):
        model.zero_grad()
        out = model(inp)
        out[0, cls].backward()
        w   = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1).squeeze())
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return F.interpolate(cam[None, None], (224, 224),
                             mode="bilinear",
                             align_corners=False).squeeze().cpu().numpy()


gradcam  = GradCAM(model, model.layer4[-1])
raw_t,  true_lbl = test_raw[chosen_idx]
norm_t, _        = test_norm[chosen_idx]

# ── LANCZOS upscale: 224 → 672px for crisp LinkedIn display ──
# CIFAR-10 is natively 32x32. We can't invent detail that isn't there,
# but LANCZOS gives the smoothest upscale (no pixelated blocks).
DISPLAY_SIZE = 672

def upscale(arr_hwc_uint8):
    return np.array(
        Image.fromarray(arr_hwc_uint8).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS))

raw_np   = (raw_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
norm_inp = norm_t.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs    = torch.softmax(model(norm_inp), dim=1)[0]
pred_cls = probs.argmax().item()
conf     = probs[pred_cls].item()
cam      = gradcam.generate(norm_inp, pred_cls)

heat    = (plt.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.uint8)
overlay = (0.5 * heat + 0.5 * raw_np).astype(np.uint8)

# Upscale all three panels
raw_up     = upscale(raw_np)
heat_up    = upscale(heat)
overlay_up = upscale(overlay)

correct_sym = "✔" if pred_cls == true_lbl else "✘"
print(f"Predicted: {CIFAR10_CLASSES[pred_cls]} {correct_sym}  "
      f"Confidence: {conf*100:.1f}%")


# ─────────────────────────────────────────────
#  Build LinkedIn figure
#  Dark background, 3 panels, no text overlap
# ─────────────────────────────────────────────
FIG_W, FIG_H = 16, 7
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#111111")

# ── Title block — at the very top, enough margin so it never overlaps ──
fig.text(0.5, 0.97,
         "Grad-CAM  ·  ResNet-18 on CIFAR-10",
         ha="center", va="top",
         fontsize=18, fontweight="bold", color="white")

fig.text(0.5, 0.90,
         f"Class: {CLASS_TO_SHOW.upper()}   |   "
         f"Predicted: {CIFAR10_CLASSES[pred_cls]} {correct_sym}   |   "
         f"Confidence: {conf*100:.1f}%",
         ha="center", va="top",
         fontsize=12, color="#999999")

# ── Three image panels — positioned below the title block ──
#    [left, bottom, width, height]  all in figure-fraction units
panels = [
    (0.03,  0.10, 0.28, 0.72),   # Original
    (0.355, 0.10, 0.28, 0.72),   # Heatmap
    (0.68,  0.10, 0.28, 0.72),   # Overlay
]
labels = ["Original image", "Grad-CAM heatmap", "Overlay"]
images = [raw_up, heat_up, overlay_up]

for (l, b, w, h), img, lbl in zip(panels, images, labels):
    ax = fig.add_axes([l, b, w, h])
    ax.imshow(img)
    ax.axis("off")
    # Panel label sits BELOW the image, not above — avoids overlap with title
    ax.text(0.5, -0.03, lbl,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=12, color="#cccccc")

# ── Colour bar legend sits at the bottom centre ──
cbar_ax = fig.add_axes([0.355, 0.04, 0.28, 0.025])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cbar_ax.imshow(gradient, aspect="auto", cmap="jet",
               extent=[0, 1, 0, 1])
cbar_ax.set_yticks([])
cbar_ax.set_xticks([0, 0.5, 1])
cbar_ax.set_xticklabels(["Low importance",
                          "Medium",
                          "High importance"],
                         color="#aaaaaa", fontsize=9)
cbar_ax.tick_params(colors="#555555", length=3)
for sp in cbar_ax.spines.values():
    sp.set_edgecolor("#333333")


# ─────────────────────────────────────────────
#  Save — 300 DPI, no popup window
# ─────────────────────────────────────────────
save_path = os.path.join(OUT_DIR,
                         f"linkedin_gradcam_{CLASS_TO_SHOW}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()

print(f"\n✅  Saved  →  {save_path}")
print(f"   300 DPI  |  Open from File Explorer to view")
