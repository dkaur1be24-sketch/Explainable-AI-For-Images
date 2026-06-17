import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ─────────────────────────────────────────────
#  CONFIG — change CLASS_TO_SHOW to any of:
#  "airplane" "automobile" "bird" "cat" "deer"
#  "dog" "frog" "horse" "ship" "truck"
# ─────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 10
DATA_DIR      = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\data"
MODEL_PATH    = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\resnet18_cifar10.pth"
OUT_DIR       = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\xai_outputs\linkedin"
CLASS_TO_SHOW = "dog"         # ← change this
LIME_SAMPLES  = 1000         # higher = better quality, slower (~10s)
LIME_FEATURES = 6             # number of top superpixels to highlight

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

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DISPLAY_SIZE = 672   # upscale all panels to this for crisp output


# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────
test_raw  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                download=True, transform=raw_transform)
test_norm = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                download=True, transform=test_transform)

target_cls = CIFAR10_CLASSES.index(CLASS_TO_SHOW)
chosen_idx = next(i for i, (_, lbl) in enumerate(test_raw)
                  if lbl == target_cls)
print(f"Using image index {chosen_idx} for class '{CLASS_TO_SHOW}'\n")


# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                 weights_only=True))
model.to(DEVICE).eval()
print(f"Model loaded ← {MODEL_PATH}\n")


# ─────────────────────────────────────────────
#  predict_fn for LIME
# ─────────────────────────────────────────────
def predict_fn(images):
    tensors = []
    for img in images:
        t = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
        t = (t - MEAN) / STD
        tensors.append(t)
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
    return probs.cpu().numpy()


# ─────────────────────────────────────────────
#  Run model prediction
# ─────────────────────────────────────────────
raw_t,  true_lbl = test_raw[chosen_idx]
norm_t, _        = test_norm[chosen_idx]
raw_np   = (raw_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
norm_inp = norm_t.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs    = torch.softmax(model(norm_inp), dim=1)[0]
pred_cls = probs.argmax().item()
conf     = probs[pred_cls].item()
correct_sym = "✔" if pred_cls == true_lbl else "✘"
print(f"Predicted: {CIFAR10_CLASSES[pred_cls]} {correct_sym}  "
      f"({conf*100:.1f}%)\n")


# ─────────────────────────────────────────────
#  LIME explanation
# ─────────────────────────────────────────────
print(f"Running LIME ({LIME_SAMPLES} samples) — please wait...")
explainer   = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(
    raw_np, predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=LIME_SAMPLES,
    random_seed=42,
    batch_size=32,
)
print("LIME done.\n")

# ── Panel 2: highlighted superpixels (important = bright, rest = dim) ──
seg     = explanation.segments
sal_map = np.zeros(seg.shape, dtype=np.float32)
for sp_id, weight in explanation.local_exp[pred_cls]:
    sal_map[seg == sp_id] = weight
sal_map -= sal_map.min()
sal_map /= (sal_map.max() + 1e-8)

highlight = raw_np.copy().astype(float)
highlight[sal_map < 0.3] *= 0.25          # dim unimportant regions heavily
highlight = highlight.astype(np.uint8)

# ── Panel 3: green boundary overlay on full-colour image ──
_, binary_mask = explanation.get_image_and_mask(
    pred_cls,
    positive_only=True,
    num_features=LIME_FEATURES,
    hide_rest=False,
)
boundary = (mark_boundaries(raw_np / 255.0, binary_mask,
                             color=(0.2, 1.0, 0.4),   # bright green
                             mode="thick") * 255).astype(np.uint8)

# ── Panel 4: LIME saliency map as heatmap ──
heat = (plt.get_cmap("RdYlGn")(sal_map)[:, :, :3] * 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  LANCZOS upscale helper
# ─────────────────────────────────────────────
def upscale(arr):
    return np.array(
        Image.fromarray(arr).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS))

raw_up      = upscale(raw_np)
highlight_up = upscale(highlight)
boundary_up  = upscale(boundary)
heat_up      = upscale(heat)


# ─────────────────────────────────────────────
#  Build LinkedIn figure  — 4 panels, dark theme
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7), facecolor="#111111")

# ── Title block ──
fig.text(0.5, 0.97,
         "LIME  ·  ResNet-18 on CIFAR-10",
         ha="center", va="top",
         fontsize=18, fontweight="bold", color="white")
fig.text(0.5, 0.90,
         f"Class: {CLASS_TO_SHOW.upper()}   |   "
         f"Predicted: {CIFAR10_CLASSES[pred_cls]} {correct_sym}   |   "
         f"Confidence: {conf*100:.1f}%",
         ha="center", va="top",
         fontsize=12, color="#999999")

# ── 4 panels ──
# [left, bottom, width, height] in figure-fraction units
panels = [
    (0.02,  0.10, 0.22, 0.72),
    (0.265, 0.10, 0.22, 0.72),
    (0.51,  0.10, 0.22, 0.72),
    (0.755, 0.10, 0.22, 0.72),
]
labels = [
    "Original image",
    "Important regions\n(dimmed = ignored)",
    "Superpixel boundaries\n(green = important)",
    "Saliency heatmap\n(green = supports class)",
]
images = [raw_up, highlight_up, boundary_up, heat_up]

for (l, b, w, h), img, lbl in zip(panels, images, labels):
    ax = fig.add_axes([l, b, w, h])
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.5, -0.04, lbl,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, color="#cccccc",
            linespacing=1.5)

# ── Colourbar for the saliency heatmap panel ──
cbar_ax = fig.add_axes([0.755, 0.04, 0.22, 0.025])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cbar_ax.imshow(gradient, aspect="auto", cmap="RdYlGn",
               extent=[0, 1, 0, 1])
cbar_ax.set_yticks([])
cbar_ax.set_xticks([0, 0.5, 1])
cbar_ax.set_xticklabels(["Against class", "Neutral", "Supports class"],
                         color="#aaaaaa", fontsize=8)
cbar_ax.tick_params(colors="#555555", length=3)
for sp in cbar_ax.spines.values():
    sp.set_edgecolor("#333333")

# ── LIME method note ──
fig.text(0.5, 0.02,
         "LIME: masks superpixel regions and fits a local linear model "
         "to find which regions drive the prediction",
         ha="center", va="bottom",
         fontsize=9, color="#666666", style="italic")


# ─────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────
save_path = os.path.join(OUT_DIR,
                         f"linkedin_lime_{CLASS_TO_SHOW}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()

print(f"✅  Saved  →  {save_path}")
print(f"   300 DPI  |  Open from File Explorer to view")
