import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

# LIME & SHAP
from lime import lime_image
import shap

# ─────────────────────────────────────────────
#  0.  Config  (must match main.py & xai_saliency.py)
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
DATA_DIR    = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\data"
MODEL_PATH  = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\resnet18_cifar10.pth"
OUT_DIR     = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\xai_metrics_outputs"

# Number of test images evaluated per class (increase for stronger results)
SAMPLES_PER_CLASS = 10
DELETION_STEPS    = 10    # granularity of deletion / insertion curves
BLUR_SIGMA        = 10    # gaussian blur sigma for insertion baseline

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck"
]

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using device : {DEVICE}")
print(f"Outputs      → {OUT_DIR}\n")


# ─────────────────────────────────────────────
#  1.  Transforms
# ─────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

raw_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),   # [0,1] float, no normalisation
])

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def normalise(raw_tensor):
    """(3,H,W) [0,1] float → normalised tensor for model."""
    return (raw_tensor - MEAN) / STD

def denormalize_np(norm_tensor):
    """(3,H,W) normalised tensor → (H,W,3) uint8 numpy."""
    t = norm_tensor.cpu() * STD + MEAN
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  2.  Load model
# ─────────────────────────────────────────────
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

model = load_model(MODEL_PATH)
print(f"Model loaded ← {MODEL_PATH}\n")


# ─────────────────────────────────────────────
#  3.  Load datasets
# ─────────────────────────────────────────────
test_dataset_raw  = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=raw_transform)
test_dataset_norm = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=test_transform)
train_dataset_norm = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=False, transform=test_transform)

bg_loader = DataLoader(train_dataset_norm, batch_size=100,
                       shuffle=True, num_workers=0)

# Gather SAMPLES_PER_CLASS indices per class
class_indices = {c: [] for c in range(NUM_CLASSES)}
for idx, (_, label) in enumerate(test_dataset_raw):
    if len(class_indices[label]) < SAMPLES_PER_CLASS:
        class_indices[label].append(idx)
    if all(len(v) == SAMPLES_PER_CLASS for v in class_indices.values()):
        break

sample_indices = [idx for cls in range(NUM_CLASSES)
                  for idx in class_indices[cls]]
print(f"Evaluating {len(sample_indices)} images "
      f"({SAMPLES_PER_CLASS} per class × {NUM_CLASSES} classes)\n")


# ─────────────────────────────────────────────
#  4.  Grad-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, image_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(image_tensor)
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1).squeeze())
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-8)
        # Upsample to input size
        cam_up  = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        return cam_up

gradcam = GradCAM(model, model.layer4[-1])


# ─────────────────────────────────────────────
#  5.  LIME saliency map (returns normalised float map)
# ─────────────────────────────────────────────
def get_lime_map(raw_img_np, pred_class, num_samples=300):
    """raw_img_np: (H,W,3) uint8  →  (H,W) float [0,1]"""
    def predict_fn(images):
        tensors = [normalise(torch.tensor(img / 255.0,
                   dtype=torch.float32).permute(2, 0, 1))
                   for img in images]
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(model(batch), dim=1).cpu().numpy()

    explainer   = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        raw_img_np, predict_fn,
        top_labels=1, hide_color=0,
        num_samples=num_samples, random_seed=42,
    )
    # Get per-superpixel weights for the predicted class
    local_exp = explanation.local_exp[pred_class]
    seg       = explanation.segments          # (H,W) int superpixel labels
    sal_map   = np.zeros(seg.shape, dtype=np.float32)
    for sp_id, weight in local_exp:
        sal_map[seg == sp_id] = weight

    # Shift to [0,1]
    sal_map = sal_map - sal_map.min()
    sal_map = sal_map / (sal_map.max() + 1e-8)
    return sal_map


# ─────────────────────────────────────────────
#  6.  SHAP saliency map
#      Includes SafeBasicBlock fix for ResNet inplace ops
# ─────────────────────────────────────────────

class SafeBasicBlock(nn.Module):
    """ResNet BasicBlock with all inplace operations removed for SHAP."""
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
        out = out + identity    # fix 2: out + identity NOT out += identity
        out = self.relu(out)
        return out


def make_shap_safe_model(trained_model):
    """Deep-copy model, replace all BasicBlocks with SafeBasicBlock."""
    safe = copy.deepcopy(trained_model)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        old_layer = getattr(safe, layer_name)
        setattr(safe, layer_name,
                nn.Sequential(*[SafeBasicBlock(b) for b in old_layer]))
    safe.relu = nn.ReLU(inplace=False)
    safe.eval()
    return safe


def build_shap_explainer(n=100):
    shap_model = make_shap_safe_model(model)
    bg = []
    for imgs, _ in bg_loader:
        bg.append(imgs)
        if sum(x.shape[0] for x in bg) >= n:
            break
    background = torch.cat(bg)[:n].to(DEVICE)
    return shap.DeepExplainer(shap_model, background)

print("Building SHAP explainer (safe model)...")
shap_explainer = build_shap_explainer(n=100)
print("SHAP explainer ready.\n")


def get_shap_map(norm_tensor, pred_class):
    """norm_tensor: (1,3,H,W) on DEVICE  →  (H,W) float [0,1]"""
    raw_sv = np.array(shap_explainer.shap_values(
        norm_tensor.detach().clone(), check_additivity=False))

    # Handle all known SHAP output shapes
    if raw_sv.ndim == 5:
        # (1, 3, H, W, NUM_CLASSES)  ← your SHAP version
        sv = raw_sv[0, :, :, :, pred_class]       # (3, H, W)
    elif raw_sv.ndim == 4 and raw_sv.shape[-1] == NUM_CLASSES:
        # (1, H, W, NUM_CLASSES)
        sv = raw_sv[0, :, :, pred_class]           # (H, W)
        sv = np.stack([sv, sv, sv], axis=0)        # (3, H, W)
    elif raw_sv.ndim == 4:
        # (1, 3, H, W)
        sv = raw_sv[0]                             # (3, H, W)
    else:
        sv = raw_sv[pred_class][0]                 # fallback list format

    sv_map = np.abs(sv).mean(axis=0)               # (H, W)
    sv_map = (sv_map - sv_map.min()) / (sv_map.max() - sv_map.min() + 1e-8)
    return sv_map


# ─────────────────────────────────────────────
#  7.  Core Metrics
# ─────────────────────────────────────────────

def get_confidence(model, tensor_1chw):
    """Returns predicted class and its softmax confidence."""
    with torch.no_grad():
        probs = torch.softmax(model(tensor_1chw), dim=1)[0]
    pred  = probs.argmax().item()
    return pred, probs[pred].item()


def deletion_auc(model, raw_tensor, saliency_map, pred_class,
                 steps=DELETION_STEPS):
    """
    Mask top-K% most salient pixels (set to 0) in increasing steps.
    Measure model confidence on predicted class at each step.
    Lower AUC = better explanation (masking important pixels hurts more).

    raw_tensor   : (3,H,W) float [0,1] UN-normalised
    saliency_map : (H,W)   float [0,1]
    Returns      : (auc_score, step_confidences)
    """
    flat_sal  = saliency_map.flatten()
    sorted_idx = np.argsort(flat_sal)[::-1]          # most important first
    total      = len(sorted_idx)
    step_confs = []

    for step in range(steps + 1):
        k      = int((step / steps) * total)
        masked = raw_tensor.clone().cpu().numpy().copy()
        masked_flat = masked.reshape(3, -1)
        masked_flat[:, sorted_idx[:k]] = 0            # zero out top-k pixels
        masked_tensor = torch.tensor(
            masked_flat.reshape(raw_tensor.shape), dtype=torch.float32)
        norm_masked = normalise(masked_tensor).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            conf = torch.softmax(model(norm_masked), dim=1)[0, pred_class].item()
        step_confs.append(conf)

    x     = np.linspace(0, 1, steps + 1)
    score = auc(x, step_confs)
    return score, step_confs


def insertion_auc(model, raw_tensor, saliency_map, pred_class,
                  steps=DELETION_STEPS):
    """
    Start from a blurred image, progressively reveal top-K% salient pixels.
    Measure confidence at each step.
    Higher AUC = better explanation (inserting important pixels helps more).

    Returns: (auc_score, step_confidences)
    """
    # Blurred baseline (gaussian blur applied per channel)
    blurred_np = raw_tensor.cpu().numpy().copy()
    for c in range(3):
        blurred_np[c] = gaussian_filter(blurred_np[c], sigma=BLUR_SIGMA)

    flat_sal   = saliency_map.flatten()
    sorted_idx = np.argsort(flat_sal)[::-1]
    total      = len(sorted_idx)
    step_confs = []
    raw_np     = raw_tensor.cpu().numpy()

    for step in range(steps + 1):
        k        = int((step / steps) * total)
        revealed = blurred_np.copy().reshape(3, -1)
        revealed[:, sorted_idx[:k]] = raw_np.reshape(3, -1)[:, sorted_idx[:k]]
        revealed_tensor = torch.tensor(
            revealed.reshape(raw_tensor.shape), dtype=torch.float32)
        norm_rev = normalise(revealed_tensor).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            conf = torch.softmax(model(norm_rev), dim=1)[0, pred_class].item()
        step_confs.append(conf)

    x     = np.linspace(0, 1, steps + 1)
    score = auc(x, step_confs)
    return score, step_confs


def sanity_check_similarity(original_map, random_map):
    """
    Pixel-wise Spearman rank correlation between two saliency maps.
    If a method is truly model-dependent, this should be LOW after randomisation.
    Returns float in [-1, 1].
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(original_map.flatten(), random_map.flatten())
    return corr


# ─────────────────────────────────────────────
#  8.  Sanity-check model — randomise last layer weights
#      (Adebayo et al. 2018 — "Sanity Checks for Saliency Maps")
# ─────────────────────────────────────────────
def build_random_model(trained_model):
    """
    Clone the trained model and randomly reinitialise only the FC layer.
    A good XAI method should produce DIFFERENT maps for this model.
    """
    rand_model = copy.deepcopy(trained_model)
    nn.init.kaiming_uniform_(rand_model.fc.weight)
    nn.init.zeros_(rand_model.fc.bias)
    rand_model.eval()
    return rand_model

rand_model  = build_random_model(model)
rand_gradcam = GradCAM(rand_model, rand_model.layer4[-1])
print("Randomised model built for sanity checks.\n")


# ─────────────────────────────────────────────
#  9.  Main evaluation loop
# ─────────────────────────────────────────────
print("── Running Quantitative Evaluation ──────────────────────────")
print(f"   Metrics: Deletion AUC ↓ | Insertion AUC ↑ | Sanity Corr ↓\n")

# Storage for per-class aggregation
records = []   # list of dicts, one per image

for i, idx in enumerate(tqdm(sample_indices, desc="Evaluating", ncols=80)):
    raw_tensor, true_label  = test_dataset_raw[idx]    # (3,224,224) [0,1]
    norm_tensor, _          = test_dataset_norm[idx]   # (3,224,224) normalised

    norm_input = norm_tensor.unsqueeze(0).to(DEVICE)   # (1,3,224,224)

    pred_class, confidence  = get_confidence(model, norm_input)

    # ── Saliency maps ────────────────────────
    raw_np     = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    gc_map     = gradcam.generate(norm_input, pred_class)
    lime_map   = get_lime_map(raw_np, pred_class, num_samples=200)
    shap_map   = get_shap_map(norm_input, pred_class)

    # ── Deletion AUC ─────────────────────────
    del_gc,   del_gc_curve   = deletion_auc(model, raw_tensor, gc_map,   pred_class)
    del_lime, del_lime_curve = deletion_auc(model, raw_tensor, lime_map, pred_class)
    del_shap, del_shap_curve = deletion_auc(model, raw_tensor, shap_map, pred_class)

    # ── Insertion AUC ────────────────────────
    ins_gc,   ins_gc_curve   = insertion_auc(model, raw_tensor, gc_map,   pred_class)
    ins_lime, ins_lime_curve = insertion_auc(model, raw_tensor, lime_map, pred_class)
    ins_shap, ins_shap_curve = insertion_auc(model, raw_tensor, shap_map, pred_class)

    # ── Sanity Check — randomised model maps ─
    rand_gc_map   = rand_gradcam.generate(norm_input, pred_class)
    sanity_gc     = sanity_check_similarity(gc_map, rand_gc_map)
    # For LIME & SHAP sanity: reuse randomised GradCAM as proxy
    # (full LIME/SHAP on random model is expensive — GradCAM is the standard)
    sanity_lime   = None   # placeholder — discussed in report
    sanity_shap   = None

    records.append({
        "true_label"     : true_label,
        "pred_label"     : pred_class,
        "correct"        : pred_class == true_label,
        "confidence"     : confidence,
        # Deletion AUC (lower = better)
        "del_gc"         : del_gc,
        "del_lime"       : del_lime,
        "del_shap"       : del_shap,
        # Insertion AUC (higher = better)
        "ins_gc"         : ins_gc,
        "ins_lime"       : ins_lime,
        "ins_shap"       : ins_shap,
        # Sanity check correlation (lower = better, method is model-sensitive)
        "sanity_gc"      : sanity_gc,
        # Store curves for first sample of each class (for plotting)
        "del_gc_curve"   : del_gc_curve   if i % SAMPLES_PER_CLASS == 0 else None,
        "del_lime_curve" : del_lime_curve if i % SAMPLES_PER_CLASS == 0 else None,
        "del_shap_curve" : del_shap_curve if i % SAMPLES_PER_CLASS == 0 else None,
        "ins_gc_curve"   : ins_gc_curve   if i % SAMPLES_PER_CLASS == 0 else None,
        "ins_lime_curve" : ins_lime_curve if i % SAMPLES_PER_CLASS == 0 else None,
        "ins_shap_curve" : ins_shap_curve if i % SAMPLES_PER_CLASS == 0 else None,
    })


# ─────────────────────────────────────────────
#  10.  Aggregate per-class results
# ─────────────────────────────────────────────
print("\n── Per-Class Results ─────────────────────────────────────────")
print(f"\n{'Class':<12} │ {'Del↓ GC':>8} {'Del↓ LI':>8} {'Del↓ SH':>8} "
      f"│ {'Ins↑ GC':>8} {'Ins↑ LI':>8} {'Ins↑ SH':>8} "
      f"│ {'Sanity↓':>8}")
print("─" * 85)

class_summaries = []
for cls_idx in range(NUM_CLASSES):
    cls_records = [r for r in records if r["true_label"] == cls_idx]
    if not cls_records:
        continue

    avg = lambda key: np.mean([r[key] for r in cls_records])

    row = {
        "class"     : CIFAR10_CLASSES[cls_idx],
        "del_gc"    : avg("del_gc"),
        "del_lime"  : avg("del_lime"),
        "del_shap"  : avg("del_shap"),
        "ins_gc"    : avg("ins_gc"),
        "ins_lime"  : avg("ins_lime"),
        "ins_shap"  : avg("ins_shap"),
        "sanity_gc" : avg("sanity_gc"),
    }
    class_summaries.append(row)

    print(f"{row['class']:<12} │ "
          f"{row['del_gc']:>8.3f} {row['del_lime']:>8.3f} {row['del_shap']:>8.3f} │ "
          f"{row['ins_gc']:>8.3f} {row['ins_lime']:>8.3f} {row['ins_shap']:>8.3f} │ "
          f"{row['sanity_gc']:>8.3f}")

# Overall averages
print("─" * 85)
overall_avg = lambda key: np.mean([r[key] for r in class_summaries])
print(f"{'AVERAGE':<12} │ "
      f"{overall_avg('del_gc'):>8.3f} {overall_avg('del_lime'):>8.3f} "
      f"{overall_avg('del_shap'):>8.3f} │ "
      f"{overall_avg('ins_gc'):>8.3f} {overall_avg('ins_lime'):>8.3f} "
      f"{overall_avg('ins_shap'):>8.3f} │ "
      f"{overall_avg('sanity_gc'):>8.3f}")

print("\n📌 Key:")
print("   Del↓  = Deletion AUC  (lower is better — masking hurts confidence)")
print("   Ins↑  = Insertion AUC (higher is better — revealing helps confidence)")
print("   Sanity↓ = Corr with random model (lower = method is model-sensitive ✔)")


# ─────────────────────────────────────────────
#  11.  Summary comparison table figure
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Quantitative XAI Evaluation — CIFAR-10 ResNet-18",
             fontsize=14, fontweight="bold")

class_names  = [r["class"] for r in class_summaries]
x            = np.arange(len(class_names))
width        = 0.28
colors       = {"GradCAM": "#e63946", "LIME": "#457b9d", "SHAP": "#2a9d8f"}

# ── Plot 1: Deletion AUC (lower = better) ──
ax = axes[0]
ax.bar(x - width, [r["del_gc"]   for r in class_summaries],
       width, label="GradCAM", color=colors["GradCAM"])
ax.bar(x,         [r["del_lime"] for r in class_summaries],
       width, label="LIME",    color=colors["LIME"])
ax.bar(x + width, [r["del_shap"] for r in class_summaries],
       width, label="SHAP",    color=colors["SHAP"])
ax.set_title("Deletion AUC ↓  (lower = better)")
ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=25, ha="right")
ax.set_ylabel("AUC"); ax.set_ylim(0, 1); ax.legend()

# ── Plot 2: Insertion AUC (higher = better) ──
ax = axes[1]
ax.bar(x - width, [r["ins_gc"]   for r in class_summaries],
       width, label="GradCAM", color=colors["GradCAM"])
ax.bar(x,         [r["ins_lime"] for r in class_summaries],
       width, label="LIME",    color=colors["LIME"])
ax.bar(x + width, [r["ins_shap"] for r in class_summaries],
       width, label="SHAP",    color=colors["SHAP"])
ax.set_title("Insertion AUC ↑  (higher = better)")
ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=25, ha="right")
ax.set_ylabel("AUC"); ax.set_ylim(0, 1); ax.legend()

# ── Plot 3: Sanity check correlation ──
ax = axes[2]
sanity_vals = [r["sanity_gc"] for r in class_summaries]
bar_colors  = ["#2a9d8f" if v < 0.5 else "#e63946" for v in sanity_vals]
ax.bar(x, sanity_vals, color=bar_colors)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7,
           label="threshold (0.5)")
ax.set_title("Sanity Check — GradCAM\nCorr with Randomised Model ↓")
ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=25, ha="right")
ax.set_ylabel("Spearman Correlation")
ax.set_ylim(-0.1, 1.1)
ax.legend()

plt.tight_layout()
table_path = os.path.join(OUT_DIR, "metrics_comparison_table.png")
plt.savefig(table_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nComparison table saved → {table_path}")


# ─────────────────────────────────────────────
#  12.  Deletion & Insertion curves
#       One figure per class (first sample of each class)
# ─────────────────────────────────────────────
print("\n── Plotting Deletion / Insertion Curves ──────────────────────")

curve_records = [r for r in records if r["del_gc_curve"] is not None]
steps_x       = np.linspace(0, 1, DELETION_STEPS + 1)

fig, axes = plt.subplots(
    len(curve_records), 2,
    figsize=(12, 3.5 * len(curve_records))
)
if len(curve_records) == 1:
    axes = [axes]   # make iterable

fig.suptitle("Deletion & Insertion Curves per Class",
             fontsize=14, fontweight="bold")

for row_idx, rec in enumerate(curve_records):
    cls_name = CIFAR10_CLASSES[rec["true_label"]]

    # Deletion
    ax = axes[row_idx][0]
    ax.plot(steps_x, rec["del_gc_curve"],   label="GradCAM", color="#e63946", lw=2)
    ax.plot(steps_x, rec["del_lime_curve"], label="LIME",    color="#457b9d", lw=2)
    ax.plot(steps_x, rec["del_shap_curve"], label="SHAP",    color="#2a9d8f", lw=2)
    ax.set_title(f"{cls_name} — Deletion (↓ drops faster = better)")
    ax.set_xlabel("Fraction of pixels masked")
    ax.set_ylabel("Model confidence")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

    # Insertion
    ax = axes[row_idx][1]
    ax.plot(steps_x, rec["ins_gc_curve"],   label="GradCAM", color="#e63946", lw=2)
    ax.plot(steps_x, rec["ins_lime_curve"], label="LIME",    color="#457b9d", lw=2)
    ax.plot(steps_x, rec["ins_shap_curve"], label="SHAP",    color="#2a9d8f", lw=2)
    ax.set_title(f"{cls_name} — Insertion (↑ rises faster = better)")
    ax.set_xlabel("Fraction of pixels revealed")
    ax.set_ylabel("Model confidence")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "deletion_insertion_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Curves saved → {curves_path}")


# ─────────────────────────────────────────────
#  13.  Sanity Check Visual — side-by-side maps
#       Trained model vs Randomised model GradCAM
# ─────────────────────────────────────────────
print("\n── Generating Sanity Check Visuals ───────────────────────────")

# Pick one image per class (first in each class block)
sanity_indices = [class_indices[c][0] for c in range(NUM_CLASSES)]

n = len(sanity_indices)
fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
fig.suptitle(
    "Sanity Check — Trained vs Randomised Model GradCAM\n"
    "Good method = maps should DIFFER after randomisation",
    fontsize=13, fontweight="bold"
)

for row, idx in enumerate(sanity_indices):
    raw_tensor,  true_label = test_dataset_raw[idx]
    norm_tensor, _          = test_dataset_norm[idx]
    norm_input              = norm_tensor.unsqueeze(0).to(DEVICE)
    raw_np = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    pred_class, _ = get_confidence(model, norm_input)

    # Trained model GradCAM
    trained_map   = gradcam.generate(norm_input, pred_class)
    # Randomised model GradCAM
    rand_map      = rand_gradcam.generate(norm_input, pred_class)
    corr          = sanity_check_similarity(trained_map, rand_map)

    axes[row][0].imshow(raw_np)
    axes[row][0].set_title(f"{CIFAR10_CLASSES[true_label]}")

    axes[row][1].imshow(trained_map, cmap="jet")
    axes[row][1].set_title("Trained model")

    axes[row][2].imshow(rand_map, cmap="jet")
    axes[row][2].set_title(f"Randomised  |  Corr: {corr:.3f}")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
sanity_path = os.path.join(OUT_DIR, "sanity_check_visual.png")
plt.savefig(sanity_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Sanity visual saved → {sanity_path}")


# ─────────────────────────────────────────────
#  14.  Final summary printout
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("  WEEK 3 COMPLETE — Summary")
print("═" * 60)
print(f"\n  Best Method by Deletion AUC  ↓ (lower = better):")
best_del = min(
    [("GradCAM", overall_avg("del_gc")),
     ("LIME",    overall_avg("del_lime")),
     ("SHAP",    overall_avg("del_shap"))],
    key=lambda x: x[1]
)
print(f"    → {best_del[0]}  (AUC = {best_del[1]:.3f})")

print(f"\n  Best Method by Insertion AUC ↑ (higher = better):")
best_ins = max(
    [("GradCAM", overall_avg("ins_gc")),
     ("LIME",    overall_avg("ins_lime")),
     ("SHAP",    overall_avg("ins_shap"))],
    key=lambda x: x[1]
)
print(f"    → {best_ins[0]}  (AUC = {best_ins[1]:.3f})")

print(f"\n  GradCAM Sanity Score (avg Spearman corr with random model):")
print(f"    → {overall_avg('sanity_gc'):.3f}  "
      f"({'✔ sensitive to model weights' if overall_avg('sanity_gc') < 0.5 else '⚠ check if method is just edge-detecting'})")

print(f"\n  Output files:")
print(f"    • metrics_comparison_table.png")
print(f"    • deletion_insertion_curves.png")
print(f"    • sanity_check_visual.png")
print(f"\n  All saved to → {OUT_DIR}")
print("═" * 60)
