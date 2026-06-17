import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# ─────────────────────────────────────────────
#  0.  Config
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
DATA_DIR    = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\data"
MODEL_PATH  = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\resnet18_cifar10.pth"
OUT_DIR     = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\xai_prototype_outputs"

N_PROTOTYPES       = 5     # nearest neighbours to show per test image
SAMPLES_PER_CLASS  = 1     # test images to explain per class
N_TSNE_SAMPLES     = 500   # images for t-SNE plot (per class)

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
    transforms.ToTensor(),
])

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def to_display(tensor):
    """(3,H,W) any tensor → (H,W,3) uint8 numpy for imshow."""
    t = tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


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
#  3.  Feature extractor
#      Strip the FC head → outputs 512-d embedding per image
# ─────────────────────────────────────────────
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(DEVICE).eval()


def extract_features(dataset, desc="Extracting features"):
    """
    Returns:
        features : (N, 512) numpy array
        labels   : (N,)     numpy array
        raw_imgs : list of (3,H,W) float tensors [0,1]  for display
    """
    loader = DataLoader(dataset, batch_size=128,
                        shuffle=False, num_workers=0)
    all_feats, all_labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=desc, ncols=80):
            feats = feature_extractor(imgs.to(DEVICE))   # (B,512,1,1)
            feats = feats.squeeze(-1).squeeze(-1)         # (B,512)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())

    return np.vstack(all_feats), np.concatenate(all_labels)


# ─────────────────────────────────────────────
#  4.  Load datasets & extract features
# ─────────────────────────────────────────────
train_dataset_norm = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True,  download=True, transform=test_transform)
test_dataset_norm  = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=test_transform)
train_dataset_raw  = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True,  download=False, transform=raw_transform)
test_dataset_raw   = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=False, transform=raw_transform)

print("Extracting training features...")
train_feats, train_labels = extract_features(train_dataset_norm,
                                             "Train features")
print(f"  Train feature matrix : {train_feats.shape}")

print("Extracting test features...")
test_feats, test_labels = extract_features(test_dataset_norm,
                                           "Test features")
print(f"  Test  feature matrix : {test_feats.shape}\n")


# ─────────────────────────────────────────────
#  5.  Fit Nearest-Neighbour index on training features
# ─────────────────────────────────────────────
nn_index = NearestNeighbors(n_neighbors=N_PROTOTYPES,
                            metric="cosine", algorithm="brute")
nn_index.fit(train_feats)
print(f"Nearest-neighbour index built  "
      f"(k={N_PROTOTYPES}, metric=cosine)\n")


# ─────────────────────────────────────────────
#  6.  Helper — predict with confidence
# ─────────────────────────────────────────────
def predict(norm_tensor_3d):
    """(3,H,W) normalised tensor → (pred_class, confidence)"""
    with torch.no_grad():
        logits = model(norm_tensor_3d.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax().item()
    return pred, probs[pred].item(), probs.cpu().numpy()


# ─────────────────────────────────────────────
#  7.  Select one test image per class
# ─────────────────────────────────────────────
class_indices = {}
for idx, (_, label) in enumerate(test_dataset_raw):
    if label not in class_indices:
        class_indices[label] = idx
    if len(class_indices) == NUM_CLASSES:
        break

sample_indices = [class_indices[c] for c in range(NUM_CLASSES)]


# ─────────────────────────────────────────────
#  8.  Prototype explanation — main loop
# ─────────────────────────────────────────────
print("── Generating Prototype Explanations ────────────────────────")

all_proto_results = []

for idx in sample_indices:
    raw_tensor,  true_label = test_dataset_raw[idx]
    norm_tensor, _          = test_dataset_norm[idx]

    pred_class, confidence, all_probs = predict(norm_tensor)

    # Feature vector of the test image
    with torch.no_grad():
        test_feat = feature_extractor(
            norm_tensor.unsqueeze(0).to(DEVICE)
        ).squeeze().cpu().numpy()   # (512,)

    # Find k nearest training images
    distances, nn_indices = nn_index.kneighbors(test_feat.reshape(1, -1))
    distances  = distances[0]    # (k,)
    nn_indices = nn_indices[0]   # (k,)

    # Collect prototype images and their labels
    prototypes = []
    for dist, tr_idx in zip(distances, nn_indices):
        proto_raw,  proto_label = train_dataset_raw[tr_idx]
        prototypes.append({
            "raw_tensor"  : proto_raw,
            "label"       : proto_label,
            "class_name"  : CIFAR10_CLASSES[proto_label],
            "distance"    : dist,
            "similarity"  : 1 - dist,     # cosine similarity
        })

    result = {
        "idx"         : idx,
        "true_label"  : true_label,
        "pred_label"  : pred_class,
        "true_name"   : CIFAR10_CLASSES[true_label],
        "pred_name"   : CIFAR10_CLASSES[pred_class],
        "confidence"  : confidence,
        "all_probs"   : all_probs,
        "correct"     : pred_class == true_label,
        "raw_tensor"  : raw_tensor,
        "prototypes"  : prototypes,
    }
    all_proto_results.append(result)

    correct_sym = "✔" if result["correct"] else "✘"
    print(f"  [{correct_sym}] {result['true_name']:<12} → pred: {result['pred_name']:<12} "
          f"({confidence*100:.1f}%)  |  top proto: {prototypes[0]['class_name']} "
          f"(sim={prototypes[0]['similarity']:.3f})")


# ─────────────────────────────────────────────
#  9.  Figure A — Prototype panels (one per class)
#      Layout: [Test image | Prototype 1..5 | Confidence bar]
# ─────────────────────────────────────────────
print("\n── Saving Prototype Panels ───────────────────────────────────")

for result in all_proto_results:
    n_protos = len(result["prototypes"])
    # Columns: original + N prototypes + confidence bar
    fig = plt.figure(figsize=(3.5 * (n_protos + 2), 5))
    gs  = gridspec.GridSpec(1, n_protos + 2, figure=fig,
                            wspace=0.08, hspace=0)

    correct_sym = "✔" if result["correct"] else "✘"
    fig.suptitle(
        f"Prototype Explanation  —  True: {result['true_name']}  "
        f"|  Predicted: {result['pred_name']} {correct_sym}  "
        f"({result['confidence']*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )

    # ── Test image ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(to_display(result["raw_tensor"]))
    ax0.set_title("Query\nImage", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # ── Prototypes ──────────────────────────────────────────
    for p_idx, proto in enumerate(result["prototypes"]):
        ax = fig.add_subplot(gs[0, p_idx + 1])
        ax.imshow(to_display(proto["raw_tensor"]))

        # Green frame if same class, red if different
        frame_color = "#2a9d8f" if proto["label"] == result["true_label"] \
                      else "#e63946"
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(3)

        ax.set_title(
            f"#{p_idx+1}  {proto['class_name']}\n"
            f"sim={proto['similarity']:.3f}",
            fontsize=8
        )
        ax.axis("off")

    # ── Confidence bar chart ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, -1])
    probs  = result["all_probs"]
    y_pos  = np.arange(NUM_CLASSES)
    colors_bar = ["#e63946" if i == result["pred_label"] else "#a8dadc"
                  for i in range(NUM_CLASSES)]
    ax_bar.barh(y_pos, probs, color=colors_bar)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Confidence", fontsize=8)
    ax_bar.set_title("Model\nOutput", fontsize=9)
    ax_bar.invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"proto_{result['true_name']}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → proto_{result['true_name']}.png")


# ─────────────────────────────────────────────
#  10. Figure B — Combined prototype dashboard
#      All 10 classes in one figure
# ─────────────────────────────────────────────
print("\n── Building Combined Prototype Dashboard ─────────────────────")

n_rows   = NUM_CLASSES
n_cols   = 1 + N_PROTOTYPES    # query + prototypes

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 2.8, n_rows * 2.8))
fig.suptitle("Prototype-Based Explanations — All Classes\n"
             "Green border = same class  |  Red border = different class",
             fontsize=13, fontweight="bold")

col_headers = ["Query"] + [f"Prototype #{i+1}" for i in range(N_PROTOTYPES)]
for col, title in enumerate(col_headers):
    axes[0][col].set_title(title, fontsize=9, fontweight="bold")

for row, result in enumerate(all_proto_results):
    correct_sym = "✔" if result["correct"] else "✘"

    # Query image
    axes[row][0].imshow(to_display(result["raw_tensor"]))
    axes[row][0].set_ylabel(
        f"{result['true_name']}\npred:{result['pred_name']}{correct_sym}",
        fontsize=7, rotation=0, labelpad=60, va="center"
    )
    axes[row][0].axis("off")

    # Prototype images
    for p_idx, proto in enumerate(result["prototypes"]):
        ax = axes[row][p_idx + 1]
        ax.imshow(to_display(proto["raw_tensor"]))
        frame_color = "#2a9d8f" if proto["label"] == result["true_label"] \
                      else "#e63946"
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(2.5)
        ax.set_xlabel(f"{proto['class_name']}\n{proto['similarity']:.2f}",
                      fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
dashboard_path = os.path.join(OUT_DIR, "prototype_dashboard.png")
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Dashboard saved → {dashboard_path}")


# ─────────────────────────────────────────────
#  11. Figure C — t-SNE Feature Space Visualisation
#      Shows how the model groups classes in 512-d space
# ─────────────────────────────────────────────
print("\n── Running t-SNE on Feature Space ───────────────────────────")
print(f"   Using {N_TSNE_SAMPLES} samples per class  "
      f"({N_TSNE_SAMPLES * NUM_CLASSES} total)...")

# Subsample for speed
tsne_feats, tsne_labels = [], []
for cls in range(NUM_CLASSES):
    cls_mask = train_labels == cls
    cls_feats = train_feats[cls_mask][:N_TSNE_SAMPLES]
    tsne_feats.append(cls_feats)
    tsne_labels.append(np.full(len(cls_feats), cls))

tsne_feats  = np.vstack(tsne_feats)
tsne_labels = np.concatenate(tsne_labels)

# PCA first to 50 dims (speeds up t-SNE significantly)
print("   Running PCA → 50 dims...")
pca       = PCA(n_components=50, random_state=42)
feats_pca = pca.fit_transform(tsne_feats)

print("   Running t-SNE → 2 dims  (may take ~2 min)...")
tsne      = TSNE(n_components=2, perplexity=40, max_iter=1000,
                 random_state=42, verbose=0)
feats_2d  = tsne.fit_transform(feats_pca)

# Colour palette — one per class
palette = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#264653", "#a8dadc", "#6d6875", "#b5838d", "#e76f51"
]

fig, ax = plt.subplots(figsize=(12, 10))
for cls in range(NUM_CLASSES):
    mask = tsne_labels == cls
    ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
               c=palette[cls], label=CIFAR10_CLASSES[cls],
               alpha=0.55, s=12, edgecolors="none")

# Mark the test query images on the t-SNE plot
for result in all_proto_results:
    with torch.no_grad():
        qf = feature_extractor(
            test_dataset_norm[result["idx"]][0]
                .unsqueeze(0).to(DEVICE)
        ).squeeze().cpu().numpy()
    # Project via PCA then approximate location in t-SNE space
    # (exact t-SNE placement requires re-running; we use PCA projection as proxy)
    qf_pca = pca.transform(qf.reshape(1, -1))
    ax.scatter(qf_pca[0, 0] * 0, qf_pca[0, 1] * 0, s=0)   # skip exact overlay

ax.set_title("t-SNE of ResNet-18 Feature Space (CIFAR-10)\n"
             "Each dot = one training image | Well-separated clusters = good features",
             fontsize=12, fontweight="bold")
ax.legend(markerscale=3, fontsize=9, loc="upper right",
          bbox_to_anchor=(1.15, 1))
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.axis("off")

plt.tight_layout()
tsne_path = os.path.join(OUT_DIR, "tsne_feature_space.png")
plt.savefig(tsne_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"t-SNE saved → {tsne_path}")


# ─────────────────────────────────────────────
#  12. Prototype purity analysis
#      For each test image: what % of its 5 nearest neighbours
#      share the TRUE class label?
# ─────────────────────────────────────────────
print("\n── Prototype Purity Analysis ─────────────────────────────────")
print(f"{'Class':<14} {'Purity':>8}  (% of top-{N_PROTOTYPES} neighbours with same label)")
print("─" * 45)

purity_vals = []
for result in all_proto_results:
    n_correct = sum(1 for p in result["prototypes"]
                    if p["label"] == result["true_label"])
    purity = n_correct / N_PROTOTYPES
    purity_vals.append(purity)
    bar = "█" * int(purity * 20)
    print(f"  {result['true_name']:<12}  {purity*100:>5.0f}%  {bar}")

print("─" * 45)
print(f"  {'AVERAGE':<12}  {np.mean(purity_vals)*100:>5.0f}%")

print("\n📌  High purity → model learned a clean class representation")
print("    Low purity  → classes with visual overlap (e.g. cat/dog)\n")

# Bar chart for purity
fig, ax = plt.subplots(figsize=(12, 4))
cls_names   = [r["true_name"] for r in all_proto_results]
bar_colors  = [palette[r["true_label"]] for r in all_proto_results]
ax.bar(cls_names, [v * 100 for v in purity_vals], color=bar_colors)
ax.axhline(y=np.mean(purity_vals) * 100, color="gray", linestyle="--",
           alpha=0.7, label=f"Average: {np.mean(purity_vals)*100:.0f}%")
ax.set_ylabel("Prototype Purity (%)")
ax.set_title(f"Prototype Purity — % of top-{N_PROTOTYPES} neighbours "
             f"matching true class")
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
purity_path = os.path.join(OUT_DIR, "prototype_purity.png")
plt.savefig(purity_path, dpi=150)
plt.close()
print(f"Purity chart saved → {purity_path}")


# ─────────────────────────────────────────────
#  13. Summary
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("  PROTOTYPE MODULE COMPLETE")
print("═" * 60)
print(f"\n  Average prototype purity : {np.mean(purity_vals)*100:.0f}%")
print(f"\n  Files saved to → {OUT_DIR}")
print("    • proto_<class>.png          individual panels")
print("    • prototype_dashboard.png    all-classes grid")
print("    • tsne_feature_space.png     feature space map")
print("    • prototype_purity.png       purity bar chart")
print("═" * 60)
