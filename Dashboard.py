import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
from sklearn.neighbors import NearestNeighbors
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from lime import lime_image
import shap
from tqdm import tqdm

# ─────────────────────────────────────────────
#  0.  Config
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
DATA_DIR    = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\data"
MODEL_PATH  = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\resnet18_cifar10.pth"
OUT_DIR     = r"C:\Users\Diljeet\OneDrive\Desktop\new_project\xai_dashboard_outputs"

# Metrics config
DELETION_STEPS   = 10
BLUR_SIGMA       = 10
N_PROTOTYPES     = 5
N_BG_SHAP        = 100
LIME_SAMPLES     = 300

# Which test images to include in dashboard (one per class)
SAMPLES_PER_CLASS = 1

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse", "ship", "truck"
]

PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#264653", "#a8dadc", "#6d6875", "#b5838d", "#e76f51"
]

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using device : {DEVICE}")
print(f"Outputs      → {OUT_DIR}\n")


# ─────────────────────────────────────────────
#  1.  Transforms & helpers
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

def normalise(raw_t):
    return (raw_t - MEAN) / STD

def to_display(t):
    """(3,H,W) any float tensor → (H,W,3) uint8."""
    return (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def to_display_norm(norm_t):
    """Denormalise then convert."""
    return to_display(norm_t * STD + MEAN)


# ─────────────────────────────────────────────
#  2.  Load model + feature extractor
# ─────────────────────────────────────────────
def load_model(path):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()

model             = load_model(MODEL_PATH)
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
print(f"Model loaded ← {MODEL_PATH}\n")


# ─────────────────────────────────────────────
#  3.  Datasets
# ─────────────────────────────────────────────
test_norm  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                download=True,  transform=test_transform)
test_raw   = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                download=False, transform=raw_transform)
train_norm = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                download=False, transform=test_transform)
train_raw  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                download=False, transform=raw_transform)

# One test image per class
class_indices = {}
for idx, (_, lbl) in enumerate(test_raw):
    if lbl not in class_indices:
        class_indices[lbl] = idx
    if len(class_indices) == NUM_CLASSES:
        break
sample_indices = [class_indices[c] for c in range(NUM_CLASSES)]


# ─────────────────────────────────────────────
#  4.  Grad-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, layer):
        self.act = self.grad = None
        layer.register_forward_hook(lambda m,i,o: setattr(self,'act',o.detach()))
        layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self,'grad',go[0].detach()))

    def __call__(self, inp, cls):
        model.zero_grad()
        out = model(inp)
        out[0, cls].backward()
        w   = self.grad.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * self.act).sum(1).squeeze())
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return F.interpolate(cam[None,None], (224,224),
                             mode="bilinear", align_corners=False
                             ).squeeze().cpu().numpy()

gradcam = GradCAM(model, model.layer4[-1])


# ─────────────────────────────────────────────
#  5.  LIME
# ─────────────────────────────────────────────
def get_lime(raw_np, pred_cls):
    def pred_fn(imgs):
        ts = [normalise(torch.tensor(i/255., dtype=torch.float32).permute(2,0,1))
              for i in imgs]
        with torch.no_grad():
            return torch.softmax(model(torch.stack(ts).to(DEVICE)),1).cpu().numpy()
    exp = lime_image.LimeImageExplainer(random_state=42).explain_instance(
        raw_np, pred_fn, top_labels=1, hide_color=0,
        num_samples=LIME_SAMPLES, random_seed=42)
    seg  = exp.segments
    sal  = np.zeros(seg.shape, dtype=np.float32)
    for sp, w in exp.local_exp[pred_cls]:
        sal[seg == sp] = w
    sal = sal - sal.min()
    return sal / (sal.max() + 1e-8)


# ─────────────────────────────────────────────
#  6.  SHAP  — with SafeBasicBlock fix for ResNet inplace ops
# ─────────────────────────────────────────────
import copy

class SafeBasicBlock(nn.Module):
    """ResNet BasicBlock with ALL inplace operations removed for SHAP."""
    expansion = 1
    def __init__(self, src):
        super().__init__()
        self.conv1 = src.conv1; self.bn1 = src.bn1
        self.relu  = nn.ReLU(inplace=False)   # fix 1: no inplace ReLU
        self.conv2 = src.conv2; self.bn2 = src.bn2
        self.downsample = src.downsample; self.stride = src.stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity    # fix 2: out + identity  NOT  out += identity
        return self.relu(out)


def make_shap_safe_model(trained_model):
    safe = copy.deepcopy(trained_model)
    for name in ["layer1","layer2","layer3","layer4"]:
        setattr(safe, name,
                nn.Sequential(*[SafeBasicBlock(b)
                                 for b in getattr(safe, name)]))
    safe.relu = nn.ReLU(inplace=False)
    return safe.eval()


print("Building SHAP background + safe model...")
bg_imgs = []
for imgs, _ in DataLoader(train_norm, batch_size=N_BG_SHAP, shuffle=True):
    bg_imgs.append(imgs); break
background = torch.cat(bg_imgs)[:N_BG_SHAP].to(DEVICE)
shap_model = make_shap_safe_model(model)
shap_exp   = shap.DeepExplainer(shap_model, background)
print("SHAP ready.\n")


def get_shap(norm_inp, pred_cls):
    raw_sv = np.array(shap_exp.shap_values(
        norm_inp.detach().clone(), check_additivity=False))
    # Handle your SHAP version's 5D output (1, 3, H, W, NUM_CLASSES)
    if raw_sv.ndim == 5:
        sv = raw_sv[0, :, :, :, pred_cls]          # (3, H, W)
    elif raw_sv.ndim == 4 and raw_sv.shape[-1] == NUM_CLASSES:
        sv = raw_sv[0, :, :, pred_cls]             # (H, W)
        sv = np.stack([sv, sv, sv], axis=0)
    elif raw_sv.ndim == 4:
        sv = raw_sv[0]                             # (3, H, W)
    else:
        sv = np.array(raw_sv[pred_cls])[0]
    sm = np.abs(sv).mean(axis=0)                   # (H, W)
    return (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)


# ─────────────────────────────────────────────
#  7.  Deletion / Insertion AUC
# ─────────────────────────────────────────────
def deletion_curve(raw_t, sal, pred_cls, steps=DELETION_STEPS):
    idx = np.argsort(sal.flatten())[::-1]
    tot = len(idx)
    cs  = []
    for s in range(steps + 1):
        k  = int(s / steps * tot)
        m  = raw_t.clone().cpu().numpy().reshape(3, -1)
        m[:, idx[:k]] = 0
        t  = normalise(torch.tensor(m.reshape(raw_t.shape),
                       dtype=torch.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cs.append(torch.softmax(model(t),1)[0,pred_cls].item())
    return np.linspace(0,1,steps+1), np.array(cs)

def insertion_curve(raw_t, sal, pred_cls, steps=DELETION_STEPS):
    blr = raw_t.cpu().numpy().copy()
    for c in range(3): blr[c] = gaussian_filter(blr[c], BLUR_SIGMA)
    idx = np.argsort(sal.flatten())[::-1]
    tot = len(idx); raw_np = raw_t.cpu().numpy()
    cs  = []
    for s in range(steps + 1):
        k  = int(s / steps * tot)
        rv = blr.reshape(3, -1).copy()
        rv[:, idx[:k]] = raw_np.reshape(3, -1)[:, idx[:k]]
        t  = normalise(torch.tensor(rv.reshape(raw_t.shape),
                       dtype=torch.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cs.append(torch.softmax(model(t),1)[0,pred_cls].item())
    return np.linspace(0,1,steps+1), np.array(cs)


# ─────────────────────────────────────────────
#  8.  Prototypes
# ─────────────────────────────────────────────
print("Extracting training features for prototype search...")
tr_feats, tr_labels = [], []
for imgs, lbls in tqdm(DataLoader(train_norm, batch_size=256,
                                  shuffle=False, num_workers=0),
                       ncols=80):
    with torch.no_grad():
        f = feature_extractor(imgs.to(DEVICE)).squeeze(-1).squeeze(-1)
    tr_feats.append(f.cpu().numpy())
    tr_labels.append(lbls.numpy())
tr_feats  = np.vstack(tr_feats)
tr_labels = np.concatenate(tr_labels)

nn_idx = NearestNeighbors(n_neighbors=N_PROTOTYPES,
                          metric="cosine", algorithm="brute")
nn_idx.fit(tr_feats)
print("Prototype index ready.\n")


# ─────────────────────────────────────────────
#  9.  Overlay helpers
# ─────────────────────────────────────────────
def gradcam_overlay(raw_np, cam, alpha=0.5):
    h, w = raw_np.shape[:2]
    cam_r = np.array(Image.fromarray(
        (cam*255).astype(np.uint8)).resize((w,h), Image.BILINEAR)) / 255.
    heat  = (plt.get_cmap("jet")(cam_r)[:,:,:3] * 255).astype(np.uint8)
    return (alpha*heat + (1-alpha)*raw_np).astype(np.uint8)

def shap_overlay(raw_np, sm):
    heat = (plt.get_cmap("hot")(sm)[:,:,:3])
    return ((0.5*heat + 0.5*raw_np/255.).clip(0,1)*255).astype(np.uint8)

def lime_overlay(raw_np, mask):
    out = raw_np.copy().astype(float)
    out[mask == 0] *= 0.35
    return out.astype(np.uint8)


# ─────────────────────────────────────────────
#  10. Main data-collection loop
# ─────────────────────────────────────────────
print("── Collecting all data for dashboard ─────────────────────────")
all_data = []

for idx in tqdm(sample_indices, desc="Processing", ncols=80):
    raw_t,  true_lbl = test_raw[idx]
    norm_t, _        = test_norm[idx]
    norm_inp         = norm_t.unsqueeze(0).to(DEVICE)
    raw_np           = to_display(raw_t)

    with torch.no_grad():
        probs     = torch.softmax(model(norm_inp),1)[0]
    pred_cls  = probs.argmax().item()
    conf      = probs[pred_cls].item()

    # Saliency maps
    gc   = gradcam(norm_inp, pred_cls)
    lime = get_lime(raw_np, pred_cls)
    sh   = get_shap(norm_inp, pred_cls)

    # Overlays
    gc_ov   = gradcam_overlay(raw_np, gc)
    lime_ov = lime_overlay(raw_np, lime > 0.3)
    sh_ov   = shap_overlay(raw_np, sh)

    # Metric curves
    x_del, gc_del   = deletion_curve(raw_t, gc,   pred_cls)
    _,     lime_del = deletion_curve(raw_t, lime,  pred_cls)
    _,     sh_del   = deletion_curve(raw_t, sh,    pred_cls)

    x_ins, gc_ins   = insertion_curve(raw_t, gc,   pred_cls)
    _,     lime_ins = insertion_curve(raw_t, lime,  pred_cls)
    _,     sh_ins   = insertion_curve(raw_t, sh,    pred_cls)

    # AUC scores
    del_aucs = {
        "GradCAM": auc(x_del, gc_del),
        "LIME"   : auc(x_del, lime_del),
        "SHAP"   : auc(x_del, sh_del),
    }
    ins_aucs = {
        "GradCAM": auc(x_ins, gc_ins),
        "LIME"   : auc(x_ins, lime_ins),
        "SHAP"   : auc(x_ins, sh_ins),
    }

    # Prototypes
    with torch.no_grad():
        qf = feature_extractor(norm_inp).squeeze().cpu().numpy()
    dists, nn_ids = nn_idx.kneighbors(qf.reshape(1,-1))
    protos = []
    for dist, ti in zip(dists[0], nn_ids[0]):
        pr, pl = train_raw[ti]
        protos.append({"img": to_display(pr), "label": pl,
                       "sim": 1 - dist,
                       "name": CIFAR10_CLASSES[pl]})

    all_data.append({
        "true_lbl" : true_lbl,  "pred_cls" : pred_cls,
        "true_name": CIFAR10_CLASSES[true_lbl],
        "pred_name": CIFAR10_CLASSES[pred_cls],
        "conf"     : conf,      "correct"  : pred_cls == true_lbl,
        "probs"    : probs.cpu().numpy(),
        "raw_np"   : raw_np,
        "gc_ov"    : gc_ov,     "lime_ov"  : lime_ov,  "sh_ov"  : sh_ov,
        "gc"       : gc,        "lime"     : lime,      "sh"     : sh,
        "x_del"    : x_del,
        "gc_del"   : gc_del,    "lime_del" : lime_del,  "sh_del" : sh_del,
        "x_ins"    : x_ins,
        "gc_ins"   : gc_ins,    "lime_ins" : lime_ins,  "sh_ins" : sh_ins,
        "del_aucs" : del_aucs,  "ins_aucs" : ins_aucs,
        "protos"   : protos,
    })

print("\nAll data collected.\n")


# ─────────────────────────────────────────────
#  11. PER-CLASS INDIVIDUAL DASHBOARD
#      Saved as one PNG per class — full detail
# ─────────────────────────────────────────────
print("── Saving per-class dashboards ───────────────────────────────")

for d in all_data:
    correct_sym = "✔" if d["correct"] else "✘"
    title_str   = (f"{d['true_name'].upper()}  |  "
                   f"Predicted: {d['pred_name']} {correct_sym}  "
                   f"({d['conf']*100:.1f}%)")

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title_str, fontsize=16, fontweight="bold",
                 color="white", y=0.98)

    # ── Grid layout ──────────────────────────────────────────────────
    # Row 0: Original | GradCAM | LIME | SHAP | Confidence
    # Row 1: Deletion curve | Insertion curve | Prototypes (span 3 cols)
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              hspace=0.35, top=0.93, bottom=0.04)

    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[0], wspace=0.08)
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.25)

    def dark_ax(ax):
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f3460")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── Row 0: saliency images ────────────────────────────────────────
    imgs_row = [
        ("Original",  d["raw_np"]),
        ("Grad-CAM",  d["gc_ov"]),
        ("LIME",      d["lime_ov"]),
        ("SHAP",      d["sh_ov"]),
    ]
    for col, (lbl, img) in enumerate(imgs_row):
        ax = fig.add_subplot(top_gs[col])
        ax.imshow(img)
        ax.set_title(lbl, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")

    # Confidence bar
    ax_conf = fig.add_subplot(top_gs[4])
    dark_ax(ax_conf)
    y_pos   = np.arange(NUM_CLASSES)
    bar_clr = [PALETTE[i] if i == d["pred_cls"] else "#334155"
               for i in range(NUM_CLASSES)]
    ax_conf.barh(y_pos, d["probs"], color=bar_clr, height=0.7)
    ax_conf.set_yticks(y_pos)
    ax_conf.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax_conf.set_xlim(0, 1); ax_conf.invert_yaxis()
    ax_conf.set_title("Confidence", fontsize=10, fontweight="bold")
    ax_conf.set_xlabel("Softmax score")

    # ── Row 1a: Deletion curve ─────────────────────────────────────────
    ax_del = fig.add_subplot(bot_gs[0])
    dark_ax(ax_del)
    ax_del.plot(d["x_del"], d["gc_del"],   color="#e63946", lw=2, label=f"GradCAM  {d['del_aucs']['GradCAM']:.3f}")
    ax_del.plot(d["x_del"], d["lime_del"], color="#457b9d", lw=2, label=f"LIME     {d['del_aucs']['LIME']:.3f}")
    ax_del.plot(d["x_del"], d["sh_del"],   color="#2a9d8f", lw=2, label=f"SHAP     {d['del_aucs']['SHAP']:.3f}")
    ax_del.set_title("Deletion AUC ↓  (lower = better)", fontsize=10)
    ax_del.set_xlabel("Fraction masked"); ax_del.set_ylabel("Confidence")
    ax_del.set_ylim(0, 1)
    leg = ax_del.legend(fontsize=8, title="Method  AUC")
    leg.get_title().set_color("white")
    for t in leg.get_texts(): t.set_color("white")

    # ── Row 1b: Insertion curve ────────────────────────────────────────
    ax_ins = fig.add_subplot(bot_gs[1])
    dark_ax(ax_ins)
    ax_ins.plot(d["x_ins"], d["gc_ins"],   color="#e63946", lw=2, label=f"GradCAM  {d['ins_aucs']['GradCAM']:.3f}")
    ax_ins.plot(d["x_ins"], d["lime_ins"], color="#457b9d", lw=2, label=f"LIME     {d['ins_aucs']['LIME']:.3f}")
    ax_ins.plot(d["x_ins"], d["sh_ins"],   color="#2a9d8f", lw=2, label=f"SHAP     {d['ins_aucs']['SHAP']:.3f}")
    ax_ins.set_title("Insertion AUC ↑  (higher = better)", fontsize=10)
    ax_ins.set_xlabel("Fraction revealed"); ax_ins.set_ylabel("Confidence")
    ax_ins.set_ylim(0, 1)
    leg2 = ax_ins.legend(fontsize=8, title="Method  AUC")
    leg2.get_title().set_color("white")
    for t in leg2.get_texts(): t.set_color("white")

    # ── Row 1c: Prototypes ─────────────────────────────────────────────
    proto_gs = gridspec.GridSpecFromSubplotSpec(
        1, N_PROTOTYPES, subplot_spec=bot_gs[2], wspace=0.06)

    for p_i, proto in enumerate(d["protos"]):
        ax_p = fig.add_subplot(proto_gs[p_i])
        ax_p.imshow(proto["img"])
        fc = "#2a9d8f" if proto["label"] == d["true_lbl"] else "#e63946"
        for sp in ax_p.spines.values():
            sp.set_edgecolor(fc); sp.set_linewidth(3)
        ax_p.set_title(f"{proto['name']}\n{proto['sim']:.2f}",
                       fontsize=7, color="white")
        ax_p.set_xticks([]); ax_p.set_yticks([])
        if p_i == 0:
            ax_p.set_ylabel("Prototypes", color="white",
                            fontsize=9, labelpad=6)

    plt.savefig(
        os.path.join(OUT_DIR, f"dashboard_{d['true_name']}.png"),
        dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    print(f"  Saved → dashboard_{d['true_name']}.png")


# ─────────────────────────────────────────────
#  12. MASTER SUMMARY DASHBOARD
#      All 10 classes — compact grid overview
#      Rows = classes | Cols = Original, GradCAM, LIME, SHAP, Del↓, Ins↑
# ─────────────────────────────────────────────
print("\n── Building Master Summary Dashboard ─────────────────────────")

n_rows    = NUM_CLASSES
col_imgs  = ["Original", "Grad-CAM", "LIME", "SHAP"]
n_img_col = len(col_imgs)

fig = plt.figure(figsize=(28, n_rows * 3.2))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Master XAI Dashboard — CIFAR-10 ResNet-18\n"
             "Saliency Methods  |  Deletion & Insertion AUC  |  Best Method",
             fontsize=15, fontweight="bold", color="white", y=1.00)

# 6 columns: 4 images + deletion bars + insertion bars
master_gs = gridspec.GridSpec(n_rows, 6, figure=fig,
                              hspace=0.08, wspace=0.06,
                              top=0.97, bottom=0.03)

for row, d in enumerate(all_data):
    correct_sym = "✔" if d["correct"] else "✘"

    # Image columns
    for col, img_key in enumerate(["raw_np","gc_ov","lime_ov","sh_ov"]):
        ax = fig.add_subplot(master_gs[row, col])
        ax.imshow(d[img_key])
        ax.axis("off")
        if row == 0:
            ax.set_title(col_imgs[col], color="white",
                         fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel(
                f"{d['true_name']}\n{correct_sym} {d['conf']*100:.0f}%",
                color="white", fontsize=7.5,
                rotation=0, labelpad=55, va="center"
            )

    # Deletion AUC bars (col 4)
    ax_d = fig.add_subplot(master_gs[row, 4])
    ax_d.set_facecolor("#16213e")
    methods = ["GradCAM","LIME","SHAP"]
    del_v   = [d["del_aucs"][m] for m in methods]
    ins_v   = [d["ins_aucs"][m] for m in methods]
    clrs    = ["#e63946","#457b9d","#2a9d8f"]
    bars    = ax_d.bar(methods, del_v, color=clrs, width=0.6)
    ax_d.set_ylim(0, 1)
    ax_d.set_yticks([0, 0.5, 1])
    ax_d.tick_params(axis="x", labelsize=6, colors="white")
    ax_d.tick_params(axis="y", labelsize=6, colors="white")
    for sp in ax_d.spines.values(): sp.set_edgecolor("#0f3460")
    if row == 0:
        ax_d.set_title("Del AUC↓", color="white", fontsize=8)
    # Highlight winner (lowest)
    win_i = int(np.argmin(del_v))
    bars[win_i].set_edgecolor("white"); bars[win_i].set_linewidth(2)

    # Insertion AUC bars (col 5)
    ax_i = fig.add_subplot(master_gs[row, 5])
    ax_i.set_facecolor("#16213e")
    bars2 = ax_i.bar(methods, ins_v, color=clrs, width=0.6)
    ax_i.set_ylim(0, 1)
    ax_i.set_yticks([0, 0.5, 1])
    ax_i.tick_params(axis="x", labelsize=6, colors="white")
    ax_i.tick_params(axis="y", labelsize=6, colors="white")
    for sp in ax_i.spines.values(): sp.set_edgecolor("#0f3460")
    if row == 0:
        ax_i.set_title("Ins AUC↑", color="white", fontsize=8)
    # Highlight winner (highest)
    win_j = int(np.argmax(ins_v))
    bars2[win_j].set_edgecolor("white"); bars2[win_j].set_linewidth(2)

master_path = os.path.join(OUT_DIR, "master_dashboard.png")
plt.savefig(master_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Master dashboard saved → {master_path}")


# ─────────────────────────────────────────────
#  13. Aggregate metric table — print + save
# ─────────────────────────────────────────────
print("\n── Final Aggregate Metric Table ──────────────────────────────")
print(f"\n{'Class':<12} │ {'Del GC':>7} {'Del LI':>7} {'Del SH':>7} "
      f"│ {'Ins GC':>7} {'Ins LI':>7} {'Ins SH':>7} "
      f"│ {'Best Del':>9} {'Best Ins':>9}")
print("─" * 90)

best_del_count = {"GradCAM":0,"LIME":0,"SHAP":0}
best_ins_count = {"GradCAM":0,"LIME":0,"SHAP":0}

for d in all_data:
    da  = d["del_aucs"]; ia = d["ins_aucs"]
    bd  = min(da, key=da.get)
    bi  = max(ia, key=ia.get)
    best_del_count[bd] += 1
    best_ins_count[bi] += 1
    print(f"  {d['true_name']:<10} │ "
          f"{da['GradCAM']:>7.3f} {da['LIME']:>7.3f} {da['SHAP']:>7.3f} │ "
          f"{ia['GradCAM']:>7.3f} {ia['LIME']:>7.3f} {ia['SHAP']:>7.3f} │ "
          f"{'★'+bd:>9} {'★'+bi:>9}")

print("─" * 90)
avg_del = {m: np.mean([d["del_aucs"][m] for d in all_data]) for m in ["GradCAM","LIME","SHAP"]}
avg_ins = {m: np.mean([d["ins_aucs"][m] for d in all_data]) for m in ["GradCAM","LIME","SHAP"]}
print(f"  {'AVERAGE':<10} │ "
      f"{avg_del['GradCAM']:>7.3f} {avg_del['LIME']:>7.3f} {avg_del['SHAP']:>7.3f} │ "
      f"{avg_ins['GradCAM']:>7.3f} {avg_ins['LIME']:>7.3f} {avg_ins['SHAP']:>7.3f}")

print(f"\n  🏆 Best Deletion  (wins): " +
      " | ".join(f"{m}: {v}" for m, v in best_del_count.items()))
print(f"  🏆 Best Insertion (wins): " +
      " | ".join(f"{m}: {v}" for m, v in best_ins_count.items()))


# ─────────────────────────────────────────────
#  14. Final summary
# ─────────────────────────────────────────────
print("\n" + "═" * 65)
print("  WEEK 4 COMPLETE — Full XAI Dashboard")
print("═" * 65)
print(f"\n  Files saved to → {OUT_DIR}")
print("    • dashboard_<class>.png   — detailed per-class dashboard")
print("    • master_dashboard.png    — compact overview of all 10 classes")
print("\n  Your project now covers:")
print("    ✅  Week 1 — Model training + early stopping")
print("    ✅  Week 2 — GradCAM, LIME, SHAP saliency maps")
print("    ✅  Week 3 — Deletion / Insertion / Sanity check metrics")
print("    ✅  Week 4 — Prototype XAI + full combined dashboard")
print("═" * 65)
