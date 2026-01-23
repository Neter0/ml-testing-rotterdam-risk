import json
import os
import time
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.models import vgg16
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


# -----------------------------
# Utility: parse Imagenet prediction
# -----------------------------
def parse_prediction(output, categories):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    return categories[top_catid], top_prob.item()


def top1_index(output: torch.Tensor) -> int:
    return int(torch.argmax(output[0]).item())


def _print_block_header(image_name: str, gt_label: str, gt_idx: int | None, eps: float, pgd_steps: int, pgd_step: float):
    print("\n" + "=" * 72)
    print(f"IMAGE: {image_name}")
    print(f"GT:    {gt_label} (idx={gt_idx}) | eps={eps} | pgd_steps={pgd_steps} | pgd_step={pgd_step}")
    print("-" * 72)


def _print_method_line(method: str, pred_name: str, top1_idx: int, top1_prob: float,
                       success, linf: float | None,
                       pixels_changed: int | None, pix_pct: float | None,
                       tsec: float | None):
    linf_str = f"{linf:.6f}" if linf is not None else "-"
    pix_str = f"{pixels_changed}" if pixels_changed is not None else "-"
    pixpct_str = f"{pix_pct:6.2f}%" if pix_pct is not None else "   -  "
    t_str = f"{tsec:.4f}" if tsec is not None else "-"
    print(f"{method:<9}| pred={pred_name:<20} idx={top1_idx:>4} p={top1_prob:>7.4f} "
          f"| success={str(success):<5} | linf={linf_str:<8} | pix={pix_str:<7} ({pixpct_str}) | t={t_str}s")


def _pixels_changed_and_pct(x_adv: torch.Tensor, x: torch.Tensor) -> tuple[int, float]:
    # x shapes: [1, 3, H, W]
    h = int(x.shape[2])
    w = int(x.shape[3])
    total_pixels = h * w

    changed = ((x_adv != x).any(dim=1)).sum().item()  # count H*W where any channel differs
    pct = 100.0 * float(changed) / float(total_pixels)
    return int(changed), float(pct)


# ================================================================
# 1. Load JSON file with images + expected human label
# ================================================================
JSON_FILE = "data/image_labels.json"
IMAGE_DIR = "images/"

with open(JSON_FILE, "r") as f:
    items = json.load(f)

# ================================================================
# 2. Load ImageNet labels
# ================================================================
with open("data/imagenet_classes.txt", "r") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]

label_to_index = {label: i for i, label in enumerate(imagenet_labels)}

# ================================================================
# 3. Model
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
net = vgg16(weights="DEFAULT").to(device)
net.eval()

# ================================================================
# 4. Image preprocessing transform
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # [0,1]
])

# ================================================================
# 5. Attack hyperparameters
# ================================================================
EPS = 0.30
PGD_STEPS = 40
PGD_STEP_SIZE = 0.01

# ================================================================
# 6. Output directory
# ================================================================
OUTDIR = "attack_results"
os.makedirs(OUTDIR, exist_ok=True)

# ================================================================
# 7. Run attacks
# ================================================================
for entry in tqdm(items, desc="Running attacks"):
    image_file = entry["image"]
    human_label = entry["label"]

    img_path = os.path.join(IMAGE_DIR, image_file)
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    true_idx = label_to_index.get(human_label, None)

    # CLEAN
    out_clean = net(x)
    pred_clean, prob_clean = parse_prediction(out_clean, imagenet_labels)
    clean_top1_idx = top1_index(out_clean)
    clean_success = (clean_top1_idx != true_idx) if true_idx is not None else None

    # FGM (timed)
    t0 = time.perf_counter()
    x_fgm = fast_gradient_method(net, x, EPS, np.inf)
    fgm_time_sec = float(time.perf_counter() - t0)

    out_fgm = net(x_fgm)
    pred_fgm, prob_fgm = parse_prediction(out_fgm, imagenet_labels)
    fgm_top1_idx = top1_index(out_fgm)
    fgm_success = (fgm_top1_idx != true_idx) if true_idx is not None else None

    fgm_linf = float((x_fgm - x).abs().max().item())
    fgm_pix, fgm_pix_pct = _pixels_changed_and_pct(x_fgm, x)

    # PGD (timed)
    t0 = time.perf_counter()
    x_pgd = projected_gradient_descent(net, x, EPS, PGD_STEP_SIZE, PGD_STEPS, np.inf)
    pgd_time_sec = float(time.perf_counter() - t0)

    out_pgd = net(x_pgd)
    pred_pgd, prob_pgd = parse_prediction(out_pgd, imagenet_labels)
    pgd_top1_idx = top1_index(out_pgd)
    pgd_success = (pgd_top1_idx != true_idx) if true_idx is not None else None

    pgd_linf = float((x_pgd - x).abs().max().item())
    pgd_pix, pgd_pix_pct = _pixels_changed_and_pct(x_pgd, x)

    # Save images
    save_image(x, os.path.join(OUTDIR, f"{image_file}_clean.png"))
    save_image(x_fgm, os.path.join(OUTDIR, f"{image_file}_fgm.png"))
    save_image(x_pgd, os.path.join(OUTDIR, f"{image_file}_pgd.png"))

    # Print in same style as HC
    _print_block_header(image_file, human_label, true_idx, EPS, PGD_STEPS, PGD_STEP_SIZE)
    _print_method_line("CLEAN", pred_clean, clean_top1_idx, prob_clean, clean_success, None, None, None, None)
    _print_method_line("FGM", pred_fgm, fgm_top1_idx, prob_fgm, fgm_success, fgm_linf, fgm_pix, fgm_pix_pct, fgm_time_sec)
    _print_method_line("PGD", pred_pgd, pgd_top1_idx, prob_pgd, pgd_success, pgd_linf, pgd_pix, pgd_pix_pct, pgd_time_sec)

    # Save per-image JSON for offline comparison
    metrics = {
        "image_name": image_file,
        "target_label": human_label,
        "target_label_index": int(true_idx) if true_idx is not None else None,
        "epsilon": float(EPS),
        "pgd_steps": int(PGD_STEPS),
        "pgd_step_size": float(PGD_STEP_SIZE),
        "clean": {
            "pred_label": pred_clean,
            "pred_prob": float(prob_clean),
            "top1_index": int(clean_top1_idx),
            "success": clean_success,
        },
        "fgm": {
            "pred_label": pred_fgm,
            "pred_prob": float(prob_fgm),
            "top1_index": int(fgm_top1_idx),
            "success": fgm_success,
            "l_inf_distance": float(fgm_linf),
            "pixels_changed": int(fgm_pix),
            "pct_pixels_changed": float(fgm_pix_pct),
            "time_sec": float(fgm_time_sec),
        },
        "pgd": {
            "pred_label": pred_pgd,
            "pred_prob": float(prob_pgd),
            "top1_index": int(pgd_top1_idx),
            "success": pgd_success,
            "l_inf_distance": float(pgd_linf),
            "pixels_changed": int(pgd_pix),
            "pct_pixels_changed": float(pgd_pix_pct),
            "time_sec": float(pgd_time_sec),
        },
    }

    json_path = os.path.join(OUTDIR, f"{os.path.splitext(image_file)[0]}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
