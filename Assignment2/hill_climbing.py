"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:

- compute_fitness
- mutate_seed
- select_best
- hill_climb

DO NOT change function signatures.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array

with open("data/imagenet_classes.txt") as f:
    class_labels = [line.strip() for line in f]

label_dictionary = {label: idx for idx, label in enumerate(class_labels)}

# ------------------------------------------------------------
# Global run stats (avoid changing required signatures)
# ------------------------------------------------------------
HC_MUTATION_NAMES = ["GRID", "HEX", "CROSSHATCH", "DIAGONAL"]

HC_LAST_RUN_STATS = {
    "last_improvement_mutation": "NONE",
    "per_mutation_best": {name: {"fitness": float("inf")} for name in HC_MUTATION_NAMES},
    "iterations_used": 0,

    # Track two different iteration counters (1-based in prints/JSON; 0 used internally for seed-best init)
    "best_fitness_iter": None,     # iteration where global best fitness was first seen
    "first_success_iter": None,    # iteration where success (top-1 != GT) first occurred
}


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Fitness (LOWER is better):
    - probability(ground-truth label).
    """
    i = label_dictionary.get(target_label)

    batch = np.expand_dims(image_array.copy(), axis=0)
    predictions = model.predict(batch, verbose=0)[0]

    if i is None:
        # fallback if GT label is missing from mapping
        return float(np.max(predictions))

    return float(predictions[i])


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ONE neighbor for EACH mutation type (no random).
    Order MUST match HC_MUTATION_NAMES.
    """
    seed = seed.astype(np.float32)
    epsilon_255 = 255 * epsilon

    mutants = [
        mutate_grid(seed, epsilon_255),
        mutate_hexagonal_grid(seed, epsilon_255),
        mutate_crosshatch(seed, epsilon_255),
        mutate_diagonal_stripes(seed, epsilon_255),
    ]
    return mutants


def mutate_hexagonal_grid(seed, epsilon, spacing=15, thickness=1.0):
    """hexagonal/honeycomb pattern"""
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape

    perturbation = np.random.uniform(-epsilon, epsilon)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    line1 = x_coords % spacing < thickness
    coords_60 = x_coords * np.cos(np.pi / 3) + y_coords * np.sin(np.pi / 3)
    line2 = coords_60 % spacing < thickness
    coords_120 = x_coords * np.cos(2 * np.pi / 3) + y_coords * np.sin(2 * np.pi / 3)
    line3 = coords_120 % spacing < thickness

    hex_mask = line1 | line2 | line3
    mutant[hex_mask] += perturbation
    return mutant.astype(np.uint8)


def mutate_crosshatch(seed, epsilon, spacing=8, thickness=1.0):
    """Diagonal crosshatch pattern"""
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape

    perturbation = np.random.uniform(-epsilon, epsilon)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    diagonal1 = (x_coords + y_coords) % spacing < thickness
    diagonal2 = (x_coords - y_coords) % spacing < thickness

    crosshatch_mask = diagonal1 | diagonal2
    mutant[crosshatch_mask] += perturbation
    return mutant.astype(np.uint8)


def mutate_diagonal_stripes(seed, epsilon, spacing=8, thickness=1):
    """diagonal stripe pattern"""
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape

    perturbation = np.random.uniform(-epsilon, epsilon)
    angle_choice = np.random.choice([45, -45, 30, -30, 60, -60])
    angle_rad = np.radians(angle_choice)

    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    diagonal = x_coords * np.cos(angle_rad) + y_coords * np.sin(angle_rad)

    stripe_position = diagonal % spacing
    stripe_mask = stripe_position < thickness
    mutant[stripe_mask] += perturbation
    return mutant.astype(np.uint8)


def mutate_grid(seed, epsilon, s=8):
    """grid pattern"""
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape

    x = np.random.uniform(-epsilon, epsilon)
    for i in range(0, h, s):
        mutant[i, :, :] += x
    for j in range(0, w, s):
        mutant[:, j, :] += x

    return mutant.astype(np.uint8)


# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

def select_best(
    mutants: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Also updates global per-mutation tracking for the first 4 mutants.
    """
    best_image = None
    best_fitness = float('inf')

    n_ops = len(HC_MUTATION_NAMES)

    for idx, m in enumerate(mutants):
        fitness = compute_fitness(m, model, target_label)

        # Track best per mutation type for operator-generated mutants only
        if idx < n_ops:
            mut_name = HC_MUTATION_NAMES[idx]
            if fitness < HC_LAST_RUN_STATS["per_mutation_best"][mut_name]["fitness"]:
                HC_LAST_RUN_STATS["per_mutation_best"][mut_name] = {
                    "fitness": float(fitness),
                    "image": m,
                }

        if fitness < best_fitness:
            best_fitness = fitness
            best_image = m

    return best_image, best_fitness


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """Main hill-climbing loop."""
    # Reset run stats
    HC_LAST_RUN_STATS["last_improvement_mutation"] = "NONE"
    HC_LAST_RUN_STATS["per_mutation_best"] = {name: {"fitness": float("inf")} for name in HC_MUTATION_NAMES}
    HC_LAST_RUN_STATS["iterations_used"] = 0
    HC_LAST_RUN_STATS["best_fitness_iter"] = None
    HC_LAST_RUN_STATS["first_success_iter"] = None

    current_image = initial_seed.astype(np.uint8)
    current_fitness = compute_fitness(current_image, model, target_label)

    # Track global-best fitness across ALL evaluated candidates during the run
    global_best_fitness = current_fitness
    HC_LAST_RUN_STATS["best_fitness_iter"] = 0  # seed is best before any iteration

    epsilon_255 = 255 * epsilon
    true_idx = label_dictionary.get(target_label)

    for it in range(iterations):
        iter_no = it + 1
        HC_LAST_RUN_STATS["iterations_used"] = iter_no

        mutants = mutate_seed(current_image, epsilon)
        mutants = [clip_helper(initial_seed, m, epsilon_255) for m in mutants]
        mutants.append(current_image)  # elitism

        best_image, best_fitness = select_best(mutants, model, target_label)

        # Update: iteration where global best fitness was first seen
        if best_fitness < global_best_fitness:
            global_best_fitness = best_fitness
            HC_LAST_RUN_STATS["best_fitness_iter"] = iter_no

        # Accept improvement (classic HC step)
        if best_fitness < current_fitness:
            # Identify which candidate index was selected (0..3 => mutation name)
            chosen_idx = None
            for idx, m in enumerate(mutants):
                if m is best_image or np.array_equal(m, best_image):
                    chosen_idx = idx
                    break

            if chosen_idx is not None and chosen_idx < len(HC_MUTATION_NAMES):
                HC_LAST_RUN_STATS["last_improvement_mutation"] = HC_MUTATION_NAMES[chosen_idx]
            else:
                HC_LAST_RUN_STATS["last_improvement_mutation"] = "NONE"

            current_image = best_image
            current_fitness = best_fitness

        # Stop if fooled (top-1 != GT index); record first_success_iter once
        if true_idx is not None:
            preds = model.predict(np.expand_dims(current_image, axis=0), verbose=0)[0]
            if int(np.argmax(preds)) != int(true_idx):
                if HC_LAST_RUN_STATS["first_success_iter"] is None:
                    HC_LAST_RUN_STATS["first_success_iter"] = iter_no
                break

    return current_image, current_fitness


def clip_helper(original, perturbed, epsilon):
    x = np.clip(perturbed - original, -epsilon, epsilon)
    return np.clip(original + x, 0, 255)


# ============================================================
# Helpers for printing/metrics
# ============================================================

def _top1_idx_and_prob(pred_vec: np.ndarray) -> Tuple[int, float]:
    idx = int(np.argmax(pred_vec))
    prob = float(pred_vec[idx])
    return idx, prob


def _metrics_vs_seed(seed: np.ndarray, img: np.ndarray) -> Tuple[float, int, float]:
    h, w = seed.shape[0], seed.shape[1]
    total_pixels = int(h * w)

    pixels_changed = int(np.any(img != seed, axis=2).sum())  # per (x,y)
    pct_changed = float(100.0 * pixels_changed / total_pixels)

    l_inf = float(np.max(np.abs(img.astype(np.float32) - seed.astype(np.float32))) / 255.0)
    return l_inf, pixels_changed, pct_changed


def _print_block_header(image_name: str, gt_label: str, gt_idx: int, epsilon: float, iterations: int):
    print("\n" + "=" * 72)
    print(f"IMAGE: {image_name}")
    print(f"GT:    {gt_label} (idx={gt_idx}) | eps={epsilon} | iters={iterations}")
    print("-" * 72)


def _print_method_line(method: str, pred_name: str, top1_idx: int, top1_prob: float,
                       success: bool, linf: float | None,
                       pixels_changed: int | None, pix_pct: float | None,
                       tsec: float | None, extra: str = ""):
    linf_str = f"{linf:.6f}" if linf is not None else "-"
    pix_str = f"{pixels_changed}" if pixels_changed is not None else "-"
    pixpct_str = f"{pix_pct:6.2f}%" if pix_pct is not None else "   -  "
    t_str = f"{tsec:.4f}" if tsec is not None else "-"
    extra_str = f" {extra}" if extra else ""

    print(f"{method:<9}| pred={pred_name:<20} idx={top1_idx:>4} p={top1_prob:>7.4f} "
          f"| success={str(success):<5} | linf={linf_str:<8} | pix={pix_str:<7} ({pixpct_str}) | t={t_str}s{extra_str}")


# ============================================================
# 5. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    model = vgg16.VGG16(weights="imagenet")

    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    epsilon = 0.30
    iterations = 300

    for item in image_list:
        image_name = item["image"]
        target_label = item["label"]

        true_idx = label_dictionary.get(target_label)
        if true_idx is None:
            print(f"\nSkipping {image_name}: ground-truth label '{target_label}' not found in imagenet_classes.txt")
            continue

        results_dir = os.path.join("hc_results", image_name.replace(".", "_"))
        os.makedirs(results_dir, exist_ok=True)

        img = load_img(os.path.join("images", image_name), target_size=(224, 224))
        seed = img_to_array(img).astype(np.uint8)

        # CLEAN
        preds_clean = model.predict(np.expand_dims(seed, axis=0), verbose=0)[0]
        clean_idx, clean_prob = _top1_idx_and_prob(preds_clean)
        clean_success = (clean_idx != true_idx)
        top_clean = decode_predictions(np.expand_dims(preds_clean, axis=0), top=5)[0]
        clean_name = top_clean[0][1]

        _print_block_header(image_name, target_label, true_idx, epsilon, iterations)
        _print_method_line("CLEAN", clean_name, clean_idx, clean_prob, clean_success, None, None, None, None)

        # HC (timed)
        t0 = time.perf_counter()
        final_img, final_fitness = hill_climb(seed, model, target_label, epsilon, iterations)
        hc_time_sec = float(time.perf_counter() - t0)

        preds_adv = model.predict(np.expand_dims(final_img, axis=0), verbose=0)[0]
        adv_idx, adv_prob = _top1_idx_and_prob(preds_adv)
        top_adv = decode_predictions(np.expand_dims(preds_adv, axis=0), top=5)[0]
        adv_name = top_adv[0][1]

        l_inf, pixels_changed, pct_changed = _metrics_vs_seed(seed, final_img)
        success = (adv_idx != true_idx)

        mut_used = HC_LAST_RUN_STATS["last_improvement_mutation"]
        best_iter = HC_LAST_RUN_STATS["best_fitness_iter"]
        succ_iter = HC_LAST_RUN_STATS["first_success_iter"]

        extra = f"({mut_used}) best_fit_iter={best_iter} first_success_iter={succ_iter}"
        _print_method_line("HC", adv_name, adv_idx, adv_prob, success,
                           l_inf, pixels_changed, pct_changed, hc_time_sec, extra=extra)

        # Per-mutation best section
        print("-" * 72)
        print("BEST PER MUTATION (single-step candidate across all iterations)")
        for mut_name in HC_MUTATION_NAMES:
            entry = HC_LAST_RUN_STATS["per_mutation_best"].get(mut_name, {})
            best_img = entry.get("image", None)
            best_fit = entry.get("fitness", float("inf"))

            if best_img is None:
                print(f"{mut_name:<9}| no data")
                continue

            preds_m = model.predict(np.expand_dims(best_img, axis=0), verbose=0)[0]
            m_idx, m_prob = _top1_idx_and_prob(preds_m)
            top_m = decode_predictions(np.expand_dims(preds_m, axis=0), top=1)[0]
            m_name = top_m[0][1]

            m_linf, m_pix, m_pct = _metrics_vs_seed(seed, best_img)
            m_success = (m_idx != true_idx)

            _print_method_line(mut_name, m_name, m_idx, m_prob, m_success,
                               m_linf, m_pix, m_pct, None,
                               extra=f"[fitness={best_fit:.6f}]")

        # ---- Comparison images (original/adversarial + diff amplified) ----
        diff = np.abs(final_img.astype(np.float32) - seed.astype(np.float32))
        diff_amplified = np.clip(diff * 10.0, 0, 255).astype(np.uint8)

        clean_path = os.path.join(results_dir, f"clean_{image_name}.png")
        adv_path = os.path.join(results_dir, f"adv_{image_name}.png")
        diff_path = os.path.join(results_dir, f"diffx10_{image_name}.png")

        array_to_img(seed).save(clean_path)
        array_to_img(final_img).save(adv_path)
        array_to_img(diff_amplified).save(diff_path)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(array_to_img(seed))
        axes[0].set_title("Original", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(array_to_img(final_img))
        axes[1].set_title("Adversarial (HC)", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(array_to_img(diff_amplified))
        axes[2].set_title("Diff x10 (clipped)", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"comparison_{image_name}"), dpi=150)
        plt.close(fig)

        # Save metadata JSON
        image_name_clean = image_name.split(".")[0]
        metadata_path = os.path.join(results_dir, f"attack_metadata_{image_name_clean}.json")

        attack_info = {
            "image_name": image_name,
            "target_label": target_label,
            "target_label_index": int(true_idx),
            "epsilon": float(epsilon),
            "iterations": int(iterations),
            "iterations_used": int(HC_LAST_RUN_STATS["iterations_used"]),
            "best_fitness_iter": int(HC_LAST_RUN_STATS["best_fitness_iter"]),
            "first_success_iter": (None if HC_LAST_RUN_STATS["first_success_iter"] is None
                                  else int(HC_LAST_RUN_STATS["first_success_iter"])),
            "final_fitness": float(final_fitness),
            "time_sec": float(hc_time_sec),
            "l_inf_distance": float(l_inf),
            "pixels_changed": int(pixels_changed),
            "pct_pixels_changed": float(pct_changed),
            "clean_top1_index": int(clean_idx),
            "adv_top1_index": int(adv_idx),
            "success": bool(success),
            "last_improvement_mutation": str(mut_used),
            "best_per_mutation_fitness": {k: float(HC_LAST_RUN_STATS["per_mutation_best"][k]["fitness"]) for k in HC_MUTATION_NAMES},
            "baseline_predictions_top5": [{"label": name, "prob": float(prob)} for _, name, prob in top_clean],
            "adversarial_predictions_top5": [{"label": name, "prob": float(prob)} for _, name, prob in top_adv],
        }

        with open(metadata_path, "w") as f:
            json.dump(attack_info, f, indent=4)
