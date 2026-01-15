"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

import random
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array

with open("data/imagenet_classes.txt") as f:
    class_labels = [line.strip() for line in f]
label_dictionary = {label: idx for idx, label in enumerate(class_labels)}

# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """
    image = image_array.copy()
    batch = np.expand_dims(image, axis=0)
    predictions = model.predict(batch, verbose=0)[0]
    top_prediction = np.argmax(predictions)
    i = label_dictionary.get(target_label)

    if top_prediction == i:
        return predictions[i]
    else:
        return -predictions[top_prediction]

# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.
    """
    seed = seed.astype(np.float32)
    epsilon_255 = 255 * epsilon
    n = 5
    mutants = []

    for i in range(n):
        strategy = random.choice([
            'mutate_grid',
            'mutate_hexagonal_grid',
            'mutate_crosshatch',
            'mutate_diagonal_stripes'
        ])

        if strategy == 'mutate_diagonal_stripes':
            mutant = mutate_diagonal_stripes(seed, epsilon_255)
        elif strategy == 'mutate_grid':
            mutant = mutate_grid(seed, epsilon_255)
        elif strategy == 'mutate_hexagonal_grid':
            mutant = mutate_hexagonal_grid(seed, epsilon_255)
        elif strategy == 'mutate_crosshatch':
            mutant = mutate_crosshatch(seed, epsilon_255)

        mutants.append(mutant)

    return mutants

def mutate_hexagonal_grid(seed, epsilon, spacing=15, thickness=1.0):
    """hexagonal/honeycomb pattern"""
    print("Mutate: hexagonal grid")
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape
    
    perturbation = np.random.uniform(-epsilon, epsilon)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    line1 = x_coords % spacing < thickness
    coords_60 = x_coords * np.cos(np.pi/3) + y_coords * np.sin(np.pi/3)
    line2 = coords_60 % spacing < thickness
    coords_120 = x_coords * np.cos(2*np.pi/3) + y_coords * np.sin(2*np.pi/3)
    line3 = coords_120 % spacing < thickness
    hex_mask = line1 | line2 | line3
    
    mutant[hex_mask] += perturbation
    return mutant.astype(np.uint8)

def mutate_crosshatch(seed, epsilon, spacing=8, thickness=1.0):
    """Diagonal crosshatch pattern"""
    print("Mutate: crosshatch pattern")
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
    print("Mutate: diagonal stripes")
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

def mutate_grid(seed, epsilon, s = 8):
    """grid pattern"""
    print("Mutate: grid pattern")
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
    """
    best_image = None
    best_fitness = float('inf')

    for m in mutants:
        fitness = compute_fitness(m, model, target_label)
        if fitness < best_fitness:
            best_fitness = fitness
            best_image = m

    print(f"Best fitness: {best_fitness:.4f}")
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
    current_image = initial_seed.astype(np.uint8)
    current_fitness = compute_fitness(current_image, model, target_label)
    epsilon_255 = 255 * epsilon

    for it in range(iterations):
        mutants = mutate_seed(current_image, epsilon)
        mutants = [clip_helper(initial_seed, m, epsilon_255) for m in mutants]
        mutants.append(current_image)

        best_image, best_fitness = select_best(mutants, model, target_label)
        if best_fitness < current_fitness:
            current_image = best_image
            current_fitness = best_fitness

        # Check if fooled (use preprocessing for prediction)
        batch = np.expand_dims(current_image.copy(), axis=0)
        predictions = model.predict(batch, verbose=0)[0]
        top_prediction = np.argmax(predictions)
        if top_prediction != label_dictionary[target_label]:
            print(f"Finished at {it}")
            break

    return current_image, current_fitness

def clip_helper(original, perturbed, epsilon):
    x = np.clip(perturbed - original, -epsilon, epsilon)
    return np.clip(original + x, 0, 255)

# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================

if __name__ == "__main__":
    model = vgg16.VGG16(weights="imagenet")

    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick target image
    item = image_list[0]
    image_name = item["image"]
    target_label = item["label"]
    epsilon = 0.30

    # create results directory
    results_dir = os.path.join("hc_results", image_name.replace(".", "_"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    img = load_img(os.path.join("images", image_name), target_size=(224, 224))
    seed = img_to_array(img).astype(np.uint8)

    print(f"Loaded: {image_name} | Target: {target_label}")
    
    # get baseline metadeta
    preds_clean = model.predict(np.expand_dims(seed, axis=0), verbose=0)
    top_clean = decode_predictions(preds_clean, top=5)[0]

    # Run Attack
    final_img, final_fitness = hill_climb(seed, model, target_label, epsilon, 300)
    
    # get adversarial metadata
    f_batch = np.expand_dims(final_img.astype(np.float32), axis=0)
    final_preds = model.predict(f_batch, verbose=0)
    top_adv = decode_predictions(final_preds, top=5)[0]
    
    # Calculate L-infinity distance
    l_inf = float(np.max(np.abs(final_img.astype(float) - seed.astype(float))) / 255.0)

    # 1. Visualization
    diff = np.abs(final_img.astype(np.float32) - seed.astype(np.float32))
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(array_to_img(seed))
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(array_to_img(final_img))
    axes[1].set_title("Adversarial", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"result_{image_name}"), dpi=150)
    plt.show()

    # save metadata
    attack_info = {
        "image_name": image_name,
        "target_label": target_label,
        "epsilon": epsilon,
        "final_fitness": float(final_fitness),
        "l_inf_distance": l_inf,
        "baseline_predictions": [
            {"label": name, "prob": float(prob)} for _, name, prob in top_clean
        ],
        "adversarial_predictions": [
            {"label": name, "prob": float(prob)} for _, name, prob in top_adv
        ],
        "success": top_clean[0][1] != top_adv[0][1]
    }

    image_name_clean = image_name.split(".")[0]
    metadata_path = os.path.join(results_dir, f"attack_metadata_{image_name_clean}.json")
    with open(metadata_path, "w") as f:
        json.dump(attack_info, f, indent=4)

    print(f"\nFinal Predictions:")
    for cl in top_adv:
        print(cl)