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
# Hide TensorFlow C++ logs and progress bars
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

# Load ImageNet classes for label mapping
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
    """
    image = image_array.copy()
    batch = np.expand_dims(image, axis=0)
    # Important: VGG16 expects preprocessed input for correct probability
    batch = preprocess_input(batch)
    predictions = model.predict(batch, verbose=0)[0]
    top_prediction = np.argmax(predictions)
    i = label_dictionary.get(target_label)

    if top_prediction == i:
        return float(predictions[i])
    else:
        return float(-predictions[top_prediction])

# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce mutated neighbors satisfying the L∞ constraint.
    """
    seed_f = seed.astype(np.float32)
    epsilon_255 = 255 * epsilon
    n = 5
    mutants = []

    for i in range(n):
        strategy = random.choice([
            'mutate_grid',
            'mutate_hexagonal_grid',
            'mutate_crosshatch'
        ])

        if strategy == 'mutate_grid':
            mutant = mutate_grid(seed_f, epsilon_255)
        elif strategy == 'mutate_hexagonal_grid':
            mutant = mutate_hexagonal_grid(seed_f, epsilon_255)
        elif strategy == 'mutate_crosshatch':
            mutant = mutate_crosshatch(seed_f, epsilon_255)

        mutants.append(mutant)

    return mutants

def mutate_hexagonal_grid(seed, epsilon, spacing=15, thickness=1.0):
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
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape
    perturbation = np.random.uniform(-epsilon, epsilon)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    diagonal1 = (x_coords + y_coords) % spacing < thickness
    diagonal2 = (x_coords - y_coords) % spacing < thickness
    crosshatch_mask = diagonal1 | diagonal2
    
    mutant[crosshatch_mask] += perturbation
    return mutant.astype(np.uint8)

def mutate_grid(seed, epsilon, s = 8):
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
    Evaluate fitness for all candidates and return the one with the lowest score.
    """
    best_image = mutants[0]
    best_fitness = compute_fitness(best_image, model, target_label)

    for m in mutants[1:]:
        fitness = compute_fitness(m, model, target_label)
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
    """
    Main hill-climbing loop.
    """
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

        # Check for success without progress bars
        batch = preprocess_input(np.expand_dims(current_image, axis=0))
        predictions = model.predict(batch, verbose=0)[0]
        if np.argmax(predictions) != label_dictionary[target_label]:
            print(f"Finished at {it}")
            break

    return current_image, current_fitness

def clip_helper(original, perturbed, epsilon):
    """Ensures L-infinity bound is maintained relative to original."""
    diff = perturbed.astype(np.float32) - original.astype(np.float32)
    return np.clip(original + np.clip(diff, -epsilon, epsilon), 0, 255).astype(np.uint8)

# ============================================================
# 5. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Redirect all console print statements to attack.txt for a clean log
    orig_stdout = sys.stdout
    f_log = open('attack.txt', 'w')
    sys.stdout = f_log

    try:
        model = vgg16.VGG16(weights="imagenet")
        with open("data/image_labels.json", "r") as f:
            image_list = json.load(f)
        
        # Select target image (Example Index 2)
        IMAGE_INDEX = 2 
        item = image_list[IMAGE_INDEX]
        image_name = item["image"]
        target_label = item["label"]
        epsilon = 0.30
        
        # Setup results folder
        results_dir = os.path.join("hc_results", image_name.replace(".", "_"))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        print(f"Image: {image_name}")
        print(f"Target label: {target_label}")

        img = load_img(os.path.join("images", image_name), target_size=(224, 224))
        seed = img_to_array(img)
        
        print("\nBaseline predictions (top-5):")
        preds = model.predict(preprocess_input(np.expand_dims(seed, axis=0)), verbose=0)
        for i, (_, name, prob) in enumerate(decode_predictions(preds, top=5)[0], 1):
            print(f"{i}. {name:<20} prob={prob:.5f}")

        # Run Hill Climbing Attack
        final_img, final_fitness = hill_climb(seed, model, target_label, epsilon)

        # Calculate Visible Perturbation Map (Amplified 10x)
        diff = np.abs(final_img.astype(np.float32) - seed.astype(np.float32))
        diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

        # Output Results to File
        print(f"\nFinal fitness: {final_fitness:.4f}")
        l_inf = np.max(np.abs(final_img.astype(float) - seed.astype(float))) / 255.0
        print(f"L∞ distance: {l_inf:.4f}")

        print("\nFinal predictions (top-5):")
        final_preds = model.predict(preprocess_input(np.expand_dims(final_img, axis=0)), verbose=0)
        for cl in decode_predictions(final_preds, top=5)[0]:
            print(cl)

        # Final Visualization with Correct RGB Colors
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(array_to_img(seed.astype(np.uint8)))
        axes[0].set_title("Original", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(array_to_img(final_img.astype(np.uint8)))
        axes[1].set_title("Adversarial", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(array_to_img(diff_amplified))
        axes[2].set_title("Perturbation (10x)", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "attack_comparison.png"), dpi=150)

    finally:
        # Restore terminal output
        sys.stdout = orig_stdout
        f_log.close()
        print(f"Attack complete. Log saved in attack.txt, images in {results_dir}")