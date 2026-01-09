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
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

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
    image = preprocess_input(image_array.copy())
    batch = np.expand_dims(image, axis=0)
    predictions = model.predict(batch)[0]
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

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbors must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """
    seed = seed.astype(np.float32)
    epsilon_255 = 255 * epsilon
    n = 5
    mutants = []

    for i in range(n):
        strategy = random.choice([
            'mutate_pixels',
            'mutate_square',
            'mutate_channel',
            'mutate_grid'
        ])

        if strategy == 'mutate_pixels':
            mutant = mutate_pixels(seed, epsilon_255, num_pixels=10)
        elif strategy == 'mutate_square':
            mutant = mutate_square(seed, epsilon_255, size=8)
        elif strategy == 'mutate_channel':
            mutant = mutate_channel(seed, epsilon_255)
        elif strategy == 'mutate_grid':
            mutant = mutate_grid(seed, epsilon_255)

        mutants.append(mutant)

    return mutants

# Per-pixel mutation, multiple random pixels
def mutate_pixels(seed, epsilon, num_pixels = 10):
    mutant = seed.copy().astype(np.float32)
    h, w, c = seed.shape

    for _ in range(num_pixels):
        channel = np.random.randint(0, c)
        i = np.random.randint(0, h)
        j = np.random.randint(0, w)
        x = np.random.uniform(-epsilon, epsilon)
        mutant[i, j, channel] += x

    return mutant.astype(np.uint8)

# Square region mutation
def mutate_square(seed, epsilon, size = 8):
    mutant = seed.copy().astype(np.float32)
    h, w, _ = seed.shape
    x = np.random.randint(0, w - size)
    y = np.random.randint(0, h - size)

    noise = np.random.uniform(-epsilon, epsilon, size=(size, size, 3))
    mutant[y:y + size, x:x + size] += noise
    return mutant.astype(np.uint8)

# Single channel mutation, whole image
def mutate_channel(seed, epsilon):
    mutant = seed.copy().astype(np.float32)
    channel = np.random.randint(0, 3)
    x = np.random.uniform(-epsilon, epsilon)
    mutant[..., channel] += x
    return mutant.astype(np.uint8)

# Inject a faint grid into the image
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
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        mutants (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """

    best_image = None
    best_fitness = float('inf')

    for m in mutants:
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

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """

    current_image = initial_seed.astype(np.uint8)
    current_fitness = compute_fitness(current_image, model, target_label)
    epsilon_255 = 255 * epsilon

    for it in range(iterations):
        mutants = mutate_seed(current_image, epsilon)
        mutants = [clip_helper(initial_seed, m, epsilon_255) for m in mutants]
        mutants.append(current_image)  # Keep the current image in candidates

        best_image, best_fitness = select_best(mutants, model, target_label)
        if best_fitness < current_fitness:
            current_image = best_image
            current_fitness = best_fitness

        predictions = model.predict(np.expand_dims(current_image, axis=0))[0]
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
    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick first entry
    item = image_list[0]
    image_path = "images/" + item["image"]
    target_label = item["label"]

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path)
    plt.imshow(img)
    plt.title("Original image")
    plt.show()

    img_array = img_to_array(img)
    seed = img_array.copy()

    # Print baseline top-5 predictions
    print("\nBaseline predictions (top-5):")
    preds = model.predict(np.expand_dims(seed, axis=0))
    for cl in decode_predictions(preds, top=5)[0]:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    # Run hill climbing attack
    final_img, final_fitness = hill_climb(
        initial_seed=seed,
        model=model,
        target_label=target_label,
        epsilon=0.30,
        iterations=300
    )

    print("\nFinal fitness:", final_fitness)

    plt.imshow(array_to_img(final_img))
    plt.title(f"Adversarial Result — fitness={final_fitness:.4f}")
    plt.show()

    # Print final predictions
    final_preds = model.predict(np.expand_dims(final_img, axis=0))
    print("\nFinal predictions:")
    for cl in decode_predictions(final_preds, top=5)[0]:
        print(cl)