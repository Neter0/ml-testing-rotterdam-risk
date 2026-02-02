# Assignment 2 — Adversarial Attacks vs. Hill-Climbing Search

**DSAIT4 – Testing AI Systems**

This assignment extends our work on hill-climbing–based adversarial image generation by introducing standard adversarial attack baselines using the CleverHans library (FGM + PGD).
We compared our method against these attacks on the same set of images.
    

## Repository Structure
```
├── baselines.py                # Provided: CleverHans FGM + PGD attacks on student images
├── hill_climbing.py            # Our implementation
├── images/                     # Input images to attack
│    ├── fish.jpg
│    ├── castle.jpg
│    ├── ...
├── data/
│    ├── image_labels.json      # Human-provided labels for the images
│    ├── imagenet_classes.txt   # List of ImageNet classes
├── requirements.txt
└── README.md
```

## Installation

We strongly recommend using a virtual environment (e.g., ``venv``):

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows PowerShell
```

To install dependencies, run the command:

```bash
pip install -r requirements.txt
```

## Running the baselines:

The baselines are implemented by CleverHans library  and can be executed via```baselines.py```.  
The script will:

* Load each image listed in data/image_labels.json
* Predict the clean label using VGG-16
* Apply FGM and PGD attacks from CleverHans
* Save attacked images to attack_results/
* Print prediction shifts for clean vs adversarial samples

You can simply run the baselines with:

```bash
python baselines.py
```

And it will produce output like:

```Image: fish.jpg 
Human label: goldfish                       --> This the seed image for the attack
Model prediction (clean): goldfish (0.949)  --> Original prediction (with confidence level)
FGM prediction: stole (0.255)               --> Prediction of the attack image by FGM (with confidence level)
PGD prediction: Maltese dog (1.000)         --> Prediction of the attack image by PGD (with confidence level)
````