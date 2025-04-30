# Neural Networks: Optimizing Performance with Adaptive Activation Functions

This project is dedicated to enhancing neural network performance through the use of adaptive parametric activation functions.


## Overview

This repository is designed to explore and implement adaptive parametric activation functions to increase the performance of convolutional neural networks especially on long-tailed datasets where the accuracy of rare-class samples is increased. 

---

## Directory Structure

### 1. `datasets`
Contains scripts and utilities for dataset preparation and loading.
- **`ImageNetLTLoader.py`**: A loader for handling the ImageNet-LT dataset.
- **`script.py`**: Helper script for dataset-related operations.
- **`scriptImageNetLT.py`**: Specific script for handling ImageNet-LT preprocessing.

[View more in the `datasets` directory](https://github.com/venusai24/neural-networks/tree/main/datasets)

---

### 2. `experiments`
Houses training and testing checkpoints to continue training a pretrained model or retrain a model from a particular epoch.

[View more in the `experiments` directory](https://github.com/venusai24/neural-networks/tree/main/experiments)

---

### 3. `initial_implementation`
The initial implementation of neural network models and utilities where we implemented the base paper whose link is given below.
[Base Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07153.pdf)

[View the complete list in the `initial_implementation` directory](https://github.com/venusai24/neural-networks/tree/main/initial_implementation)

---

### 4. `modifications`
Contains modified implementations and utilities for adaptive activation functions where we modified the existing adaptive activation function ie. APA and developed a hybrid activation module combining APA, AdAct-style hinges, and frequency-conditioned parameters, resulting in a more flexible and adaptive network. Frequency-Conditioned parameters dynamically adjust activation steepness (κ) based on input frequency content.

$$
\text{HybridAPA}(z) = \underbrace{(\lambda e^{-\kappa z} + 1)^{-1/\lambda}}_{\text{APA}} + \underbrace{\sum_{h \in \mathcal{H}} \max(0, z - h)}_{\text{AdAct Hinges}}
$$

where

$$
\kappa = \kappa_{\text{base}} + \alpha \cdot \text{FFT-Mag}(z)
$$

and

$$
\mathcal{H} = \{0.2, 0.5, 0.8\}
$$

- $\kappa$ is the frequency-adapted steepness.
- $\mathcal{H}$ are the learnable hinge thresholds.
- $\lambda$ is the asymmetry parameter.
- $\text{FFT-Mag}(z)$ is the average magnitude of the FFT of $z$.

[View the complete list in the `modifications` directory](https://github.com/venusai24/neural-networks/tree/main/modifications)

---

### 5. `requirements`
Includes dependency management files.
- **`build.txt`**: Build-time dependencies.
- **`runtime.txt`**: Runtime dependencies.
- **`optional.txt`**: Optional packages for extended functionalities.

[View more in the `requirements` directory](https://github.com/venusai24/neural-networks/tree/main/requirements)

---

### Additional Files
- **`README.md`**: Project documentation (this file).
- **`LICENSE`**: Licensing information.
- **`requirements.txt`**: General dependency list.

---

## Getting Started

To get started with this repository:
1. Clone the repo:
   ```bash
   git clone https://github.com/venusai24/neural-networks.git
2. Install required dependencies
   ```bash
   pip install -r requirements.txt

