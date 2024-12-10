# 🔒 Comprehensive Label Flipping Attack on MNIST Classification


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

Welcome to the **Comprehensive Label Flipping Attack on MNIST Classification** application! This interactive Streamlit app demonstrates the impact of **Label Flipping** data poisoning attacks on a Convolutional Neural Network (CNN) trained to classify handwritten digits from the MNIST dataset. By manipulating training labels, this app showcases how malicious alterations can degrade model performance and introduce specific vulnerabilities.

## Features

- **Customizable Label Flipping Attack:**
  - **Source Label Selection:** Choose the digit label you want to flip from (e.g., 7).
  - **Target Label Selection:** Choose the digit label you want to flip to (e.g., 1).
  - **Flip Ratio Adjustment:** Determine the percentage of source labels to flip (up to 50%) for a stronger attack.

- **Model Training and Retraining:**
  - **Original Model:** Trained on clean MNIST data.
  - **Poisoned Model:** Retrained on data with flipped labels to demonstrate the attack's effect.

- **Interactive Predictions Comparison:**
  - View side-by-side predictions from both models on selected test images.
  - **Flipped Samples Indicator:** Easily identify images affected by the label flipping attack.

- **Comprehensive Evaluation Metrics:**
  - **Accuracy and Loss:** Compare overall performance of both models.
  - **Confusion Matrices:** Visualize class-wise performance differences.
  - **Prediction Distribution Charts:** Understand shifts in prediction patterns due to the attack.
  - **Precision, Recall, and F1-Score:** Dive deeper into the impact on specific labels.

- **Summary Analysis:**
  - Quantitative insights into how many labels were affected.
  - Visual representations of the attack's effectiveness.

## How It Works

1. **Label Flipping Attack:**
   - Select a **Source Label** (the original class you want to manipulate).
   - Select a **Target Label** (the class you want the source label to be misclassified as).
   - Adjust the **Flip Ratio** to determine the proportion of source labels to flip.

2. **Model Training:**
   - The **Original Model** is trained on the unaltered MNIST dataset.
   - Upon applying the attack, the **Poisoned Model** is retrained on the dataset with flipped labels, showcasing the attack's impact.

3. **Model Evaluation:**
   - Compare predictions between the original and poisoned models.
   - Evaluate overall performance metrics and delve into detailed analyses to understand the attack's effectiveness.

## Installation

### Prerequisites

- **Python 3.7 - 3.9**
- **pip** package manager

### Package Installation
```bash
pip install -r requirements.txt
```


### Usage
```bash
streamlit run app.py
```