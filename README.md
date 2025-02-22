# Digit-Classification
## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Description
This project implements a neural network from scratch using TensorFlow to classify handwritten digits from the MNIST dataset. The goal is to demonstrate a clear understanding of neural network fundamentals, including forward propagation, backpropagation (using TensorFlow's GradientTape), and gradient descent.

Key components of the project include:
- Custom implementation of dense layers.
- A sequential model to stack layers.
- Batch training for efficient learning.
- Manual backpropagation using TensorFlow's GradientTape.



## Features
- Custom dense layer implementation.
- Sequential model for stacking layers.
- Batch training with adjustable batch size.
- Manual backpropagation using TensorFlow's GradientTape.
- Achieves ~95% accuracy on the MNIST test set.


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AlexGundrum/Digit-Classification.git
   cd Digit-Classification
   ```
2. Install the dependencies:
   ```bash
   pip install tensorflow numpy
   ```
## Usage
1. Run the script to train the model:
   ```bash
   python mnist_classifier.py
   ```
