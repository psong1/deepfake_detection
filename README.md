# Deepfake Detection: Real vs. Fake Photos/Videos

## Description

A modular, hardware-accelerated deep learning pipeline designed to detect facial manipulations in images in videos. This projects utilizes ResNet18 backbone for spatial feature extraction and MTCNN for robust face localization, optimized for NVIDIA RTX 40-series GPUs.

## Key Features

- **Modular Architecture**: Separated concerns for data acquisition, model utilities, visualization, training, and inference.
- **Hardware Optimized**: Leverages PyTorch Automated Mixed Precision (AMP) to utilize Tensor Cores on RTX 4070, reducing training time by ~50% without loss in accuracy.
- **Unified Inference**: A single inference for processing both static images and video streams using a temporal voting strategy.
- **Automated Workflow**: Scripted data acquistion via the Kaggle API and environment-aware data loading for Windows/Linux compatibility.

## System Architecture

1. **Data Acquisition** (`download_dataset.py`)

   Automates the setup process by authenticating with Kaggle API and managing local dataset persistence to ensure a reproducible development environment.

2. **Shared Utilities** (`model_utils.py`)

   Acts as the single source of truth for the model architecture and preprocessing logic.

- Standardized ImageNet-level normalization and resizing.
- Manages `DataLoader` configurations, including `pin_memory` and Windows-specific `num_workers` optimizations.

3. **Training Engine** (`deepfake_detection.py`)

   Handles the core machine learning lifecycle:

- **Optimization**: Uses the Adam optimizer and Cross-Entropy loss.
- **Performance**: Implements `torch.amp` for accelerated 16-bit training.
- **Validation**: Performs real-time validation after each epoch and saves only the highest-performing weights.

4. **Inference Engine** (`inference.py`)

   Transforms the trained model into a usable tool:

- **Face Localization**: Integrates MTCNN to crop and align faces, ensuring the classifier focuses only on relevant spatial artifacts.
- **Video Analysis**: Samples frames at variable rates and aggregates scores to generate a confidence rating for entire video files.

## Tech Stack

- **Language**: Python 3.12
- **Framework**: PyTorch
- **Computer Vision**: OpenCV, PIL, MTCNN
- **Optimization**: NVIDIA CUDA 12.1, AMP
- **Visualization**: Matplotlib

## Process

1. **Data Preparation**

- This project uses supervised learning because every image comes with label. Raw images are messy so we extract the features using ResNet18.
  - ResNet18 is chosen as the "brain" because deep learning can suffer from the Vanishing Gradient Problem where the signal can get lost as it travels back through the layers.
  - ResNet allows information to bypass certain layers and this ensures that gradients can flow back to the early layers without disappearing, even in deep networks.
- To normalize the data, we subtract the mean and divide by the standard deviation of the ImageNet dataset. This normalization helps the optimizer converge faster because the gradients won't be too large or small.
- Dataset is split into three distinct folders (Train, Validation, and Test) to prevent overfitting.

2. **Optimization**

- Cross-Entropy Loss is used to measure error and penalize the model heavily if it is confident about a wrong answer.
- Adam Optimizer calculates an "individual learning rate" for every single parameter. If a parameter is changing rapidly, Adam slows down; if it's barely moving, Adam speeds up. This makes it much more robust and less likely to get stuck in local "potholes."

3. **Training Loop: Backpropagation**

- Forward Pass: The image goes through the ResNet and the model makes a guess
- Loss Calculation: Compare the guess to the actual label
- Backward Pass: Calculate the gradient to move the weights to reduce the loss.
