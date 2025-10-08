# Deep Learning Assignment 4: Convolutional Neural Networks for Face Identification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project implements and compares different approaches for face identification using the [Labeled Faces in the Wild (LFW) dataset](http://vis-www.cs.umass.edu/lfw/). The project demonstrates the superiority of Convolutional Neural Networks (CNNs) over traditional computer vision methods like Eigenfaces for face recognition tasks.

## Project Goals

- **Compare Traditional Methods**: Evaluate the performance of the classical Eigenfaces approach
- **Implement CNN Models**: Build multiple CNN architectures with different optimizers
- **Analyze Performance**: Compare different deep learning approaches
- **Visualize Results**: Display correctly and incorrectly identified images with training samples

## Key Features

- **Baseline Implementation**: Eigenfaces with Principal Component Analysis (PCA)
- **Multiple CNN Architectures**: 4 different CNN models with varying complexity
- **Optimizer Comparison**: Adam, RMSprop, and SGD with momentum
- **Data Augmentation**: Advanced preprocessing techniques
- **Visualization Tools**: Comprehensive result analysis and visualization

## Dataset

- **Source**: Labeled Faces in the Wild (LFW) dataset
- **Images**: 3,023 grayscale face images
- **Subjects**: 62 different people (classes)
- **Image Size**: 62×47 pixels
- **Split**: 90% training, 10% testing (stratified)

## Technical Requirements

### System Requirements
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU support (optional, for faster training)

### Dependencies
```bash
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install tensorflow>=2.0.0
pip install keras
```
## 🔬 Methodology

### 1. Baseline Method: Eigenfaces
- Principal Component Analysis (PCA) for dimensionality reduction
- Logistic regression classifier
- Feature extraction from flattened image vectors

### 2. CNN Models
- **Model 1**: Deep CNN with Adam optimizer
- **Model 2**: Lightweight CNN with RMSprop
- **Model 3**: Residual CNN with SGD+Momentum
- **Model 4**: Optimized CNN with HeUniform initialization

### 3. Key Techniques
- Batch normalization for stable training
- Dropout for regularization
- Data augmentation for improved generalization
- Class weight balancing for imbalanced datasets

## Results

The project demonstrates significant improvements of CNN models over traditional Eigenfaces:

- **Eigenfaces Baseline**: ~3.3% test accuracy
- **CNN Models**: Significantly higher accuracy with different architectures
- **Best Model**: Achieves >70% test accuracy target

## Model Architectures

### Model 1: Deep CNN with Adam
- Multiple convolutional layers with batch normalization
- Data augmentation with rotation, shifting, and zooming
- Adam optimizer with adaptive learning rates

### Model 2: Lightweight CNN with RMSprop
- Compact architecture for efficiency
- RMSprop optimizer with decay
- Minimal data augmentation

### Model 3: Residual CNN with SGD+Momentum
- Residual connections to prevent vanishing gradients
- SGD with Nesterov momentum
- Advanced data augmentation

### Model 4: Optimized CNN with HeUniform
- HeUniform initialization for better gradient flow
- L2 regularization
- Comprehensive data augmentation

## Visualization

The notebook includes comprehensive visualizations:
- **Eigenfaces**: First 10 principal components
- **Training Progress**: Loss and accuracy curves
- **Results Analysis**: Correctly and incorrectly classified images
- **Model Comparison**: Performance metrics across different approaches

## Educational Value

This project demonstrates:
- **Traditional vs. Modern**: Comparison of classical and deep learning approaches
- **Architecture Design**: Different CNN architectures and their trade-offs
- **Optimization**: Various optimizers and their effects
- **Regularization**: Techniques to prevent overfitting
- **Data Augmentation**: Methods to improve model generalization

## Customization

### Modifying Models
- Adjust network architecture in the model definition sections
- Experiment with different optimizers and learning rates
- Try different data augmentation strategies

### Hyperparameter Tuning
- Learning rates: 0.0001 to 0.01
- Batch sizes: 32, 64, 128
- Epochs: 15-50 depending on convergence
- Dropout rates: 0.25 to 0.5

## References

- [Labeled Faces in the Wild Dataset](http://vis-www.cs.umass.edu/lfw/)
- [Eigenfaces Method](https://en.wikipedia.org/wiki/Eigenface)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- LFW dataset creators for providing the face recognition dataset
- TensorFlow and Keras teams for the excellent deep learning frameworks
- The open-source community for various libraries and tools
