# Robustness-Evaluation-of-CNN-ResNet-and-CLIP-Models-Against-Adversarial-Attacks
Robustness evaluation of CNN, ResNet, and CLIP models against adversarial attacks including FGSM, PGD, Spatial, TDPA, and BA using CIFAR-10, MNIST, and image-caption datasets


# Robustness Comparison of Neural Network Models Against Adversarial Attacks and Preventive Measures

## Overview

This repository presents a comprehensive evaluation of neural network robustness against various adversarial attacks. The study compares the vulnerability of different architectures—CNNs, ResNets, and CLIP (Contrastive Language-Image Pretraining)—to attacks such as Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Spatial Attacks, Targeted Data Poisoning Attacks (TDPA), and Backdoor Attacks (BA). The effectiveness of SafeCLIP as a defense mechanism is also assessed.


## Datasets

### CIFAR-10
- 60,000 color images across 10 classes
- Image size: 32x32 pixels
- Used for CNN and ResNet training and evaluation

### MNIST
- 70,000 grayscale images of handwritten digits (0–9)
- Image size: 28x28 pixels
- Used to evaluate CNN and ResNet robustness

### Image-Caption Dataset
- Taken from Kaggle: [Flickr8k Image Captioning Dataset](https://www.kaggle.com/datasets/aladdinpersson/flickr8kimagescaptions)
- Used for training and evaluating the CLIP model

## Models

- **CNN**: Trained on MNIST and CIFAR-10
- **ResNet**: Trained on MNIST and CIFAR-10
- **CLIP**: Trained on image-caption dataset using contrastive learning

## Attacks

### Fast Gradient Sign Method (FGSM)
Single-step perturbation using the sign of the gradient.

### Projected Gradient Descent (PGD)
Iterative attack that applies multiple steps of FGSM with projection back into epsilon-ball.

### Spatial Attack
Applies geometric transformations like translation, rotation, and scaling.

### Targeted Data Poisoning Attack (TDPA)
Modifies the labels of selected training data to force targeted misclassification.

### Backdoor Attack (BA)
Injects triggers into training data so that their presence during inference causes incorrect classification.

## Defense: SafeCLIP

### Unimodal Contrastive Learning
Separates image and text features to avoid poisoned associations.

### Gaussian Mixture Modeling (GMM)
Identifies safe and risky image-caption pairs using cosine similarity distribution.

### Selective CLIP Loss
Applies contrastive loss only on safe data, while training risky data using unimodal CL losses.

## Performance Metrics

- **Clean Accuracy**: Accuracy on original data
- **Adversarial Accuracy**: Accuracy on adversarial examples
- **Attack Success Rate**: Proportion of successful adversarial manipulations

## Experimental Procedure

### CNN and ResNet
1. Train on 80% of CIFAR-10 and MNIST data
2. Apply FGSM, PGD, and Spatial attacks on test data
3. Measure performance metrics

### CLIP
1. Poison the image-caption training data using TDPA and BA
2. Train CLIP on poisoned data
3. Apply SafeCLIP and evaluate robustness

## Results Summary

- ResNet shows higher robustness than CNN under all attack types
- CNN exhibits significant accuracy degradation under PGD and FGSM
- CLIP is vulnerable to poisoning, but SafeCLIP significantly reduces attack success rates while preserving accuracy

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)  
2. Deng, L. (2012). The MNIST Database of Handwritten Digit Images for Machine Learning Research. IEEE Signal Processing Magazine  
3. Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. University of Toronto  
4. Yang, W., Gao, J., & Mirzasoleiman, B. (2024). Better Safe than Sorry: Pre-training CLIP against Targeted Data Poisoning and Backdoor Attacks. ICML 2024

## License

This repository is intended for academic and research purposes only.
