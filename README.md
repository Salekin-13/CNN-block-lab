# 🧠 CNN Architectures for Image Classification

This repository explores and compares several convolutional neural network (CNN) architectures for image classification tasks using the CIFAR-10 dataset. It goes step-by-step from the base classical model LeNet, to modifying the network using dilated convolution, better activation function and batchnorms, then goes to explore the benefits of depthwise separable convolution for a lightweight model, to then finally under the effect of residual blocks to make up for lost accuracy.

## 📁 Project Structure

```bash
intro-to-ml-basics/
├── utils/              # Reusable Python modules
│   ├── dataset_loader.py
│   ├── model_architecture.py
│   └── train.py
├── experiments.ipynb            # Experiments and visualizations
├── requirements.txt
└── README.md
```

## 🏗️ Models Implemented

### 1. **LeNet**

A classic architecture with:

* Tanh activations
* Average pooling
* 2 convolutional layers and 3 fully connected layers

### 2. **mod\_CNN**

An enhanced version of LeNet featuring:

* Additional convolutional layers
* Batch normalization
* Average pooling
* ReLU activations
* Dilated convolution for larger receptive fields
* Dropout for regularization

### 3. **mod\_CNN\_depthwise**

A lightweight alternative using:

* Depthwise separable convolutions for computational efficiency
* Batch normalization and ReLU
* Similar architecture depth to mod\_CNN but with fewer parameters

### 4. **mod\_CNN\_depth\_res**

Similar architecture with additional:

* Residual Blocks for better gradient flow

## 📦 Dependencies

* Python 3.10+
* PyTorch
* torchvision
* matplotlib
* numpy

Install the dependencies using:


## 🚀 Getting Started

1. Clone this repo:

   ```bash
   git clone https://github.com/yourusername/cnn-image-classification.git
   cd cnn-image-classification
   ```

2. Intrall requirements using:
    ```bash
    pip install -r requirements.txt
    ```
    
3. Run the notebook:
   Open `experiments.ipynb` in Jupyter Notebook or VS Code to explore training logs, accuracy plots, and comparisons.

## 📊 Results Overview

| Model Name               | Accuracy (Test) | Parameters | Training Time | Notes                                      |
| ------------------------ | --------------- | ---------- | ------------- | ------------------------------------------ |
| **LeNet**                | \~57.25%       | 62,006      | \~5m 35s      | Classical model, good baseline             |
| **mod\_CNN**             | \~70.98%        | 127,858    | \~7m 56s      | BatchNorm, dilation, better generalization |
| **mod\_CNN\_depthwise**  | \~61.36%        | 109,196      | \~4m 51s      | Lightweight, faster, lower accuracy          |
| **mod\_CNN\_depth\_res** | \~67.13%        | 235,868     | \~5m 19s      | Residual + depthwise, lightweight with increased accuracy       |

---

The notebook also includes:

* Insights on architecture choices (e.g., impact of dilation or depthwise convolutions)

## 📌 Highlights

* Modular design for easy model switching
* Insightful comments and architecture intuition
* Depthwise CNN for embedded or low-resource deployment

## 📚 References

* LeCun et al. (1998) [LeNet-5](http://yann.lecun.com/exdb/lenet/)
* Howard et al. (2017) [MobileNets](https://arxiv.org/abs/1704.04861)

## 🧑‍💻 Author

Made by \[Sumaiya Salekin]
to better understand the tradeoff between speed and performance 🌱
