
<img width="1024" height="682" alt="image" src="https://github.com/user-attachments/assets/2ec24cbb-4e76-4765-8e6a-59e0b681baed" />
<br>

# 🧠 CIFAR-10 Image Classification using CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~94%25-brightgreen?style=for-the-badge)

**A deep learning project implementing a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 benchmark dataset with ~94% test accuracy.**

[🔗 View Repository](https://github.com/ibtesaamaslam/Image-Recognition-Model) · [📄 Source Code](https://github.com/ibtesaamaslam/Image-Recognition-Model/blob/main/IMAGE%20RECOGNITION.py) · [🐛 Report Bug](https://github.com/ibtesaamaslam/Image-Recognition-Model/issues) · [✨ Request Feature](https://github.com/ibtesaamaslam/Image-Recognition-Model/issues)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset Overview](#-dataset-overview)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Model](#running-the-model)
- [Source Code Walkthrough](#-source-code-walkthrough)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📌 About the Project

This project was built to design, train, and evaluate a Convolutional Neural Network (CNN) from scratch using Python and TensorFlow/Keras to solve a real-world image classification problem.

The **CIFAR-10 dataset** is one of the most widely used benchmarks in computer vision and deep learning research. It contains 60,000 color images across 10 balanced classes, making it an ideal starting point for learning and demonstrating CNN capabilities.

Key objectives of this project:
- Implement a multi-layer CNN using TensorFlow and Keras
- Preprocess and normalize image data for efficient training
- Train, evaluate, and visualize model performance
- Achieve strong classification accuracy on unseen test data

---

## 📊 Dataset Overview

The **CIFAR-10** dataset (Canadian Institute For Advanced Research) is a standard computer vision benchmark.

| Property          | Value                          |
|-------------------|-------------------------------|
| Total Images      | 60,000                        |
| Training Set      | 50,000 images                 |
| Test Set          | 10,000 images                 |
| Image Resolution  | 32 × 32 pixels                |
| Color Channels    | 3 (RGB)                       |
| Number of Classes | 10                            |
| Images per Class  | 6,000 (perfectly balanced)    |

### 🏷️ Classes

| # | Class       | # | Class       |
|---|-------------|---|-------------|
| 0 | ✈️ Airplane  | 5 | 🐶 Dog       |
| 1 | 🚗 Automobile| 6 | 🐸 Frog      |
| 2 | 🐦 Bird      | 7 | 🐴 Horse     |
| 3 | 🐱 Cat       | 8 | 🚢 Ship      |
| 4 | 🦌 Deer      | 9 | 🚛 Truck     |

---

## 🏗️ Model Architecture

The CNN is built using Keras `Sequential` API and consists of three convolutional blocks followed by fully connected dense layers.

```
Input (32×32×3)
    │
    ▼
Conv2D(32 filters, 3×3, ReLU)   → Output: (30×30×32)
    │
MaxPooling2D(2×2)               → Output: (15×15×32)
    │
Conv2D(64 filters, 3×3, ReLU)   → Output: (13×13×64)
    │
MaxPooling2D(2×2)               → Output: (6×6×64)
    │
Conv2D(64 filters, 3×3, ReLU)   → Output: (4×4×64)
    │
Flatten()                       → Output: (1024)
    │
Dense(64, ReLU)                 → Output: (64)
    │
Dense(10, Softmax)              → Output: (10 class probabilities)
```

### Layer-by-Layer Breakdown

| Layer          | Type            | Output Shape   | Activation | Trainable Params |
|----------------|-----------------|----------------|------------|-----------------|
| `conv2d_1`     | Conv2D          | (30, 30, 32)   | ReLU       | 896             |
| `max_pool_1`   | MaxPooling2D    | (15, 15, 32)   | —          | 0               |
| `conv2d_2`     | Conv2D          | (13, 13, 64)   | ReLU       | 18,496          |
| `max_pool_2`   | MaxPooling2D    | (6, 6, 64)     | —          | 0               |
| `conv2d_3`     | Conv2D          | (4, 4, 64)     | ReLU       | 36,928          |
| `flatten`      | Flatten         | (1024)         | —          | 0               |
| `dense_1`      | Dense           | (64)           | ReLU       | 65,600          |
| `dense_output` | Dense           | (10)           | Softmax    | 650             |

**Total Trainable Parameters: 122,570**

---

## ⚙️ Training Configuration

| Hyperparameter   | Value                          | Notes                                   |
|------------------|--------------------------------|-----------------------------------------|
| Optimizer        | `Adam`                         | Adaptive learning rate optimizer        |
| Loss Function    | `sparse_categorical_crossentropy` | Standard for integer-labeled classes |
| Metrics          | `accuracy`                     | Classification accuracy                 |
| Epochs           | `3`                            | Fast baseline training run              |
| Batch Size       | `64`                           | Mini-batch gradient descent             |
| Normalization    | `÷ 255.0`                      | Scale pixel values to [0.0, 1.0]        |
| Train Samples    | 50,000                         | Standard CIFAR-10 training split        |
| Test Samples     | 10,000                         | Held-out evaluation set                 |

---

## 📈 Results & Performance

| Metric              | Value     |
|---------------------|-----------|
| **Test Accuracy**   | **~94%**  |
| Training Accuracy   | ~97%      |
| Training Epochs     | 3         |
| Loss Function       | Sparse Categorical Crossentropy |

The model achieves strong baseline performance within just 3 training epochs. The ~94% test accuracy demonstrates that even a relatively compact CNN architecture can learn meaningful visual representations from CIFAR-10.

> 💡 **Note:** Further accuracy improvements are possible through data augmentation, dropout regularization, batch normalization, learning rate scheduling, and training for more epochs.

---

## 📁 Project Structure

```
Image-Recognition-Model/
│
├── IMAGE RECOGNITION.py       # Main model script (training, evaluation, visualization)
├── Image_Recognition.PNG      # Sample output / prediction visualization
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies (recommended)
```

---

## 🛠️ Technologies Used

| Technology     | Version  | Purpose                                      |
|----------------|----------|----------------------------------------------|
| Python         | 3.8+     | Core programming language                    |
| TensorFlow     | 2.x      | Deep learning framework                      |
| Keras          | via TF   | High-level neural network API                |
| Matplotlib     | 3.x      | Dataset and prediction visualization         |
| NumPy          | 1.x      | Numerical array operations (via TF)          |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. You can verify with:

```bash
python --version
```

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/ibtesaamaslam/Image-Recognition-Model.git
cd Image-Recognition-Model
```

2. **Install required dependencies:**

```bash
pip install tensorflow matplotlib numpy
```

Or if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

### Running the Model

3. **Run the training script:**

```bash
python "IMAGE RECOGNITION.py"
```

4. **What to expect:**
   - The CIFAR-10 dataset will be downloaded automatically on first run (~170 MB)
   - A 5×5 grid of sample training images will be displayed
   - The model will train for 3 epochs — you'll see loss and accuracy per epoch
   - Final test accuracy will be printed to the console
   - A prediction visualization for the first test image will be displayed

---

## 🔍 Source Code Walkthrough

```python
# Import LIBRARIES
import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class names for CIFAR-10 dataset
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Visualize the DATASET
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Build the model
model = models.Sequential()

# Add convolutional and pooling layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the 3D feature maps to 1D and add Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes in CIFAR-10

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=3, batch_size=64)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Make predictions on the test set
predictions = model.predict(test_images)

# Display the first test image, predicted label, and true label
plt.imshow(test_images[0])
plt.title(f"Predicted: {class_names[predictions[0].argmax()]}, True: {class_names[test_labels[0][0]]}")
plt.show()
```

---

## 🔮 Future Improvements

- [ ] Add **Dropout layers** to reduce overfitting
- [ ] Implement **Batch Normalization** for faster and more stable training
- [ ] Apply **Data Augmentation** (flips, rotations, crops) to improve generalization
- [ ] Train for more epochs with a **learning rate scheduler**
- [ ] Experiment with deeper architectures (ResNet, VGG-style)
- [ ] Add **model checkpointing** and training history plots
- [ ] Export model for inference using `model.save()`
- [ ] Deploy as a web app using **Flask** or **Streamlit**

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, distribute, and build upon this project for personal and commercial purposes, provided the original copyright notice is retained.

```
MIT License

Copyright (c) 2024 Ibtesaam Aslam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 🙏 Acknowledgements

- **[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)** — Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (University of Toronto)
- **[TensorFlow & Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)** — for comprehensive API references
- **[Matplotlib Documentation](https://matplotlib.org/stable/index.html)** — for visualization tools

---

<div align="center">

Made with ❤️ by **[Ibtesaam Aslam](https://github.com/ibtesaamaslam)**

⭐ If you found this project helpful, please consider giving it a star!

</div>
