# 🔢 MNIST Handwritten Digit Classification

A progressive deep learning project that benchmarks three neural network architectures — Perceptron, ANN, and CNN — on the MNIST handwritten digit dataset. Each model is trained and evaluated side-by-side to demonstrate how increasing architectural complexity improves classification accuracy.

## 🔍 Overview

The notebook classifies handwritten digits (0–9) from the MNIST dataset using three increasingly powerful models. Results show a clear accuracy progression from a simple Perceptron all the way to a Convolutional Neural Network, illustrating the practical value of deeper and more specialised architectures for image data.

## 📁 File

```
MNIST_dataset_model.ipynb   ← Google Colab notebook
train.csv                   ← Training dataset (42,000 samples, 785 columns)
test.csv                    ← Test dataset
```

## 📊 Dataset

The MNIST dataset contains grayscale images of handwritten digits, each 28×28 pixels, flattened into 784 pixel values.

| Column | Description |
|---|---|
| `label` | Target digit (0–9) |
| `pixel0` – `pixel783` | Pixel intensity values (0–255) |

- **Training samples:** 42,000
- **Test samples:** Available separately via `test.csv`
- **Classes:** 10 (digits 0 through 9)
- **Source:** Downloaded via `kagglehub` (`hojjatk/mnist-dataset`)

## 🧠 Models

### 1. Perceptron (Baseline)
```
Input (28×28) → Flatten → Dense(10, Softmax)
```

| Detail | Value |
|---|---|
| Optimizer | SGD |
| Loss | Categorical Crossentropy |
| Epochs | 5 |
| Final Val Accuracy | **90.06%** |

### 2. Artificial Neural Network (ANN)
```
Input (28×28) → Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

| Detail | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 5 |
| Final Val Accuracy | **97.24%** |

### 3. Convolutional Neural Network (CNN)
```
Input (28×28×1) → Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Conv2D(64, 3×3, ReLU)
→ MaxPool(2×2) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

| Detail | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 5 |
| Final Val Accuracy | **98.98%** |

## 📈 Results Summary

| Model | Final Validation Accuracy |
|---|---|
| Perceptron | 90.06% |
| ANN | 97.24% |
| **CNN** | **98.98%** |

The CNN outperforms both the Perceptron and ANN by a significant margin, demonstrating the power of convolutional layers for spatial image data.

## 🧩 Neural Network Concepts Used

**Flatten** — Converts the 2D image (28×28) into a 1D vector (784 values) so fully connected layers can process it.

**Dense (Fully Connected Layer)** — Connects every neuron in one layer to every neuron in the next. Used for learning complex decision boundaries.

**ReLU Activation** — Returns zero for negative values and passes positive values unchanged. Introduces non-linearity and helps avoid the vanishing gradient problem.

**Softmax Activation** — Used in the output layer to convert raw scores into class probabilities that sum to 1, enabling multi-class classification.

**Conv2D** — Extracts spatial features from images using learnable filters. Captures patterns like edges, curves, and shapes — critical for image recognition.

**MaxPooling2D** — Reduces the spatial size of feature maps by taking the maximum value in each region. Decreases computation and helps the model become invariant to small shifts.

**Dropout** — Randomly sets a fraction of neuron activations to zero during training to prevent overfitting and improve generalisation.

**Categorical Crossentropy Loss** — Measures how far the predicted probability distribution is from the true one-hot encoded label. Standard loss for multi-class classification.

**Adam Optimizer** — Adaptive learning rate optimiser that combines momentum and RMSProp, generally faster and more stable than plain SGD.

**One-Hot Encoding (`to_categorical`)** — Converts integer labels (e.g., 5) into binary vectors (e.g., [0,0,0,0,0,1,0,0,0,0]) required for Softmax + Crossentropy training.

**Confusion Matrix** — A 10×10 matrix visualising where the CNN makes correct and incorrect predictions across all digit classes.

## 🔄 Notebook Workflow

1. **Import libraries** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras
2. **Load dataset** — Download via `kagglehub` and read `train.csv` and `test.csv`
3. **EDA & Preprocessing** — Check nulls, normalise pixel values to [0, 1], reshape for each model type, one-hot encode labels
4. **Perceptron** — Build, compile, train, evaluate
5. **ANN** — Build with two hidden layers, compile, train, evaluate
6. **CNN** — Build with two conv blocks + dropout, compile, train, evaluate
7. **Visualisation** — Training/validation accuracy & loss curves per model, side-by-side comparison chart, confusion matrix for CNN

## 🚀 Getting Started

### Run on Google Colab (recommended)
Open the notebook in [Google Colab](https://colab.research.google.com/) and run all cells — `kagglehub` handles dataset download automatically.

### Run locally
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn kagglehub
jupyter notebook MNIST_dataset_model.ipynb
```

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn
- kagglehub (dataset download)

## 📄 License

Open source — for educational use.
