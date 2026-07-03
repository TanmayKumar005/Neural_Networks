# 🌸 Iris Species Classification — Neural Network

A neural network built with TensorFlow/Keras to classify Iris flower species based on sepal and petal measurements. This notebook also includes a Perceptron baseline model for comparison, along with exploratory data analysis using Seaborn visualizations.

## 🔍 Overview

The Iris dataset is one of the most well-known datasets in machine learning. This notebook trains a feedforward neural network to classify Iris flowers into three species — **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica** — using four physical measurements as input features.

## 📁 File

```
Untitled7.ipynb    ← Google Colab notebook
Iris.csv           ← Dataset (150 samples, 6 columns)
```

## 📊 Dataset

| Column | Description |
|---|---|
| `Id` | Row identifier |
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm` | Sepal width in centimeters |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm` | Petal width in centimeters |
| `Species` | Target class (Iris-setosa / Iris-versicolor / Iris-virginica) |

- **Total samples:** 150 (50 per species)
- **Features used:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species (encoded as integers)

## 🧠 Models

### 1. Perceptron (Baseline)
- Built using scikit-learn's `Perceptron`
- Used as a simple linear baseline model

### 2. Neural Network (Main Model)
```
Input (4 features) → Dense(8, ReLU) → Dropout → Dense(3, Softmax) → Output
```

| Component | Details |
|---|---|
| Framework | TensorFlow / Keras |
| Architecture | Sequential |
| Hidden Layer | Dense(8, ReLU) + Dropout |
| Output Layer | Dense(3, Softmax) |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |

## 📈 Notebook Workflow

1. **Import libraries** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras
2. **Load dataset** — Read `Iris.csv` into a Pandas DataFrame
3. **Exploratory Data Analysis** — Seaborn pair plot colored by species to visualize feature distributions and separability
4. **Preprocessing** — LabelEncoder for species labels, StandardScaler for feature normalization, one-hot encoding (`to_categorical`) for neural network output
5. **Train/test split** — Standard train/test split using scikit-learn
6. **Baseline model** — Train and evaluate a Perceptron
7. **Neural network** — Build, compile, and train the Keras Sequential model
8. **Evaluation** — Accuracy score and classification report

## 🧩 Neural Network Concepts Used

This notebook demonstrates several foundational neural network concepts:

**Activation Functions**
- **ReLU (Rectified Linear Unit)** — Used in the hidden layer. Introduces non-linearity by outputting zero for negative inputs and passing positive inputs unchanged. Helps the network learn complex patterns.
- **Softmax** — Used in the output layer for multi-class classification. Converts raw scores into probabilities that sum to 1, one per class.

**Dropout Regularization**
- Randomly "drops" (sets to zero) a fraction of neurons during training. This prevents the network from overfitting by forcing it to not rely too heavily on any single neuron.

**One-Hot Encoding**
- The three Iris species labels are converted into binary vectors (e.g., Iris-setosa → [1, 0, 0]) so the Softmax output layer can be directly compared against them.

**Categorical Crossentropy Loss**
- The loss function used for multi-class classification. Measures how far the model's predicted probability distribution is from the true one-hot encoded label. Lower loss = better predictions.

**StandardScaler (Feature Normalization)**
- Input features are scaled to have zero mean and unit variance before training. This ensures no single feature dominates learning due to difference in scale, and helps the network converge faster.

**Label Encoding**
- Species names (strings) are converted to integers before one-hot encoding, as neural networks work with numbers, not text.

**Perceptron (Baseline)**
- A single-layer linear classifier — the simplest form of a neural network. Used here as a baseline to show how much a multi-layer network improves over a simple linear model on this dataset.

**Forward Pass & Backpropagation** *(implicit)*
- During training, inputs flow forward through layers to produce predictions (forward pass), and errors flow backward to update weights (backpropagation) — the core learning mechanism of all neural networks.

## ⚠️ Disclaimer

This notebook is built for **educational purposes** to demonstrate multi-class classification using neural networks. The Iris dataset is small and clean by design — real-world datasets require more preprocessing, validation strategies, and larger models before drawing meaningful conclusions.

## 🚀 Getting Started

### Run on Google Colab (recommended)
Open the notebook directly in [Google Colab](https://colab.research.google.com/) and upload `Iris.csv` to the session — no setup required.

### Run locally
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Untitled7.ipynb
```

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn

## 📄 License

Open source — for educational use.
