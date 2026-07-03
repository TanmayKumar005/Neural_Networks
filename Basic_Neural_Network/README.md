# 🌱 Plant Watering Prediction — Basic Neural Network

A beginner-level deep learning project that uses a feedforward neural network built with TensorFlow/Keras to predict whether a plant needs water, based on environmental sensor readings.

## 🔍 Overview

Given 3 input features — soil moisture, temperature, and sunlight hours — the model learns to classify whether a plant needs watering (`1`) or not (`0`). The dataset is small and hand-crafted (16 records), making this a great introductory notebook for understanding how neural networks are built, trained, and evaluated from scratch.

## 📁 File

```
Basic_neural_network.ipynb   ← Google Colab notebook
```

## 🧠 Model Architecture

```
Input (3 features) → Dense(8, ReLU) → Dense(1, Sigmoid) → Output
```

| Layer | Units | Activation | Purpose |
|---|---|---|---|
| Input | 3 | — | soil_moisture, temperature_c, sunlight_hours |
| Hidden | 8 | ReLU | Feature learning |
| Output | 1 | Sigmoid | Binary classification (needs water: yes/no) |

## 📊 Dataset

Hand-crafted dataset with 16 samples representing different environmental conditions:

| Feature | Range | Description |
|---|---|---|
| `soil_moisture` | 0.05 – 0.80 | Fraction of soil water content |
| `temperature_c` | 19°C – 35°C | Ambient temperature in Celsius |
| `sunlight_hours` | 1 – 12 hrs | Daily sunlight exposure |
| `needs_water` | 0 or 1 | Target label (1 = needs water) |

Train/test split: **75% train / 25% test** with `stratify=y` to preserve class balance.

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Optimizer | SGD (Stochastic Gradient Descent) |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 100 |
| Batch Size | 4 |

## 📈 Results

| Metric | Train | Validation |
|---|---|---|
| Accuracy | 91.67% | **100%** |
| Loss (final) | ~0.368 | ~0.382 |

The model achieved 100% validation accuracy from epoch 1 and maintained it through all 100 epochs, with training loss steadily decreasing.

## 🔄 Notebook Workflow

1. **Import libraries** — `pandas`, `numpy`, `tensorflow`, `keras`, `sklearn`
2. **Create dataset** — 16-sample DataFrame with 3 features and 1 binary label
3. **Feature/target split** — `X = [soil_moisture, temperature_c, sunlight_hours]`, `y = needs_water`
4. **Manual min-max normalization** — scales all features to [0, 1] range using `(X - X_min) / (X_max - X_min + 1e-8)` (epsilon added to avoid division by zero)
5. **Train/test split** — `train_test_split` with `test_size=0.25`, `stratify=y`, `random_state=42`
6. **Build model** — Keras Sequential with Input → Dense(8, ReLU) → Dense(1, Sigmoid)
7. **Compile** — SGD optimizer, binary crossentropy loss
8. **Train** — 100 epochs, batch size 4, with validation data monitored per epoch

## 🚀 Getting Started

### Run on Google Colab (recommended)
Open the notebook in [Google Colab](https://colab.research.google.com/) and run all cells — no setup needed.

### Run locally
```bash
pip install tensorflow pandas numpy scikit-learn
jupyter notebook Basic_neural_network.ipynb
```

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn

## 📌 Notes

- The dataset is intentionally tiny (16 samples) — this is a learning project focused on understanding neural network structure, not production-level accuracy.
- Manual min-max scaling is used instead of `sklearn.preprocessing.MinMaxScaler` to demonstrate the underlying math.
- The high validation accuracy (100%) is expected given the small, clean dataset — this would not generalize without more real-world data.

## ⚠️ Disclaimer

Heads up — this neural network was trained on just **16 rows of data**. That's roughly the size of a school attendance register. While the model hits 100% validation accuracy, don't let that number fool you — it's essentially memorizing a handful of examples, not actually "learning" in any meaningful generalizable sense. If you fed it real-world sensor data from an actual garden, it would very likely fall apart.

Think of this project as a working demo of *how* a neural network is wired together — the layers, the training loop, the loss going down epoch by epoch — not as something you'd trust to keep your plants alive. For anything beyond learning purposes, you'd need hundreds or thousands of real labeled samples, proper cross-validation, and a lot more variety in the data.

**TL;DR:** Great for understanding the basics. Not suitable for real-world use as-is.

## 📄 License

Open source — for educational use.
