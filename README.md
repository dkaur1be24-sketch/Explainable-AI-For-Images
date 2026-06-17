# 🔍 Explainable AI Dashboard for Image Classification (CIFAR-10)

---

**Explainable AI Dashboard for Deep Learning Models using Grad-CAM, LIME, and SHAP**

---

## 2. Problem Statement

Deep learning models, especially Convolutional Neural Networks (CNNs), achieve high accuracy in image classification tasks but act as **black boxes**. This lack of transparency makes it difficult to understand *why* a model makes a particular decision.

This project addresses this issue by:

* Providing **visual explanations** for model predictions
* Comparing multiple Explainable AI (XAI) techniques
* Evaluating explanation quality using quantitative metrics

👉 Goal: Build a system that explains, evaluates, and visualizes model decisions in a clear and interpretable way.

---

## 3. Role of Edge Computing

This project is designed to be deployable on **edge devices (e.g., Jetson Nano)**.

### Why Edge Computing?

* **Low Latency** → Real-time predictions and explanations
* **Offline Capability** → No dependency on cloud services
* **Efficiency** → Reduced data transfer and cost

### What runs on edge:

* Trained ResNet18 model
* XAI techniques (Grad-CAM, LIME, SHAP)
* Dashboard visualization generation

---

## 4. Methodology / Approach

### 🔁 Overall Pipeline

```
Input Image → Preprocessing → Model Prediction → XAI Explanations →
Evaluation Metrics → Prototype Matching → Dashboard Visualization
```

### 🧩 Steps Explained:

1. **Input**: Images from CIFAR-10 dataset
2. **Preprocessing**:

   * Normalization for model input
   * Raw images for visualization
3. **Model**:

   * ResNet18 predicts class probabilities
4. **Explainability**:

   * Grad-CAM → attention heatmaps
   * LIME → local region importance
   * SHAP → pixel-level contribution
5. **Evaluation**:

   * Deletion & Insertion curves
   * AUC scores
6. **Prototype Matching**:

   * Finds similar training images using feature embeddings
7. **Dashboard Generation**:

   * Combines all outputs into a visual panel

---

## 5. Model Details

* **Model**: ResNet18 (CNN architecture)
* **Dataset**: CIFAR-10
* **Input Size**: 32 × 32 RGB images
* **Framework**: PyTorch
* **Feature Extraction**:

  * Final layer removed to extract embeddings
* **Special Handling**:

  * Modified ResNet blocks to fix SHAP compatibility (removed inplace operations)

---

## 6. Training Details

* **Dataset**: CIFAR-10 (10 classes)
* **Training Process**:

  * Standard supervised learning
  * Cross-entropy loss
* **Evaluation**:

  * Accuracy and loss tracking
* **Output**:

  * Trained model saved as `.pth` file

---

## 7. Results / Output

### 📊 Model Output

* Predicted class with confidence score

### 🧠 Explainability Outputs

* **Grad-CAM**: Heatmaps showing important regions
* **LIME**: Segmented region importance
* **SHAP**: Pixel-level contributions

### 📉 Evaluation Metrics

* **Deletion Curve** ↓ → Faster drop = better explanation
* **Insertion Curve** ↑ → Faster rise = better explanation
* **AUC Scores** → Quantitative comparison of XAI methods

### 🧬 Prototype Analysis

* Displays top-5 similar images from training dataset

### 🖼️ Dashboard Output

Each dashboard includes:

* Original Image
* Grad-CAM, LIME, SHAP overlays
* Prediction + Confidence
* Deletion & Insertion curves
* Prototype images

---

## 8. Dashboard (Core Feature)

The `dashboard.py` file builds a **complete Explainable AI visualization system**.

### What it does:

* Runs predictions on test images
* Generates explanations using:

  * Grad-CAM
  * LIME
  * SHAP
* Evaluates explanation quality
* Finds similar images (prototypes)
* Creates a **final dashboard image per class**

### Output:

* High-quality visual panels saved as `.png` files

👉 This transforms raw model outputs into **interpretable insights**

---

## 9. Setup Instructions

### 🔧 Requirements

* Python 3.x
* PyTorch
* torchvision
* numpy
* matplotlib
* scikit-learn
* lime
* shap

---

### ⚙️ Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn lime shap
```

---

### ▶️ Run the Project

```bash
python prototype.py
python dashboard.py
```

---

### 📂 Outputs Generated

* Explanation visualizations
* Evaluation graphs
* Dashboard images

---

## 10. Key Highlights

✔ Combines **three XAI methods** (Grad-CAM, LIME, SHAP)
✔ Provides **quantitative evaluation (AUC, curves)**
✔ Includes **prototype-based interpretability**
✔ Fixes **SHAP-PyTorch compatibility issue**
✔ Generates **complete visual dashboards**
✔ Ready for **edge deployment**

---

## 11. Future Improvements

* Real-time dashboard UI (web app)
* Support for larger datasets (ImageNet)
* Model comparison (multiple architectures)
* Optimization using TensorRT for edge devices

---

## 12. Conclusion

This project bridges the gap between **model accuracy and interpretability** by combining multiple XAI techniques into a unified dashboard. It not only explains predictions but also evaluates and visualizes them, making deep learning models more transparent and trustworthy.

---
