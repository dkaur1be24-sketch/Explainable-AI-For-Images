# 🔍 Explainable AI (XAI) Pipeline with Evaluation & Prototype Reasoning

---

## 📌 Project Overview

This project focuses on building a **comprehensive Explainable AI (XAI) pipeline** for image classification models.
Instead of relying on a single explanation method, this system integrates **multiple XAI techniques**, evaluates their reliability, and enhances interpretability using **prototype-based reasoning**.

The goal is not just to *visualize explanations*, but to **measure their correctness and trustworthiness**.

---

## ❗ Problem Statement

Deep learning models (especially CNNs) achieve high accuracy but act as **black boxes**.
This lack of transparency creates challenges in:

* Trusting model decisions
* Debugging incorrect predictions
* Deploying models in sensitive domains

This project addresses the question:

> *“Can we trust model explanations, and how do we validate them?”*

---

## 🧠 Methodology / Approach

The system follows a structured pipeline:

```
Input Image 
   → Preprocessing 
   → CNN Model (Feature Extraction + Prediction) 
   → XAI Methods (GradCAM, LIME, SHAP) 
   → Evaluation Metrics (Faithfulness + Sanity Checks) 
   → Prototype Reasoning (Nearest Neighbor Explanation)
   → Output Visualization
```

---

## 🤖 Model Details

* **Architecture**: Convolutional Neural Network (ResNet-based)
* **Framework**: PyTorch
* **Input Size**: 224 × 224 RGB images
* **Feature Dimension**: 512-d embedding space
* **Dataset**: CIFAR-10

---

## 🔍 Explainability Methods Used

### 1. GradCAM

* Produces heatmaps highlighting important regions
* Uses gradient flow in convolution layers

### 2. LIME (Local Interpretable Model-Agnostic Explanations)

* Perturbs input image
* Learns local surrogate model

### 3. SHAP (SHapley Additive exPlanations)

* Based on game theory
* Assigns importance values to image regions

---

## 📊 Evaluation Metrics (Core Strength of Project)

This project goes beyond visualization by **quantitatively evaluating explanations**:

### 🔻 Deletion AUC

* Measures how quickly prediction confidence drops when important pixels are removed
* Lower is better

### 🔺 Insertion AUC

* Measures how quickly prediction confidence increases when important pixels are added
* Higher is better

### 🔄 Sanity Check

* Tests whether explanations change when model weights are randomized
* Ensures explanations are **model-dependent (not random)**

---

## 🧩 Prototype-Based Reasoning (Unique Feature)

A key innovation of this project is **case-based explanation**:

* Extract feature embeddings from the model
* Use **cosine similarity + nearest neighbors**
* Retrieve most similar training images

### Why this matters:

> “The model predicted this image as a *cat* because it looks similar to these training examples.”

### Additional Metric:

* **Prototype Purity**:
  Percentage of nearest neighbors belonging to the same class

---

## 📈 Feature Space Visualization

* PCA and t-SNE used to visualize embeddings
* Helps understand:

  * Class clustering
  * Feature separability
  * Model representation quality

---

## 📊 Results

The project produces:

* GradCAM heatmaps
* LIME explanation overlays
* SHAP value visualizations
* Deletion/Insertion metric graphs
* Prototype nearest neighbor results
* Feature space plots

---

## 🎥 Demo Video

👉 *(Add your Google Drive / YouTube link here)*

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Project

```
python main.py
```

---

## 📁 Project Structure

```
project/
│
├── main.py
├── config.py
├── preprocessing.py
├── inference.py
├── utils.py
│
├── xai/
│   ├── gradcam.py
│   ├── lime.py
│   ├── shap.py
│
├── evaluation/
│   ├── metrics.py
│   ├── sanity_checks.py
│
├── prototype/
│   ├── prototype.py
│
├── models/
├── results/
└── requirements.txt
```

---

## 🚀 Key Highlights

✔ Multi-method XAI comparison
✔ Quantitative evaluation of explanations
✔ Sanity checks for reliability
✔ Prototype-based reasoning system
✔ Feature space visualization

---

## 🔮 Future Improvements

* Add real-world dataset (e.g., medical / environmental images)
* Build web interface (Streamlit / Flask)
* Extend to multi-class and multi-object scenarios
* Improve t-SNE visualization with test integration

---

## 👩‍💻 Author

**Diljeet Kaur**

---

## ⭐ Final Note

This project is not just about generating explanations —
it is about **understanding, validating, and trusting AI decisions**.
