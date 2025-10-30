# 🛢️ AI SpillGuard: Oil Spill Detection Using Deep Learning

**Intern:** Aditya Kumar Upadhyay &nbsp; | &nbsp; **Organization:** Infosys Springboard &nbsp; | &nbsp; **Duration:** 8 Weeks &nbsp; | &nbsp; **Mentor:** Eekshitha Namala

---

## 📘 Project Overview

AI SpillGuard is an AI-powered deep learning system that detects and segments oil spills in satellite images. Using a **U-Net convolutional neural network (CNN)**, the model identifies spill regions with high accuracy, enabling faster environmental response and analysis.

This project integrates the complete ML pipeline from **data collection** to **model deployment via Streamlit**.

---

## 🎯 Objectives

* Develop a U-Net segmentation model for oil spill detection.
* Preprocess and augment satellite imagery data.
* Train and evaluate using Dice and IoU metrics.
* Visualize segmentation results with overlays.
* Deploy a real-time prediction web app using Streamlit.

---

## 📂 Folder Structure

```
AI_SpillGuard/
│
├── dataset/                     # Train/Val/Test images and masks
├── notebooks/
│   ├── Week1_2_DataPrep.ipynb
│   ├── Week3_4_ModelTraining.ipynb
│   ├── Week5_6_Visualization.ipynb
│   └── Week7_8_Deployment.ipynb
│
├── model/
│   ├── unet_oilspill_final.h5
│   └── unet_oilspill_best.h5
│
├── app/
│   └── app.py                   # Streamlit app for real-time detection
│
├── outputs/
│   ├── metrics.csv
│   ├── overlay_examples/
│   └── training_curves/
│
└── README.md
```

---

## ⚙️ Installation

Install all required libraries:

```bash
pip install tensorflow keras opencv-python matplotlib streamlit pyngrok
```

---

## ▶️ Usage Instructions

### **1️⃣ Load Dataset in Google Colab**

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q "/content/drive/MyDrive/Oil_Spill_Project/dataset.zip" -d /content/oil_spill_data
```

### **2️⃣ Train the Model**

```bash
!python notebooks/Week3_4_ModelTraining.ipynb
```

### **3️⃣ Run the Streamlit App**

```bash
!streamlit run app/app.py & npx localtunnel --port 8501
```

---

## 🧩 System Architecture

```
        ┌──────────────────────────────┐
        │  Satellite Image Input       │
        └──────────────┬───────────────┘
                       │
                 Preprocessing
                       │
               ┌───────▼────────┐
               │   U-Net Model  │
               └───────┬────────┘
                       │
             Mask Prediction Output
                       │
         Visualization (Overlay + Metrics)
                       │
              Streamlit Web Dashboard
```

---

## 🧠 Model Details

| Layer Type               | Filters | Activation | Purpose                       |
| ------------------------ | ------- | ---------- | ----------------------------- |
| Conv2D + MaxPooling      | 16–64   | ReLU       | Feature extraction (Encoder)  |
| Bottleneck               | 128     | ReLU       | Deep spatial feature learning |
| Conv2DTranspose + Concat | 64–16   | ReLU       | Upsampling & reconstruction   |
| Output (1×1 Conv)        | 1       | Sigmoid    | Binary segmentation mask      |

---

## ⚙️ Training Configuration

| Parameter     | Value                              |
| :------------ | :--------------------------------- |
| Optimizer     | Adam                               |
| Loss Function | Binary Cross-Entropy               |
| Metrics       | Dice Coefficient, IoU, Accuracy    |
| Epochs        | 15                                 |
| Batch Size    | 8                                  |
| Callbacks     | ModelCheckpoint, ReduceLROnPlateau |

---

## 📊 Evaluation Metrics

| Metric                            | Description                                                            |
| :-------------------------------- | :--------------------------------------------------------------------- |
| **Dice Coefficient**              | Measures overlap between predicted and actual mask (higher is better). |
| **IoU (Intersection over Union)** | Evaluates intersection vs union of predicted and actual regions.       |
| **Accuracy**                      | Percentage of correctly classified pixels.                             |

---

## 🧾 Results Summary

| Metric           |   Value  |
| :--------------- | :------: |
| Dice Coefficient | **0.91** |
| IoU              | **0.86** |
| Accuracy         | **0.94** |

**Observations:**

* Loss decreased steadily.
* Dice and IoU improved across epochs.
* Model accurately identifies oil spill regions with minimal false positives.

---

## 🌈 Visualization

**Overlay Examples:**

* Red = Predicted Spill
* Blue = Background
* Transparency (α = 0.5) = both visible clearly.

Shows real-time segmentation comparison:

1. Original Satellite Image
2. Ground Truth Mask
3. Predicted Overlay Mask

---

## 🌐 Streamlit Web App

**Features:**

* Upload satellite image.
* Model predicts oil spill region.
* Overlay displayed instantly.
* Hosted with ngrok tunnel for Colab.

---

## 🧩 Future Improvements

* Integrate **multispectral satellite data**.
* Implement **real-time ocean monitoring**.
* Improve accuracy with **ResU-Net / Attention U-Net**.

---

## 🧑‍💼 Intern Details

**Name:** Aditya Kumar Upadhyay
**Organization:** Infosys Springboard
**Project Duration:** 8 Weeks
**Role:** Deep Learning Intern
**Project Title:** AI SpillGuard – Oil Spill Detection Using Deep Learning

---

## 📚 References

* Ronneberger et al. (2015) – *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
* Kaggle Oil Spill Dataset.
* TensorFlow and Keras Documentation.
* OpenCV & Matplotlib Libraries.

---

## ✅ Final Deliverables

| Deliverable               | Description                                        | Status      |
| :------------------------ | :------------------------------------------------- | :---------- |
| Notebook (Weeks 1–6)      | Full pipeline for preprocessing and model training | ✅ Completed |
| Streamlit App (Weeks 7–8) | Web dashboard for real-time predictions            | ✅ Completed |
| Final Report              | Detailed technical documentation                   | ✅ Completed |
| Presentation Slides       | Summary slides for academic presentation           | ✅ Completed |
| README + Docs             | Comprehensive project guide                        | ✅ Completed |

---

**© 2025 – Aditya Kumar Upadhyay | Infosys Springboard Internship**
