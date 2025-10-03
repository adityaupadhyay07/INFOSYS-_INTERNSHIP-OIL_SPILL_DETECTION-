# INFOSYS-INTERNSHIP-OIL_SPILL_DETECTION-

🌊 Oil Spill Image Segmentation

This project focuses on detecting and segmenting oil spills in satellite images using Deep Learning (U-Net). The workflow covers data preprocessing, visualization, model building, training, and evaluation.

📂 Project Structure
.
├── week1_2.ipynb   # Dataset extraction, preprocessing, and visualization
├── week3_4.ipynb   # U-Net model implementation and training
├── dataset.zip      # (Not included here; mounted from Google Drive in Colab)
└── README.md        # Project documentation

🚀 Features

> Dataset Handling

Extracts dataset from Google Drive

Organizes train/validation images and masks

Counts and visualizes dataset structure

> Preprocessing

Image resizing and normalization

Speckle noise reduction

Visualization of images and masks

> Model

U-Net architecture for segmentation

Metrics: Dice Coefficient, Intersection-over-Union (IoU), Accuracy

Binary cross-entropy loss with Adam optimizer

> Training

Training with checkpoints

Validation monitoring

Visual performance plots

🛠️ Installation & Requirements

This project is designed to run in Google Colab.

Dependencies:

numpy
opencv-python
matplotlib
tensorflow

To install locally:

pip install numpy opencv-python matplotlib tensorflow

▶️ How to Run

1.Open the notebooks in Google Colab.

2.Mount your Google Drive to access the dataset:

from google.colab import drive 

drive.mount('/content/drive')

3.Extract the dataset:

!unzip "/content/drive/MyDrive/Oil_Spill_Project/dataset.zip" -d "oil_spill_data"

4.Run all cells in week1_2.ipynb (for preprocessing & visualization).
5.Run all cells in week3_4.ipynb (for U-Net model training).

📊 Results
Sample Input vs Predicted Segmentation
| Input Image                         | Ground Truth Mask                 | Predicted Mask                         |
| ----------------------------------- | --------------------------------- | -------------------------------------- |
| ![Input](results/sample1_input.png) | ![Mask](results/sample1_mask.png) | ![Predicted](results/sample1_pred.png) |
| ![Input](results/sample2_input.png) | ![Mask](results/sample2_mask.png) | ![Predicted](results/sample2_pred.png) |


🔮 Future Work

Hyperparameter tuning for improved accuracy

Experiment with other segmentation models (e.g., DeepLab, SegNet)

Deploy model via web app for real-time predictions

👨‍💻 Author:

Aditya Kuamr Upadhyay
