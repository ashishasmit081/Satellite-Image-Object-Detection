# Object Detection in Satellite Images using YOLOv8

## 🚀 Project Overview
This project utilizes **YOLOv8n** for **object detection in satellite images**, specifically using the **DOTA (Dataset for Object Detection in Aerial Images)**. The model is trained to detect multiple objects such as **vehicles, buildings, ships, and roads** in aerial imagery.

## 📌 Dataset
- **Dataset Used:** [DOTA Dataset (Complete)](https://www.kaggle.com/datasets/shadow4ever/dota-dataset-complete-new)
- The dataset consists of annotated aerial images containing multiple object categories.

## 🏗️ Project Workflow
1. **Data Preprocessing**
   - Loaded and preprocessed images from the DOTA dataset.
   - Converted annotation format for YOLOv8 compatibility.
2. **Model Training**
   - Trained **YOLOv8n** for **150 epochs** using the processed dataset.
   - Optimized training hyperparameters to improve detection accuracy.
3. **Evaluation**
   - Assessed model performance using **Mean Average Precision (mAP)** and **Intersection over Union (IoU)**.
4. **Inference & Visualization**
   - Ran inference on test images and visualized detections using OpenCV.

## 🛠️ Technologies Used
- **YOLOv8n** (You Only Look Once v8)
- **Python**
- **OpenCV**
- **PyTorch / TensorFlow**
- **DOTA Dataset**

## 📂 File Structure
```
├── yolov8n-dota-150epochs-final-new.ipynb  # Jupyter Notebook with code
├── dataset/                               # Contains training images & labels
├── models/                                # Saved model weights
├── results/                               # Output images with detected objects
└── README.md                              # Project documentation
```

## 🚀 How to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install ultralytics opencv-python torch torchvision
```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/YOLOv8-DOTA-Object-Detection.git
cd YOLOv8-DOTA-Object-Detection
```

### 3️⃣ Run Jupyter Notebook
```bash
jupyter notebook yolov8n-dota-150epochs-final-new.ipynb
```

## 🎯 Results & Performance
- The model was trained for **150 epochs**, achieving **high accuracy in detecting objects in aerial images**.
- Evaluated using **mAP** and **IoU**, optimizing detection for real-world applications.

## 📌 Future Improvements
- Fine-tune model for **better precision and recall**.
- Implement **real-time deployment** on cloud-based services.
- Train on an **expanded dataset** for improved generalization.

## 🤝 Contributing
Feel free to **fork this repository** and contribute improvements via **pull requests**!

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **Author:** Your Name  
📧 Contact: your.email@example.com  
🔗 LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
