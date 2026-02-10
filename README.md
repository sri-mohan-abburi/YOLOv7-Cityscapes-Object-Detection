# YOLOv7 Object Detection on Cityscapes 🚗

**Author:** Sri Mohan Abburi  
**Course:** Digital Image Processing (Bonus Project)  

---

## 📌 Project Overview
Navigating urban environments requires more than just seeing—it requires understanding. This project implements a real-time object detection pipeline using **YOLOv7** trained on the **Cityscapes Dataset**. 

The goal was to move beyond basic detection and tackle the nuances of urban "clutter," accurately identifying vehicles, pedestrians, and riders in high-density street scenes.

## 🛠️ Key Features & Problem Solving

### 1. Custom Data Engineering
Cityscapes labels aren't ready for YOLO out of the box. I developed a custom pipeline to handle the heavy lifting:
* **Coordinate Transformation:** Built scripts to convert annotations from COCO/VOC formats ($x_{min}, y_{min}, w, h$) to the normalized YOLO format ($x_{center}, y_{center}, w, h$).
* **Automated Prepping:** `data_prep.py` automates the mapping of `gtFine` labels to `leftImg8bit` images and manages the train/val splits.

### 2. Hardware-Aware Training
Training SOTA models usually demands massive compute. To successfully train on an **NVIDIA RTX 3060 (12GB/16GB)**, I optimized the training strategy:
* **Memory Management:** Fine-tuned hyperparameters and utilized a batch size of 1 to ensure convergence without exhausting VRAM.
* **Transfer Learning:** Leveraged pre-trained YOLOv7 weights to reduce training time and improve feature extraction for urban classes.

### 3. Video Processing Pipeline
Urban detection is best viewed in motion. I integrated `MoviePy` and `PyAV` to process raw image sequences into annotated video streams, providing a "driver's eye" view of the model's performance.

---

## 📂 Repository Structure

* `utils/bbox_converter.py`: Core logic for bounding box math and coordinate normalization.
* `utils/data_prep.py`: Automation script for dataset organization and label export.
* `docs/`: Contains the detailed project report and DIP assignment specifications.

---

## 🚀 Getting Started

### Installation
Clone the repo and install the requirements:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt