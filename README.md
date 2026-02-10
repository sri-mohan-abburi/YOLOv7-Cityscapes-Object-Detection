# YOLOv7 Object Detection on Cityscapes 🚗

**Author:** Sri Mohan Abburi
**Course:** Digital Image Processing (Bonus Project)

## 📌 Project Overview
This project implements a real-time object detection pipeline for urban street scenes using **YOLOv7** trained on the **Cityscapes Dataset**. It addresses the challenge of detecting dynamic objects (vehicles, pedestrians, riders) in complex urban environments.



## 🛠️ Key Features
* **Custom Data Pipeline:** Engineered Python scripts to convert Cityscapes annotations to YOLO format and transform bounding box coordinates (COCO -> VOC -> YOLO).
* **Hardware Optimization:** Fine-tuned training parameters (Batch Size=1) to train successfully on constrained hardware (**NVIDIA RTX 3060 16GB**).
* **Video Inference:** Integrated `MoviePy` and `PyAV` to process raw image sequences into video streams for detection.

## 📂 Repository Structure
* `utils/bbox_converter.py`: Handles coordinate transformations (x_min, y_min, width, height -> x_center, y_center, w, h).
* `utils/data_prep.py`: Automates train/val split generation and label export.
* `docs/`: Detailed project report and assignment specifications.

## 📊 Methodology
1.  **Data Preparation:** Downloaded Cityscapes packages (`leftImg8bit`, `gtFine`) and normalized labels.
2.  **Training:** Transferred learning from pre-trained YOLOv7 weights.
3.  **Inference:** Ran detection on `leftImg8bit_demoVideo` sequences.

## 🚀 Usage
```bash
# Example: Convert Dataset Formats
python utils/data_prep.py --source-folder-path ./cityscapes/images