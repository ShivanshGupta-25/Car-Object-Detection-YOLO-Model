
# 🚗 AI-Powered Car Parts Detection & Instance Segmentation (YOLOv8)

This repository presents a **fully end-to-end AI system** for **car parts object detection and instance segmentation** using **YOLOv8**.  
The project also includes a **fully working Streamlit web application** that allows users to upload images or videos and run real-time inference using the trained model.

The trained model is exportable to **ONNX**, making it suitable for **cloud, edge, and production deployment**.

---

## 📌 Project Overview

- **Task**: Car parts detection & segmentation
- **Model**: YOLOv8 Segmentation (`yolov8*-seg`)
- **Framework**: Ultralytics YOLOv8 (PyTorch)
- **Web App**: Streamlit
- **Deployment Formats**: PyTorch, ONNX
- **Supported Inputs**: Images, Videos, Webcam

---

## 🎯 Problem Statement

Automotive inspection systems require precise identification of car components.  
Traditional bounding-box detection is insufficient for complex shapes such as bumpers, doors, and windshields.

This project solves that by:
- Performing **pixel-level instance segmentation**
- Supporting **real-time inference**
- Providing a **user-friendly web interface** via Streamlit

---

## 🧱 System Architecture

Raw Car Images / Video  
↓  
Image Preprocessing (Resize + Normalization)  
↓  
YOLOv8 Backbone (CSPDarknet)  
↓  
FPN + PAN Feature Aggregation  
↓  
Segmentation Head (Boxes + Masks)  
↓  
Non-Max Suppression  
↓  
Visualized Car Parts Output  

---

## ✅ Key Features

- Single YOLOv8 segmentation model
- Multi-class car parts detection
- Pixel-accurate masks
- Streamlit-based interactive UI
- Google Colab & local system support
- Edge & cloud deployment ready
- ONNX export for production inference

---

## 🧠 Model Design

- **Model Type**: YOLOv8 Instance Segmentation
- **Loss Functions**:
  - Box Loss
  - Segmentation Loss
  - Classification Loss
  - Distribution Focal Loss (DFL)
- **Annotation Format**: Polygon-based YOLO segmentation labels

---

## 📊 Model Performance (Validation)

```
Box mAP@0.5        : ~0.70+
Mask mAP@0.5       : ~0.65+
Precision          : High
Recall             : High
Inference Speed    : Real-time on GPU
```

*(Metrics vary based on dataset size and model variant)*

---

## 📂 Repository Structure

```
├── Dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│
├── runs/
│   └── segment/
│
├── Images/
├── Predicted/
│
├── Stream_YoloCarDetect.py
├── TestingModel.ipynb
├── YoloCarDetection.ipynb
│
├── data.yaml
├── requirements.txt
└── README.md
```

---

## 🏷️ Annotation Format (YOLO Segmentation)

```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

- Coordinates normalized between **0 and 1**
- Supports complex object shapes

---

## ⚙️ Installation & Setup

### Clone Repository

```bash
git clone https://github.com/your-username/yolo-car-parts-segmentation.git
cd yolo-car-parts-segmentation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Verify YOLO:
```bash
yolo version
```

---

## 🏋️ Training the Model

```python
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

model.train(
    task="segment",
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    project="runs",
    name="car_parts_seg"
)
```

---

## 🧪 Validation

```python
model.val(data="data.yaml", split="valid")
```

---

## 🔍 Inference (Local)

```python
model.predict(
    source="CarPartsYoloF/test/images",
    conf=0.25,
    iou=0.5,
    save=True
)
```

---

## 🌐 Streamlit Web Application (FULLY WORKING)

### 🔹 Features
- Upload **images or videos**
- Run real-time YOLOv8 segmentation
- Display masks, bounding boxes, and class labels
- Download processed results

### 🔹 Run Streamlit App Locally

```bash
cd streamlit_app
streamlit run app.py
```

Open browser at:
```
http://localhost:8501
```

### 🔹 Streamlit App Workflow

User Upload (Image / Video)  
↓  
YOLOv8 Model Inference  
↓  
Visualization of Masks & Boxes  
↓  
Download Output  

---

## 🚀 Model Export (ONNX)

```bash
yolo export model=runs/segment/car_parts_seg/weights/best.pt format=onnx
```

### Supported Formats
- ONNX
- TensorRT
- OpenVINO
- RKNN
- TFLite

---

## 🧪 Deployment Readiness

- Single trained model
- CPU & GPU compatible
- Streamlit-based UI
- ONNX for production inference
- No Python dependency after export

---

## 🚗 Use Cases

- Automated vehicle inspection
- Car damage analysis
- Insurance claim automation
- Smart garages
- Autonomous vehicle perception

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

👤 **Author**: Shivansh Gupta  
📧 **Email**: shivanshgupta2505@gmail.com  
💼 **LinkedIn**: https://www.linkedin.com/in/shivansh-gupta-b3b75b210/

---

## 💖 Support & Contributions

⭐ Star this repository if you find it useful  
💬 Issues and pull requests are welcome  

> *Keep Learning. Keep Building. Keep Driving AI Forward!* 🚀

---

## 🏁 Future Enhancements

- Car damage & dent segmentation
- Real-time video analytics
- Edge device optimization
- Performance dashboards
- Cloud-hosted Streamlit deployment
