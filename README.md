# 🎥 Deepfake Detection with CNNs

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-FF0000?logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![CNN Model](https://img.shields.io/badge/Model-CNN-teal?logo=neural_network&logoColor=white)](https://en.wikipedia.org/wiki/Convolutional_neural_network)  

---

## 🤖 Project Overview  
This project implements a **Convolutional Neural Network (CNN)**-based system to detect deepfake images/videos. It focuses on distinguishing manipulated media from genuine content, leveraging modern image processing, CNN architectures and forensic-AI methods.

---

## ⚙️ Key Features  
- ✅ CNN-based architecture trained for binary classification (Real vs. Fake)  
- ✅ Preprocessing pipeline: face-detection, frame extraction (for videos) / image input  
- ✅ Support for image and video datasets – scalable for research use  
- ✅ Visualization of detection results: heatmaps or activation maps highlighting manipulated regions  
- ✅ Extensible for stronger architectures (EfficientNet, ResNet, Vision Transformers) or multi-modal input  

---

## 🧩 Architecture  
```text
Input (image/frame)  
  → Face / Region Extraction  
  → Convolutional Neural Network (feature extraction)
```

## 📊 Tech Stack
| Category     | Tools                                |
|--------------|--------------------------------------|
| Language     | Python 3.x                           |
| Framework    | PyTorch                              |
| Model Type   | CNN (Image/Frame Classification)     |
| Task Domain  | Deepfake Detection · Media Forensics |

## 🔧 Installation & Setup
Clone the repository:
git clone https://github.com/NirEllor/Deepfake-Detection-with-CNNs.git
cd Deepfake-Detection-with-CNNs

(Optional) create virtual environment:
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run:
Example: train the detection model:
python train.py --epochs 30 --batch_size 64

Example: evaluate on test set:
python evaluate.py --model_path models/checkpoint.pth --data_dir data/test

## 📥 Dataset

Use an image/video dataset containing “real” and “fake” media (e.g., face-swapped or manipulated).
Ensure preprocessing: face/region cropping, normalization, frame extraction.

## 📋 Expected Output

Trained model achieving high classification accuracy (Real/Fake)

Confidence scores + visualization of activated features (via heatmaps)

## Example outputs:
Input frame: “fake_face_01.jpg” → Predicted: FAKE (0.93 confidence)
Input frame: “real_face_12.jpg” → Predicted: REAL (0.89 confidence)

## 👨‍💻 Author
Nir Ellor
Full-Stack Web3 & AI Developer
  → Fully-connected layers (classification)  
  → Output: Real / Fake + Confidence Score  
