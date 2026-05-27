# 🐦 Backyard Bird Classifier

A Streamlit-based computer vision project that classifies backyard bird species using transfer learning (EfficientNetB0) and tests robustness under real-world image conditions such as blur, noise, lighting changes, and compression.

---

## 🚀 Demo Features

- 🧠 Transfer learning with EfficientNetB0 (ImageNet pretrained)
- 🐦 Classifies 10 common backyard bird species
- 🎨 Real-world image augmentations:
  - Blur
  - Low light
  - Sensor noise
  - Random crop
  - JPEG compression
- 📊 Visualizes prediction confidence changes under different conditions
- 🖼️ Interactive Streamlit web app

---

## 🧠 Model Overview

- **Architecture:** EfficientNetB0
- **Input size:** 224 × 224 RGB images
- **Training type:** Fine-tuned on bird species dataset
- **Output:** 10-class softmax probability distribution

---

## 📦 Dataset

This project uses the **200 Bird Species Dataset (CUB-200 extension)**:

Download: https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images

- 11,788 images total
- 200 bird species
- Images organized in class-based folders

Only a subset of 10 species is used for training in this project.

---

## 🐦 Selected Classes (10)

- American Robin  
- American Goldfinch  
- Dark-eyed Junco  
- Blue Jay  
- Black-capped Chickadee  
- Northern Cardinal  
- House Sparrow  
- Downy Woodpecker  
- Mourning Dove  
- White-breasted Nuthatch  

---

## ⚙️ Installation

git clone https://github.com/ecooley/bird-classifier
cd bird-classifier

pip install -r requirements.txt

---

## ▶️ Run the App

streamlit run app.py

---

## 🏋️ Training (optional)

To retrain the model:

python train.py

Model weights will be saved to:
weights/bird_model.keras
