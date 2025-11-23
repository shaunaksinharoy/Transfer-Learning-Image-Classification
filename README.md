# AgriGuard – Intelligent Plant Disease Detection using Transfer Learning

**Course:** CSA2001 – Fundamentals in AI and ML  
**VIT Bhopal | Submitted by:** Shaunak Sinha Roy | 25BEC10032 | Nov 2025  
**Course Outcome:** CO5 – Case Studies: Transfer Learning from pre-trained models

## Project Overview
AgriGuard is an intelligent web application that detects 38 types of plant diseases from leaf images using **Transfer Learning** (ResNet50 + EfficientNetB4). It provides disease name, confidence score, and remedy suggestions in English & Hindi.

**Live Demo:** Run `streamlit run app.py`  
**Dataset:** PlantVillage (87,900 images, 38 classes)

## Key Features
- Upload leaf image → Instant disease prediction
- 98.5% accuracy using ensemble transfer learning
- Grad-CAM visualization (shows infected area)
- Remedy suggestions from SQLite database
- Supports Tomato, Potato, Pepper, Maize, etc.

## Results
| Model                  | Accuracy | Training Time | Parameters |
|------------------------|----------|---------------|------------|
| From Scratch           | 68.2%    | 6+ hours      | 25M+       |
| Transfer Learning (ResNet50) | 97.1% | 38 mins       | 2.1M       |
| Ensemble (ResNet + EfficientNet) | **98.5%** | 52 mins   | 3.4M       |

## Tech Stack
- Python, TensorFlow 2.10, Keras
- Streamlit (Web UI)
- OpenCV, NumPy, Pandas
- SQLite, Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
