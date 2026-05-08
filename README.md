# Salient Object Detection — End-to-End CNN Project

A deep learning project that detects and segments the most visually prominent objects in an image using a custom-built CNN encoder-decoder architecture trained from scratch with PyTorch.

---

## Project Overview

Salient Object Detection (SOD) identifies the most visually important regions in an image — the parts that naturally draw human attention. Unlike traditional object detection, SOD outputs a **saliency mask** that highlights the most prominent region in a single-image context.

This project was built as part of **Cohort V — Project #3: End-to-End ML/DL Project**, covering the full pipeline from data loading to model training, evaluation, and demo.

---

## 📁 Project Structure

```
salient-object-detection/
├── data_loader.py          # Dataset loading, preprocessing, and augmentation
├── sod_model.py            # CNN encoder-decoder architecture (built from scratch)
├── train.py                # Training and validation loop with logging
├── evaluate.py             # Evaluation metrics and visualization
├── demo_notebook.ipynb     # Interactive demo — upload image, get saliency mask
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

A custom **CNN Encoder-Decoder** built from scratch in PyTorch:

- **Input:** RGB image (3 × 128 × 128 or 224 × 224)
- **Encoder:** 3–4 Conv2D layers with ReLU activations + MaxPooling
- **Decoder:** 3–4 ConvTranspose2D (upsampling) layers with ReLU activations
- **Output:** 1-channel Sigmoid mask (same size as input)
- **Loss:** Binary Cross-Entropy + 0.5 × (1 − IoU)
- **Optimizer:** Adam (lr = 1e-3)
- **Training:** 15–25 epochs with early stopping

---

## 📦 Dataset

**ECSSD** — Extended Complex Scene Saliency Dataset  
Contains semantically meaningful images with pixel-accurate saliency masks.

**Dataset splits:**
- Train: 70%
- Validation: 15%
- Test: 15%

**Preprocessing:**
- Resize all images to 128×128 or 224×224
- Normalize pixel values to [0, 1]
- Augmentations: horizontal flip, random crop, brightness variation

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/cizdem/salient-object-detection.git
cd salient-object-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Download the ECSSD dataset and place it in a `data/` folder:
```
data/
├── images/
└── masks/
```
Then run:
```bash
python data_loader.py
```

### 4. Train the model
```bash
python train.py
```
Checkpoints are saved automatically after each epoch. Training can be resumed from the last checkpoint if interrupted.

### 5. Evaluate
```bash
python evaluate.py
```

### 6. Run the demo
Open `demo_notebook.ipynb` in Jupyter and upload any image to get a predicted saliency mask overlaid on it.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| IoU (Intersection over Union) | Overlap between predicted and ground-truth mask |
| Precision | Correctly predicted salient pixels |
| Recall | Coverage of ground-truth salient regions |
| F1-Score | Harmonic mean of Precision and Recall |
| MAE (optional) | Mean Absolute Error on mask values |

---

## 🖼️ Demo Output

The demo notebook (`demo_notebook.ipynb`) accepts any input image and outputs:
- Input image
- Ground-truth saliency mask
- Predicted saliency mask
- Overlay (predicted + input)
- Inference time per image

---

## 🛠️ Tools & Environment

| Tool | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | Latest |
| NumPy | Latest |
| OpenCV | Latest |
| Matplotlib | Latest |
| scikit-learn | Latest |
| tqdm | Latest |

**Recommended environment:** Google Colab, Kaggle Notebooks, or local GPU setup.

---

## ⚙️ Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📝 License

This project was developed as part of an academic ML/DL cohort program. For educational use only.
