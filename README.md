# 👁️ RetinaLens

**Retinal fundus image analysis for diabetic retinopathy (DR) grading, with explainable AI and LLM-generated explanations.**

## ✨ Features

- **Five-class DR grading:** No DR, Mild, Moderate, Severe, Proliferative DR
- **Transfer learning:** ImageNet-pretrained EfficientNet-B0 fine-tuned on retinal images
- **Official data splits:** trains on `train`, selects the best checkpoint on `valid`, and reports final metrics once on the untouched `test` split
- **Class-imbalance handling:** weighted cross-entropy loss
- **Explainability:** Grad-CAM overlay highlighting regions that influenced the prediction
- **LLM explanations:** Claude explains the model output in plain language, without claiming to inspect the image itself
- **Evaluation dashboard:** accuracy, macro precision/recall/F1 and per-class metrics shown in the app

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 / 3.11 |
| Deep learning | PyTorch, torchvision (EfficientNet-B0) |
| Data handling | pandas, Pillow, NumPy |
| Evaluation | scikit-learn (accuracy, precision, recall, F1, confusion matrix) |
| Explainability | Grad-CAM (custom implementation with forward/backward hooks), OpenCV |
| Web interface | Streamlit |
| LLM explanations | Anthropic Claude API (`anthropic` SDK) |
| Config | python-dotenv |

---

## 🏗️ How It Works

```
Fundus image
     │
     ▼
Resize 224×224 + ImageNet normalisation
     │
     ▼
EfficientNet-B0 (fine-tuned, 5-class head)
     │
     ├──► Softmax class scores ──► Predicted grade
     │
     ├──► Grad-CAM heatmap (last conv block)
     │
     └──► Claude API ──► Plain-language explanation of the model output
```

**Training pipeline**
1. Read labels from each split's `annotations.csv` (`Image name`, `Retinopathy grade`)
2. Train on the official train split with augmentation (flip, rotation, brightness/contrast jitter)
3. AdamW optimiser with ReduceLROnPlateau; best checkpoint chosen by validation accuracy
4. One final evaluation of that checkpoint on the official test split → `models/metrics.json`

---

## 📂 Project Structure

```
RetinaLens/
├── app.py              # Streamlit app: prediction, Grad-CAM, Claude explanation, metrics
├── train.py            # Training + evaluation using the official splits
├── requirements.txt
├── .env.example        # Template for your Anthropic API key
├── models/
│   └── metrics.json    # Held-out test metrics (weights are git-ignored)
├── notebooks/
└── dataset/            # Not included; see below
```

---

## 📊 Dataset

The expected layout is:

```
dataset/Datasets/Diabetic Retinopathy/
├── train/  annotations.csv  + images
├── valid/  annotations.csv  + images
└── test/   annotations.csv  + images
```

Each `annotations.csv` contains `Image name`, `Retinopathy grade` (0–4), `Risk of macular edema` and `Caption`. Only the grade is used as the target.

| Split | Images |
|---|---|
| Train | 1,577 |
| Valid | 339 |
| Test | 338 |


## 📈 Results

Evaluated once on the official held-out test split (338 images; EfficientNet-B0, 15 epochs, 224×224, CPU training):

| Metric | Score |
|---|---|
| Accuracy | 65.4% |
| Macro precision | 60.4% |
| Macro recall | 58.3% |
| Macro F1 | 58.4% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 – No DR | 0.80 | 0.76 | 0.78 | 178 |
| 1 – Mild | 0.28 | 0.48 | 0.36 | 44 |
| 2 – Moderate | 0.72 | 0.55 | 0.63 | 76 |
| 3 – Severe | 0.68 | 0.52 | 0.59 | 25 |
| 4 – Proliferative DR | 0.53 | 0.60 | 0.56 | 15 |

Macro F1 is the more honest headline number given the class imbalance. Mild DR is the hardest grade for the model, as it is subtle even for human graders. Per-class figures for Severe and Proliferative are based on very few test images and are noisy.

---

## ⚠️ Limitations

- Image-level splits from the dataset provider; patient identifiers are unavailable, so patient leakage and duplicate images cannot be ruled out
- Small dataset and strong class imbalance
- 224×224 input loses fine lesion detail; higher resolution is likely to help
- No external validation on an independent dataset
- Softmax scores are uncalibrated
- Grad-CAM shows influential regions for the model, not proof of pathology
- Claude receives only the predicted class and scores, never the image, and does not diagnose

## 🔭 Future Work

Higher-resolution training, stronger augmentation, probability calibration, image-quality checks, patient-level splitting, external validation, and use of the macular-edema label for multi-task learning.

---

## 👩‍💻 Author

**Akanksha Rayabharam**

## 📄 Disclaimer

For educational and research purposes only. Not for clinical use.
