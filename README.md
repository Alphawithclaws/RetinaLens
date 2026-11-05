# RetinaLens

A small research prototype for five-class retinal fundus image classification, model evaluation, Grad-CAM visualisation, and Claude-generated explanations.

> **Safety:** This is an educational/research prototype, not a medical device. It must not be used for diagnosis, screening, treatment, or clinical decisions. Model softmax scores are not calibrated clinical probabilities.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- VS Code + PowerShell
- Internet connection for the first pretrained-weight download
- Optional NVIDIA GPU with a compatible PyTorch CUDA setup
- Optional Anthropic API key for Claude explanations

## 2. Dataset layout

The training script currently expects an **ImageFolder** layout: one folder per class, with images inside each folder.

Example:

```text
RetinaLens/
  dataset/
    0/
      image_001.png
      image_002.png
    1/
      image_003.png
    2/
    3/
    4/
```

Descriptive names are also fine, for example:

```text
dataset/
  No_DR/
  Mild/
  Moderate/
  Severe/
  Proliferative_DR/
```

The script uses the folder names as the class labels and sorts them alphabetically. Check the printed class order before interpreting results.

If the Kaggle download instead contains a CSV plus one shared images folder, this version needs a small dataset-loader adaptation; do not just point `--data-dir` at the raw folder and expect it to work.

**Important:** If patient IDs are available, split by patient rather than randomly by image to avoid patient leakage. This starter script makes a stratified image-level split and should be upgraded to patient-level splitting for a serious final-year evaluation.

## 3. Install and train (PowerShell)

From the RetinaLens project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put the extracted class folders inside `dataset/`, then run:

```powershell
python train.py --data-dir dataset --epochs 5 --batch-size 8
```

For a quick smoke test, use `--epochs 1`. Five epochs are only a starting point; choose training duration based on validation performance and compute.

The first run downloads pretrained EfficientNet-B0 weights. The trained model and held-out metrics are saved to:

- `models/retinal_efficientnet.pth`
- `models/metrics.json`

The test set is held out from training and validation. Do not tune the model repeatedly against the test set.

## 4. Configure Claude

Create a `.env` file in the project root by copying `.env.example`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set your key:

```env
ANTHROPIC_API_KEY=your_real_key
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

Use a Claude model name currently enabled for your Anthropic account if that example model is unavailable. Never commit `.env` or expose the key in frontend code.

Claude receives only the predicted class and model scores in this prototype. It does not independently diagnose the image.

## 5. Launch the app

From the project root:

```powershell
streamlit run app.py
```

Upload a JPG/PNG fundus image. The dashboard displays class scores, Grad-CAM, Claude explanation (if configured), and held-out evaluation metrics.

## 6. What the percentages mean

- **Test accuracy:** fraction of held-out images classified correctly.
- **Macro precision/recall/F1:** class-wise metric calculated per class and averaged equally.
- **Per-class metrics:** precision, recall and F1 for each folder/class.
- **Softmax score:** relative model score across the five classes. It is not automatically calibrated and is not the probability that a patient has a disease.

Do not claim a target accuracy in advance. Report the actual held-out results, dataset size, class distribution, split method, model settings, and limitations.

## 7. Current limitations / next research improvements

1. Confirm the dataset's label definitions and class order against its source documentation.
2. Check duplicate images and patient identifiers; use patient-level splits where possible.
3. Address class imbalance and report per-class metrics, not accuracy alone.
4. Tune augmentation and training settings using validation data only.
5. Calibrate probabilities on validation data if probability estimates are needed.
6. Consider external validation on an independent dataset.
7. Review Grad-CAM maps critically: they show influential regions for the model, not proof of pathology.
8. Add unit tests, image-quality checks, and robust error handling before a formal demonstration.
