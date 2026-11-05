import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="RetinaLens | Retinal Image Analysis", page_icon="👁️", layout="wide")

MODEL_PATH = Path("models/retinal_efficientnet.pth")
METRICS_PATH = Path("models/metrics.json")
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["class_names"]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, class_names, device


def preprocess(image, image_size=224):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tfm(image.convert("RGB")).unsqueeze(0)


def gradcam_overlay(model, tensor, original_image, device):
    """Grad-CAM for EfficientNet-B0's final feature block."""
    target_layer = model.features[-1]
    activations, gradients = [], []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.to(device))
        class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()
        acts = activations[0].detach()
        grads = gradients[0].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        import cv2
        rgb = np.array(original_image.convert("RGB").resize((224, 224)))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.55 * rgb + 0.45 * heat)
        return Image.fromarray(overlay)
    finally:
        h1.remove()
        h2.remove()


def claude_explanation(class_name, confidence, probabilities):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_claude_api_key_here":
        return "Claude is not configured. Add ANTHROPIC_API_KEY to your .env file to enable explanations."
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        probability_text = "\n".join(
            f"- {name}: {value*100:.2f}%" for name, value in probabilities.items()
        )
        prompt = f"""
You explain the output of an experimental retinal image classification model.
This is not a clinical diagnosis. Do not claim to have inspected the image or
identified lesions. Explain only the model output supplied below.

Predicted dataset class: {class_name}
Model softmax score: {confidence*100:.2f}%
Class scores:
{probability_text}

Write a concise report with:
1. What the predicted dataset category generally means.
2. What this model output does and does not tell the user.
3. Why the score is not the same as clinical certainty.
4. A clear recommendation to seek assessment from a qualified eye-care professional.
Do not invent symptoms, patient details, or image findings.
"""
        response = client.messages.create(
            model=model_name,
            max_tokens=450,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    except Exception as exc:
        return f"Claude request failed: {exc}"


st.title("👁️ RetinaLens")
st.caption("Experimental retinal fundus image classification and explainability")
st.warning(
    "Research prototype only — not a medical device and not suitable for diagnosis, "
    "screening decisions, or treatment. Consult a qualified ophthalmologist."
)

model, class_names, device = load_model()
if model is None:
    st.error("Trained model not found. Train the model first using the command below.")
    st.code("python train.py --epochs 5 --batch-size 8", language="powershell")
    st.info("Dataset expected at dataset/Datasets/Diabetic Retinopathy/{train,valid,test}.")
    st.stop()

st.sidebar.success(f"Model loaded · {device}")
uploaded = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    tensor = preprocess(image)
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx])
    probability_map = {name: float(prob) for name, prob in zip(class_names, probs)}

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Model prediction")
        st.metric("Predicted dataset class", predicted_class)
        st.metric("Highest class score", f"{confidence*100:.2f}%")
        st.caption("Softmax score; not a calibrated probability of disease.")
        for name, prob in probability_map.items():
            st.write(f"**{name}** — {prob*100:.2f}%")
            st.progress(float(prob))

    st.divider()
    st.subheader("Grad-CAM visualisation")
    try:
        overlay = gradcam_overlay(model, tensor, image, device)
        st.image(overlay, caption="Grad-CAM overlay — regions influencing this model's score; not a clinical lesion map.", use_container_width=True)
    except Exception as exc:
        st.warning(f"Grad-CAM could not be generated: {exc}")

    st.divider()
    st.subheader("Claude explanation")
    if st.button("Generate explanation with Claude"):
        with st.spinner("Generating explanation..."):
            explanation = claude_explanation(predicted_class, confidence, probability_map)
        st.write(explanation)

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    st.divider()
    st.subheader("Held-out test-set performance")
    st.caption(f"Evaluated on {metrics.get('test_images', '—')} held-out images. These are dataset-specific experimental metrics.")
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("Accuracy", "test_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
    ]):
        col.metric(label, f"{metrics[key]*100:.2f}%")
    st.subheader("Per-class evaluation")
    import pandas as pd
    report = metrics.get("classification_report", {})
    rows = []
    for name in metrics.get("class_names", []):
        if name in report:
            rows.append({
                "Class": name,
                "Precision": f"{report[name]['precision']*100:.2f}%",
                "Recall": f"{report[name]['recall']*100:.2f}%",
                "F1-score": f"{report[name]['f1-score']*100:.2f}%",
                "Support": report[name]["support"],
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Metrics are computed by train.py on a held-out split. They are not evidence of clinical readiness.")