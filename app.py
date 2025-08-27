import io
import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------- UI THEME ----------------
st.set_page_config(page_title="EuroSAT Land-Use (ResNet-50)", page_icon="🌍", layout="centered")
st.markdown(
    """
    <style>
      .stApp { background-color: #0b1220; color: #e6edf3; }
      .hdr { font-size: 1.9rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0; }
      .sub { color:#9aa4b2; margin-bottom:1rem; }
      .card {
        background: #111827; border: 1px solid #1f2937; border-radius: 14px;
        padding: 14px; box-shadow: 0 4px 18px rgba(0,0,0,0.25);
      }
      .chip {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        background:#1f2937; border:1px solid #2b3750; margin-right:6px; margin-bottom:6px;
        font-weight:600;
      }
      .ok { color:#a7f3d0; } .med { color:#fde68a; } .bad { color:#fca5a5; }
      .foot { color:#6b7280; font-size:0.85rem; margin-top:1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
<style>
div[data-testid="stMetricValue"]{color:#e6edf3!important;font-weight:800!important;}
div[data-testid="stMetricDelta"]{color:#a7f3d0!important;}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS / CONFIG ----------------
HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "weights" / "resnet50_eurosat.pt"
LABELS_PATH  = HERE / "assets"  / "label2idx.json"
SAMPLE_DIR   = HERE / "sample_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------- HELPERS ----------------
@st.cache_data
def load_classes():
    if not LABELS_PATH.exists():
        st.error("Missing assets/label2idx.json"); st.stop()
    return sorted(json.loads(LABELS_PATH.read_text()).keys())

@st.cache_resource
def load_model(quantize_cpu: bool):
    classes = load_classes()
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    if not WEIGHTS_PATH.exists():
        st.error("Missing weights/resnet50_eurosat.pt"); st.stop()
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    if quantize_cpu and DEVICE == "cpu":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return model, classes

def to_cam_rgb(t: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=t.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD, device=t.device)[:, None, None]
    x = (t * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return x

def logits_single(model, pil_img: Image.Image):
    x = eval_tf(pil_img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        return model(x).squeeze(0)

def logits_tta(model, pil_img: Image.Image, max_views: int = 8):
    views = [
        pil_img,
        ImageOps.mirror(pil_img),
        ImageOps.flip(pil_img),
        ImageOps.mirror(ImageOps.flip(pil_img)),
        pil_img.rotate(-10, resample=Image.BILINEAR),
        pil_img.rotate(10,  resample=Image.BILINEAR),
        pil_img.rotate(-5,  resample=Image.BILINEAR),
        pil_img.rotate(5,   resample=Image.BILINEAR),
    ][:max_views]
    outs = []
    with torch.inference_mode():
        for v in views:
            outs.append(model(eval_tf(v).unsqueeze(0).to(DEVICE)))
    return torch.stack(outs).mean(0).squeeze(0)

def predict_and_cam(model, classes, pil_img, use_tta, temperature, cam_layer, cam_opacity):
    logits = logits_tta(model, pil_img) if use_tta else logits_single(model, pil_img)
    probs = F.softmax(logits / temperature, dim=0).cpu().numpy()

    layer = model.layer4[-1] if cam_layer == "layer4" else model.layer3[-1]
    x = eval_tf(pil_img).unsqueeze(0).to(DEVICE)
    with GradCAM(model=model, target_layers=[layer]) as cam:
        mask = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(probs.argmax()))])[0]
    cam_rgb = show_cam_on_image(to_cam_rgb(x[0]), mask, use_rgb=True, image_weight=cam_opacity)
    return probs, cam_rgb

def chip(text, score):
    css = "ok" if score >= 0.75 else ("med" if score >= 0.5 else "bad")
    return f'<span class="chip {css}">{text} — {score*100:.1f}%</span>'

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown('<div class="hdr">⚙️ Settings</div>', unsafe_allow_html=True)
    st.write(f"**Device:** `{DEVICE}`")
    use_tta     = st.checkbox("Use TTA (x8)", value=True)
    temperature = st.slider("Temperature", 0.5, 3.0, 1.0, 0.05)
    warn_thr    = st.slider("Low-confidence warning", 0.0, 1.0, 0.5, 0.01)
    cam_layer   = st.selectbox("Grad-CAM layer", ["layer4", "layer3"], index=0)
    cam_opacity = st.slider("CAM opacity", 0.2, 0.8, 0.5, 0.05)
    quantize_cpu= st.checkbox("INT8 quantize (CPU only)", value=False)

    model, classes = load_model(quantize_cpu)
    st.write("**Classes (10):**")
    st.caption(", ".join(classes))

# ---------------- HEADER ----------------
st.markdown('<div class="hdr">🌍 EuroSAT Land-Use Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">ResNet-50 fine-tuned on EuroSAT RGB • Grad-CAM explainability</div>', unsafe_allow_html=True)

# ---------------- SINGLE IMAGE ----------------
left, right = st.columns([3, 1])
with left:
    uploaded = st.file_uploader("Upload one tile (jpg/png)", type=["jpg", "jpeg", "png"])
with right:
    samples = sorted(list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png")))
    if samples and st.button("🎲 Try sample"):
        uploaded = io.BytesIO(Path(random.choice(samples)).read_bytes())

if uploaded:
    pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    with st.spinner("Running inference…"):
        probs, cam_img = predict_and_cam(model, classes, pil, use_tta, temperature, cam_layer, cam_opacity)

    topk_idx = probs.argsort()[-5:][::-1]
    top1_idx = int(topk_idx[0])
    top1_label = classes[top1_idx]
    top1_prob  = float(probs[top1_idx])

    if top1_prob < warn_thr:
        st.warning(f"Low confidence: {top1_prob*100:.1f}% — review Grad-CAM or try TTA.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric(label="Prediction", value=top1_label, delta=f"{top1_prob*100:.1f}% confidence")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input")
        st.image(pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Grad-CAM")
        st.image(cam_img, use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(cam_img).save(buf, format="PNG"); buf.seek(0)
        st.download_button("Download Grad-CAM", data=buf.getvalue(),
                           file_name=f"gradcam_{top1_label}.png", mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top predictions")
    st.markdown(" ".join(chip(classes[i], float(probs[i])) for i in topk_idx[:3]), unsafe_allow_html=True)

    df = pd.DataFrame({"class": [classes[i] for i in topk_idx[:5]],
                       "probability": [float(probs[i]) for i in topk_idx[:5]]}
                      ).sort_values("probability", ascending=True).reset_index(drop=True)
    try:
        chart = (
            alt.Chart(df)
              .mark_bar(size=22, color="#60a5fa")
              .encode(
                  x=alt.X("probability:Q", title="Probability",
                          scale=alt.Scale(domain=[0,1]),
                          axis=alt.Axis(format=".0%", labelColor="#e6edf3", titleColor="#e6edf3")),
                  y=alt.Y("class:N", sort="-x", title=None,
                          axis=alt.Axis(labelColor="#e6edf3")),
                  tooltip=[alt.Tooltip("class:N", title="Class"),
                           alt.Tooltip("probability:Q", title="Probability", format=".1%")],
              )
              .configure_view(fill="transparent", strokeOpacity=0)
              .configure_axis(grid=False)
              .properties(height=180, width="container")
        )
        st.altair_chart(chart, use_container_width=True, theme=None)
    except Exception:
        st.bar_chart(df.set_index("class"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BATCH MODE ----------------
st.markdown("—")
st.subheader("Batch classify (optional)")
batch_files = st.file_uploader("Upload multiple tiles", type=["jpg","jpeg","png"], accept_multiple_files=True)
if batch_files:
    rows = []
    with st.spinner("Running batch…"):
        for f in batch_files:
            pil = Image.open(io.BytesIO(f.read())).convert("RGB")
            probs, _ = predict_and_cam(model, classes, pil, use_tta, temperature, "layer4", cam_opacity)
            idx = int(probs.argmax())
            rows.append({"file": f.name, "pred": classes[idx], "conf": float(probs[idx])})
    df_batch = pd.DataFrame(rows).sort_values("conf", ascending=False)
    st.dataframe(df_batch, use_container_width=True)
    csv = df_batch.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown('<div class="foot">Made with PyTorch • torchvision • grad-cam • Streamlit</div>', unsafe_allow_html=True)
