import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

from augmentations import apply_all_augmentations, AUGMENTATION_NAMES
from model import load_model, predict
import json
import os
import random
from PIL import Image

DEMO_DIR = "assets/demo_birds"

with open("weights/class_map.json", "r") as f:
    CLASS_MAP = json.load(f)

BIRD_CLASSES = [CLASS_MAP[str(i)] for i in range(len(CLASS_MAP))]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backyard Bird Classifier",
    page_icon="🐦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2d6a4f;
        border-left: 4px solid #52b788;
        padding-left: 10px;
        margin: 1.2rem 0 0.6rem 0;
    }
    .pill {
        display: inline-block;
        background: #d8f3dc;
        color: #1b4332;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .confidence-bar-label {
        font-size: 0.85rem;
        color: #333;
    }
    .aug-caption {
        font-size: 0.78rem;
        color: #666;
        text-align: center;
    }
    .model-info {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #166534;
    }
    .stProgress > div > div > div > div {
        background-color: #52b788;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🐦 Backyard Bird Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a bird photo → get a species prediction + see how '
    'real-world camera conditions affect model confidence via live augmentation.</div>',
    unsafe_allow_html=True,
)

# Model info banner
st.markdown("""
<div class="model-info">
    <b>Transfer Learning:</b> EfficientNetB0 pre-trained on ImageNet, fine-tuned on 10 common
    North American backyard bird species &nbsp;|&nbsp;
    <b>Augmentations:</b> Gaussian blur · Brightness/contrast jitter · Gaussian noise ·
    Random crop · JPEG compression
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    with st.spinner("Loading EfficientNetB0 model…"):
        return load_model()

model = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_aug_detail = st.checkbox("Show augmentation details", value=True)
    num_top_preds = st.slider("Top N predictions to show", 3, 10, 5)
    aug_intensity = st.select_slider(
        "Augmentation intensity",
        options=["Mild", "Moderate", "Aggressive"],
        value="Moderate",
    )
    st.markdown("---")
    st.markdown("### 🐦 Supported Species")
    for bird in BIRD_CLASSES:
        st.markdown(f'<span class="pill">{bird}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📚 Model Details")
    st.markdown("""
- **Base model:** EfficientNetB0  
- **Pre-trained on:** ImageNet (1.2M images)  
- **Fine-tuned on:** iNaturalist bird subset  
- **Input size:** 224 × 224  
- **Classes:** 10 backyard species  
    """)

# ── Main upload area ──────────────────────────────────────────────────────────
col_upload, col_gap, col_result = st.columns([1.1, 0.08, 1.82])

with col_upload:
    st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a bird photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Works best with clear photos. Try it on blurry or dark shots too!",
    )

    use_demo = st.checkbox("Use a demo image instead", value=False)

    if uploaded_file is None and use_demo:
        demo_files = [
            f for f in os.listdir(DEMO_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        chosen = random.choice(demo_files)
        demo_path = os.path.join(DEMO_DIR, chosen)

        image = Image.open(demo_path).convert("RGB")

        st.image(image, caption=f"Demo image: {chosen}", use_container_width=True)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width="stretch")
        st.markdown(f"**Size:** {image.size[0]} × {image.size[1]} px")
    else:
        image = None
        st.info("👆 Upload a photo or tick 'Use a demo image' to get started.")

# ── Results ───────────────────────────────────────────────────────────────────
with col_result:
    if image is not None:

        # ── 1. Prediction ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Prediction</div>', unsafe_allow_html=True)

        with st.spinner("Running inference…"):
            probs = predict(model, image)
            time.sleep(0.3)  # tiny pause so spinner is visible

        top_idx = np.argsort(probs)[::-1]
        top_species = BIRD_CLASSES[top_idx[0]]
        top_conf = probs[top_idx[0]]

        pred_col, conf_col = st.columns(2)
        pred_col.metric("🏆 Predicted Species", top_species)
        conf_col.metric("Confidence", f"{top_conf * 100:.1f}%")

        # Confidence bar chart
        st.markdown("**Top predictions**")
        fig_pred, ax_pred = plt.subplots(figsize=(5.5, 2.8))
        top_n = num_top_preds
        top_birds = [BIRD_CLASSES[i] for i in top_idx[:top_n]]
        top_probs = [probs[i] for i in top_idx[:top_n]]
        colors = ["#2d6a4f" if i == 0 else "#74c69d" for i in range(top_n)]
        bars = ax_pred.barh(top_birds[::-1], top_probs[::-1], color=colors[::-1], height=0.55)
        ax_pred.set_xlim(0, 1)
        ax_pred.set_xlabel("Confidence", fontsize=9)
        for bar, prob in zip(bars, top_probs[::-1]):
            ax_pred.text(
                min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%", va="center", fontsize=8.5, color="#1b4332",
            )
        ax_pred.spines[["top", "right"]].set_visible(False)
        ax_pred.tick_params(labelsize=9)
        fig_pred.tight_layout()
        st.pyplot(fig_pred, width="stretch")
        plt.close(fig_pred)

        st.markdown("---")

        # ── 2. Augmentation viewer ────────────────────────────────────────────
        st.markdown('<div class="section-header">🎨 Augmentation Viewer</div>', unsafe_allow_html=True)
        st.caption(
            "These augmentations simulate real-world camera conditions a backyard birder might encounter. "
            "Each version is also passed through the model to show confidence drift."
        )

        intensity_map = {"Mild": 0.3, "Moderate": 0.6, "Aggressive": 1.0}
        intensity_val = intensity_map[aug_intensity]

        with st.spinner("Applying augmentations…"):
            aug_images = apply_all_augmentations(image, intensity=intensity_val)

        # Grid: original + 5 augmentations
        all_imgs = [("Original", image)] + list(zip(AUGMENTATION_NAMES, aug_images))

        fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
        fig.patch.set_facecolor("#f8fdf9")

        for ax, (name, img) in zip(axes.flatten(), all_imgs):
            aug_probs = predict(model, img)
            aug_top = BIRD_CLASSES[np.argmax(aug_probs)]
            aug_conf = aug_probs.max()

            ax.imshow(np.array(img))
            ax.set_title(name, fontsize=10, fontweight="bold", pad=4, color="#1b4332")
            label_color = "#2d6a4f" if aug_top == top_species else "#c1121f"
            ax.set_xlabel(
                f"{aug_top}  ({aug_conf * 100:.0f}%)",
                fontsize=8.5,
                color=label_color,
                labelpad=3,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#b7e4c7")
                spine.set_linewidth(1.5)

        fig.suptitle(
            "Model predictions across augmented views  |  🟢 Correct species  🔴 Prediction changed",
            fontsize=9.5,
            color="#555",
            y=0.02,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # ── 3. Robustness summary ─────────────────────────────────────────────
        if show_aug_detail:
            st.markdown('<div class="section-header">📊 Robustness Summary</div>', unsafe_allow_html=True)

            aug_results = []
            for name, aug_img in zip(AUGMENTATION_NAMES, aug_images):
                ap = predict(model, aug_img)
                aug_results.append({
                    "Augmentation": name,
                    "Predicted": BIRD_CLASSES[np.argmax(ap)],
                    "Confidence": f"{ap.max() * 100:.1f}%",
                    "Same prediction?": "✅" if BIRD_CLASSES[np.argmax(ap)] == top_species else "❌",
                })

            st.table(aug_results)

            # Confidence drift chart
            aug_confs = [predict(model, img).max() for _, img in zip(AUGMENTATION_NAMES, aug_images)]
            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
            ax2.plot(
                ["Original"] + AUGMENTATION_NAMES,
                [top_conf] + aug_confs,
                marker="o", linewidth=2, color="#52b788", markersize=7,
                markerfacecolor="#2d6a4f",
            )
            ax2.axhline(top_conf, linestyle="--", color="#aaa", linewidth=1, label="Baseline")
            ax2.set_ylabel("Confidence", fontsize=9)
            ax2.set_ylim(0, 1.05)
            ax2.set_title("Confidence drift across augmentations", fontsize=10)
            ax2.tick_params(labelsize=8)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=20, ha="right")
            fig2.tight_layout()
            st.pyplot(fig2, width="stretch")
            plt.close(fig2)
