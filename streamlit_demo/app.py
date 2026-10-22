from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

# Reuse the original repository's model implementation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.models import Decoder, VGGEncoder
from utils.utils import adaptive_instance_normalizaiton

VGG_PATH = ROOT / "vgg_normalised.pth"
DECODER_PATH = ROOT / "experiment" / "experiment1" / "decoder_128.pth"
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

st.set_page_config(page_title="StyleForge AI | Neural Style Transfer", page_icon="🎨", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.7rem 1.8rem; border-radius: 22px; background: linear-gradient(120deg, rgba(93,63,211,.16), rgba(28,166,163,.12)); border: 1px solid rgba(128,128,128,.2); margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .4rem 0; font-size: 2.5rem;}
.hero p {margin: 0; font-size: 1.05rem; opacity: .85;}
.small-note {opacity: .72; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎨 StyleForge AI</h1><p>Turn photographs into art with Adaptive Instance Normalization (AdaIN).</p></div>', unsafe_allow_html=True)
st.write("Choose a **content image**, choose a **style reference**, tune the strength, and generate an artistic transformation using your trained PyTorch model.")

with st.sidebar:
    st.header("⚙️ Stylization settings")
    alpha = st.slider("Style strength (α)", 0.0, 1.0, 1.0, 0.05, help="0 keeps the content features; 1 applies the full AdaIN style transformation.")
    image_size = st.select_slider("Processing resolution", options=[256, 320, 384, 448, 512], value=384, help="Lower resolutions are faster; higher resolutions may require more memory.")
    st.caption("The first inference may take longer while the model is loaded.")
    st.divider()
    st.markdown("**Model architecture**\n\n- VGG-19 feature encoder\n- Adaptive Instance Normalization\n- Trained convolutional decoder")
    st.caption("The model runs on the host machine. GPU acceleration is used only if the host provides CUDA.")


def read_image(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        if max(image.size) > 2400:
            image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
        return image
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        return None


@st.cache_resource(show_spinner="Loading the AdaIN model…")
def load_model(vgg_path: str, decoder_path: str, device_name: str):
    device = torch.device(device_name)
    encoder = VGGEncoder(vgg_path).to(device)
    decoder = Decoder().to(device)
    # Only load trusted state-dict checkpoints from your own project.
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    encoder.eval()
    decoder.eval()
    return encoder, decoder, device


def stylize(content, style, encoder, decoder, device, alpha_value, size):
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    content_tensor = transform(content).unsqueeze(0).to(device)
    style_tensor = transform(style).unsqueeze(0).to(device)
    with torch.inference_mode():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)
        target_features = adaptive_instance_normalizaiton(content_features, style_features)
        target_features = alpha_value * target_features + (1.0 - alpha_value) * content_features
        result = decoder(target_features).squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(result)


def image_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


left, right = st.columns(2, gap="large")
with left:
    st.subheader("1 · Content image")
    content_upload = st.file_uploader("Upload the photograph to transform", type=ALLOWED_TYPES, key="content_upload")
    content_image = read_image(content_upload)
    if content_image is not None:
        st.image(content_image, caption="Content image", use_container_width=True)
    else:
        st.info("Upload a JPG or PNG photograph to get started.")
with right:
    st.subheader("2 · Style reference")
    style_upload = st.file_uploader("Upload a painting, drawing, or texture", type=ALLOWED_TYPES, key="style_upload")
    style_image = read_image(style_upload)
    if style_image is not None:
        st.image(style_image, caption="Style reference", use_container_width=True)
    else:
        st.info("Upload an artwork or any image whose visual style you want to transfer.")

st.divider()
missing_files = []
if not VGG_PATH.is_file():
    missing_files.append(str(VGG_PATH.relative_to(ROOT)))
if not DECODER_PATH.is_file():
    missing_files.append(str(DECODER_PATH.relative_to(ROOT)))
if missing_files:
    st.warning("The demo interface is ready, but inference is disabled until the model checkpoints are added to the repository.")
    st.markdown("**Missing model files:**")
    for file_name in missing_files:
        st.code(file_name, language="text")
    st.markdown("Add the compatible pretrained AdaIN VGG weights and your trained decoder checkpoint at the paths above, or update `VGG_PATH` and `DECODER_PATH` in `streamlit_demo/app.py`. These weights were not included in the uploaded ZIP.")

generate_col, info_col = st.columns([1, 2], vertical_alignment="center")
with generate_col:
    generate = st.button("✨ Generate stylized image", type="primary", use_container_width=True, disabled=(content_image is None or style_image is None or bool(missing_files)))
with info_col:
    st.caption("For a quick demo, try 256–384 px first. Results depend on the trained decoder, the images, and the selected style strength.")

if generate:
    try:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        with st.spinner("Loading the model and creating your artwork…"):
            encoder, decoder, device = load_model(str(VGG_PATH), str(DECODER_PATH), device_name)
            output_image = stylize(content_image, style_image, encoder, decoder, device, alpha, image_size)
        st.success(f"Stylization complete · running on {device.type.upper()}")
        st.subheader("3 · Your result")
        result_col, original_col = st.columns(2, gap="large")
        with result_col:
            st.image(output_image, caption="AI-generated stylization", use_container_width=True)
            st.download_button("⬇️ Download result (PNG)", data=image_bytes(output_image), file_name="styleforge_result.png", mime="image/png", use_container_width=True)
        with original_col:
            st.image(content_image, caption="Original content", use_container_width=True)
        st.caption(f"Style strength α = {alpha:.2f} · Processing resolution = {image_size}px")
    except Exception as exc:
        st.error("The model could not generate an image. Check that the checkpoint files match this repository's model architecture.")
        st.exception(exc)

st.divider()
st.markdown('<p class="small-note">Built with PyTorch, VGG-19, Adaptive Instance Normalization, and Streamlit. Upload only images you have permission to use. Public demo uploads should not be considered private.</p>', unsafe_allow_html=True)
