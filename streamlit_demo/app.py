"""StyleForge AI - Streamlit version of the Flask/HTML front-end.

Same model, same look, no login/database: pick a content photo, pick a style
(preset or upload), set the strength and press "Transfer Style".

Run locally from the repo root:
    streamlit run streamlit_demo/app.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.models import Decoder, VGGEncoder  # noqa: E402
from utils.utils import adaptive_instance_normalizaiton  # noqa: E402

VGG_PATH = ROOT / "vgg_normalised.pth"
DECODER_PATH = ROOT / "experiment" / "experiment1" / "decoder_128.pth"
STYLES_DIR = ROOT / "static" / "styles"
EXAMPLES_DIR = ROOT / "examples"
ALLOWED_TYPES = ["png", "jpg", "jpeg"]
MAX_TOTAL_BYTES = 10 * 1024 * 1024  # same 10 MB limit as the Flask app

st.set_page_config(page_title="StyleForge AI", page_icon="🎨", layout="wide")

# --------------------------------------------------------------------------
# Look & feel (ported from templates/index.html)
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
.stApp {
  background-image: radial-gradient(1200px 600px at 15% -10%, rgba(99,102,241,.18), transparent 60%),
                    radial-gradient(900px 500px at 90% 10%, rgba(244,114,182,.12), transparent 60%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1150px; padding-top: 1.5rem; padding-bottom: 4rem; }

.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero h1 {
  font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: clamp(2rem, 6vw, 3.5rem);
  background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 .6rem;
}
.hero p { color: #94a3b8; font-size: 1.2rem; margin: 0; }

.card-title {
  font-family: 'Orbitron', sans-serif; letter-spacing: 1px; text-transform: uppercase;
  font-size: .9rem; color: #f8fafc; text-align: center; padding: .4rem 0 .8rem;
  border-bottom: 1px solid rgba(255,255,255,.1); margin-bottom: 1rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(30,41,59,.6); border-radius: 20px;
  border: 1px solid rgba(255,255,255,.1) !important;
}
.placeholder {
  height: 260px; display: flex; align-items: center; justify-content: center; text-align: center;
  border: 2px dashed #334155; border-radius: 10px; color: #64748b; background: #0f172a; padding: 1rem;
}
.sel-name { font-size: .85rem; text-align: center; color: #cbd5e1; margin: .2rem 0 .3rem; }
.sel-name.active { color: #c084fc; font-weight: 600; }

div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); border: none; border-radius: 50px;
  font-family: 'Orbitron', sans-serif; letter-spacing: 2px; text-transform: uppercase;
  padding: .8rem 2rem; box-shadow: 0 10px 20px -10px rgba(168,85,247,.6);
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }
.small-note { color: #64748b; font-size: .85rem; text-align: center; }
.section-title { font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 1px;
  text-align: center; margin: 2.5rem 0 1rem; color: #f8fafc; font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model (loaded once per server process, shared by all visitors)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the AdaIN model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VGGEncoder(str(VGG_PATH)).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(
        torch.load(DECODER_PATH, map_location=device, weights_only=True)
    )
    encoder.eval()
    decoder.eval()
    return encoder, decoder, device


def stylize(content: Image.Image, style: Image.Image, alpha: float) -> Image.Image:
    encoder, decoder, device = load_model()
    to_tensor = transforms.ToTensor()
    c = to_tensor(content).unsqueeze(0).to(device)
    s = to_tensor(style).unsqueeze(0).to(device)
    with torch.inference_mode():
        c_feat = encoder(c, is_test=True)
        s_feat = encoder(s, is_test=True)
        t_feat = adaptive_instance_normalizaiton(c_feat, s_feat)
        t_feat = alpha * t_feat + (1 - alpha) * c_feat
        out = decoder(t_feat)
    out = out.squeeze(0).cpu().clamp(0, 1)
    img = transforms.ToPILImage()(out)
    # The encoder rounds sizes up to a multiple of 8; crop back to the input size
    # so "before" and "after" line up exactly.
    return img.crop((0, 0, content.width, content.height))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def prepare(img: Image.Image, max_side: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)  # never upscales
    return img


def read_upload(file) -> Image.Image | None:
    if file is None:
        return None
    try:
        file.seek(0)
        return Image.open(file).convert("RGB")
    except Exception:
        st.error("Could not read that image. Please upload a valid PNG or JPG.")
        return None


@st.cache_data(show_spinner=False)
def load_style_library() -> list[dict]:
    """styles.json first; any other image dropped in static/styles is added under 'More'."""
    items, seen = [], set()
    manifest = STYLES_DIR / "styles.json"
    if manifest.is_file():
        for e in json.loads(manifest.read_text(encoding="utf-8")):
            f = STYLES_DIR / str(e.get("file", ""))
            if f.is_file() and e["id"] not in seen:
                seen.add(e["id"])
                items.append({"id": e["id"], "file": f.name,
                              "name": e.get("name") or e["id"],
                              "category": e.get("category") or "More"})
    listed = {i["file"] for i in items}
    for p in sorted(STYLES_DIR.glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name not in listed:
            items.append({"id": p.stem.lower(), "file": p.name,
                          "name": p.stem.replace("_", " ").replace("-", " ").title(),
                          "category": "More"})
    return items


@st.cache_data(show_spinner=False)
def style_thumb(file_name: str) -> bytes:
    """Small JPEG for the gallery so 60+ tiles stay light."""
    thumb_path = STYLES_DIR / "thumbs" / file_name
    path = thumb_path if thumb_path.is_file() else STYLES_DIR / file_name
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((220, 220))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue()


@st.cache_data(show_spinner=False)
def style_full(file_name: str) -> Image.Image:
    """Preset style at inference size only (some originals are > 6000 px wide)."""
    with Image.open(STYLES_DIR / file_name) as im:
        im.draft("RGB", (1024, 1024))  # fast JPEG downscale on decode
        return im.convert("RGB")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def compare_slider(before: Image.Image, after: Image.Image):
    """Before/after slider (same idea as the .compare widget in index.html)."""
    aspect = before.width / before.height
    max_w = 560
    h = int(max_w / aspect) + 20
    html = f"""
    <style>
      body{{margin:0;background:transparent;font-family:Inter,sans-serif}}
      .wrap{{max-width:{max_w}px;margin:0 auto}}
      .cmp{{position:relative;width:100%;aspect-ratio:{before.width}/{before.height};overflow:hidden;
            border-radius:14px;background:#0b1220;user-select:none}}
      .cmp img{{position:absolute;inset:0;width:100%;height:100%;object-fit:fill;pointer-events:none}}
      #after{{clip-path:inset(0 0 0 50%)}}
      .line{{position:absolute;top:0;bottom:0;left:50%;width:3px;background:#fff;
             box-shadow:0 0 12px #a855f7;transform:translateX(-50%);pointer-events:none}}
      .tag{{position:absolute;top:10px;padding:3px 10px;border-radius:12px;font-size:12px;
            color:#fff;background:rgba(15,23,42,.7)}}
      input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
    </style>
    <div class="wrap"><div class="cmp" id="c">
      <img src="{data_uri(before)}"><img id="after" src="{data_uri(after)}">
      <div class="line" id="l"></div>
      <span class="tag" style="left:10px">Original</span><span class="tag" style="right:10px">Stylized</span>
      <input type="range" min="0" max="100" value="50" id="r">
    </div></div>
    <script>
      const r=document.getElementById('r'),a=document.getElementById('after'),l=document.getElementById('l');
      r.addEventListener('input',()=>{{a.style.clipPath='inset(0 0 0 '+r.value+'%)';l.style.left=r.value+'%'}});
    </script>"""
    if hasattr(st, "iframe"):  # Streamlit >= 1.5x
        st.iframe(html, height=h)
    else:
        st.components.v1.html(html, height=h)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("style_id", None)
st.session_state.setdefault("result", None)

library = load_style_library()
by_id = {s["id"]: s for s in library}

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>STYLEFORGE AI</h1>'
    "<p>Redefine Reality with AI-Powered Artistry</p></div>",
    unsafe_allow_html=True,
)

missing = [str(p.relative_to(ROOT)) for p in (VGG_PATH, DECODER_PATH) if not p.is_file()]
if missing:
    st.error("Model files are missing from the repository: " + ", ".join(missing))
    st.stop()

load_model()  # warm up now so the first "Transfer Style" click is fast

# --------------------------------------------------------------------------
# Content + style upload
# --------------------------------------------------------------------------
col_c, col_s = st.columns(2, gap="large")

with col_c:
    with st.container(border=True):
        st.markdown('<div class="card-title">Content Source</div>', unsafe_allow_html=True)
        content_file = st.file_uploader("Content image", type=ALLOWED_TYPES,
                                        key="content_up", label_visibility="collapsed")
        content_img = read_upload(content_file)
        if content_img is not None:
            st.image(content_img, width="stretch")
        else:
            st.markdown('<div class="placeholder">📷 Upload the photo you want to transform<br>'
                        "PNG / JPG · 10 MB total</div>", unsafe_allow_html=True)

with col_s:
    with st.container(border=True):
        st.markdown('<div class="card-title">Style Reference</div>', unsafe_allow_html=True)
        style_file = st.file_uploader("Style image", type=ALLOWED_TYPES,
                                      key="style_up", label_visibility="collapsed")
        style_upload_img = read_upload(style_file)
        preset = by_id.get(st.session_state.style_id)
        if style_upload_img is not None:
            st.image(style_upload_img, width="stretch")
            st.caption("Using your uploaded style (it overrides the library choice).")
        elif preset is not None:
            st.image(style_thumb(preset["file"]), width="stretch")
            st.caption(f"Library style: **{preset['name']}**")
        else:
            st.markdown('<div class="placeholder">🎨 Upload a style image, or pick one '
                        "from the Style Library below</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Style library
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="card-title">Style Library</div>', unsafe_allow_html=True)
    categories = list(dict.fromkeys(s["category"] for s in library))
    chosen_cat = st.pills("Category", ["All"] + categories, default=categories[0],
                          label_visibility="collapsed", key="cat")
    shown = [s for s in library if chosen_cat in (None, "All", s["category"])]
    per_row = 5
    for i in range(0, len(shown), per_row):
        cols = st.columns(per_row)
        for col, s in zip(cols, shown[i:i + per_row]):
            with col:
                active = s["id"] == st.session_state.style_id
                st.image(style_thumb(s["file"]), width="stretch")
                st.markdown(f'<div class="sel-name {"active" if active else ""}">'
                            f'{"✓ " if active else ""}{s["name"]}</div>', unsafe_allow_html=True)
                if st.button("Selected" if active else "Select", key=f"pick_{s['id']}",
                             width="stretch", type="primary" if active else "secondary"):
                    st.session_state.style_id = s["id"]
                    st.rerun()
    st.markdown('<p class="small-note">Pick a preset above, or upload your own style image in the box.</p>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Strength + run
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="card-title">Style Strength</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        alpha = st.slider("Style strength (α)", 0.0, 1.0, 1.0, 0.05,
                          help="1.0 = full style. Lower values keep more of your original photo. "
                               "Try 0.5–0.7 for portraits, 0.8–1.0 for landscapes.")
        res = st.select_slider("Processing size (longest side, px)", options=[256, 384, 512], value=256,
                               help="256 is fastest. Larger is sharper but slower and uses more memory.")

ready = content_img is not None and (style_upload_img is not None or preset is not None)
_, btn, _ = st.columns([1, 2, 1])
with btn:
    go = st.button("✨ Transfer Style", type="primary", width="stretch", disabled=not ready)
if not ready:
    st.markdown('<p class="small-note">Add a content image and a style to enable the button.</p>',
                unsafe_allow_html=True)

if go:
    total = (content_file.size if content_file else 0) + (style_file.size if style_file else 0)
    if total > MAX_TOTAL_BYTES:
        st.error("Upload too large: the two images together must be under 10 MB.")
    else:
        try:
            style_src = style_upload_img if style_upload_img is not None else style_full(preset["file"])
            style_name = "Your upload" if style_upload_img is not None else preset["name"]
            with st.status("Neural network is dreaming…", expanded=True) as status:
                t0 = time.time()
                c = prepare(content_img, res)
                s = prepare(style_src, res)
                out = stylize(c, s, alpha)
                status.update(label=f"Done in {time.time() - t0:.1f}s", state="complete", expanded=False)
            st.session_state.result = {"content": c, "output": out, "alpha": alpha,
                                       "style": style_name, "res": res}
        except Exception as exc:  # noqa: BLE001
            st.session_state.result = None
            st.error("Style transfer failed. Please try different images.")
            st.caption(f"Details: {exc}")

# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
r = st.session_state.result
if r:
    with st.container(border=True):
        st.markdown('<div class="card-title" style="color:#34d399">✔ Your Stylized Image</div>',
                    unsafe_allow_html=True)
        t_cmp, t_res, t_org = st.tabs(["Compare", "Stylized", "Original"])
        with t_cmp:
            compare_slider(r["content"], r["output"])
            st.markdown('<p class="small-note">Drag across the image to compare.</p>', unsafe_allow_html=True)
        with t_res:
            st.image(r["output"], width="stretch")
        with t_org:
            st.image(r["content"], width="stretch")
        st.caption(f"Style: {r['style']} · strength α = {r['alpha']:.2f} · {r['res']}px")
        st.download_button("⬇ Download PNG", data=png_bytes(r["output"]),
                           file_name="styleforge_result.png", mime="image/png",
                           type="primary", width="stretch")

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">Examples</div>', unsafe_allow_html=True)
st.markdown('<p class="small-note">Content image + style image → output.</p>', unsafe_allow_html=True)
examples = [
    ("brad pitt.jpg", "sketch.png", "stylized_brad_pitt.jpg"),
    ("brad pitt.jpg", "picasso seated nude hr.jpg", "stylized_brad pitt (1).jpg"),
]
for content_name, style_name_, out_name in examples:
    paths = [EXAMPLES_DIR / n for n in (content_name, style_name_, out_name)]
    if not all(p.is_file() for p in paths):
        continue
    with st.container(border=True):
        a, b, c = st.columns(3)
        for col, label, p in zip((a, b, c), ("Content", "Style", "Output"), paths):
            with col:
                st.caption(label)
                st.image(str(p), width="stretch")

# --------------------------------------------------------------------------
# FAQ
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">FAQ</div>', unsafe_allow_html=True)
FAQ = [
    ("How do I turn my photo into art?",
     "Upload your photo as the content image, choose a style from the Style Library (or upload your own), "
     "set the style strength and press **Transfer Style**. Use the Compare tab for a before/after slider, "
     "then download the PNG."),
    ("What does Style Strength do?",
     "It blends your original photo with the fully stylized version. At 1.0 you get the full style; lower values "
     "keep more of the original. Try 0.5–0.7 for portraits and 0.8–1.0 for landscapes and objects."),
    ("Why do faces sometimes look blurry or distorted?",
     "The model matches the overall colour and texture of the style across the whole picture and doesn't know "
     "where eyes, nose or mouth are. For portraits try a lower strength, softer styles such as Watercolor Wash "
     "or Pastel Dream, and a sharp, front-facing, well-lit photo."),
    ("Why is my result smaller than my original photo?",
     "To keep it fast, images are resized so the longest side is at most the 'Processing size' you chose "
     "(256 px by default). The aspect ratio is kept."),
    ("Which files can I upload?",
     "PNG and JPG/JPEG. The two images together must be under 10 MB."),
    ("Are my images stored?",
     "No. Images are processed in memory for your session and are not saved or used to train the model."),
    ("What model is behind this?",
     "Adaptive Instance Normalization (AdaIN). A frozen, pretrained VGG-19 reads the features of both images, "
     "AdaIN matches the photo's feature statistics to the style's, and a trained decoder paints the result — "
     "all in one forward pass, with no per-style training."),
]
for q, a in FAQ:
    with st.expander(q):
        st.write(a)

st.markdown('<p class="small-note" style="margin-top:2rem">Built with PyTorch, VGG-19, AdaIN and Streamlit.</p>',
            unsafe_allow_html=True)
