<div align="center">

# StyleForge AI
### Arbitrary Neural Style Transfer with Adaptive Instance Normalization

Transform a photograph with the colour, texture, and visual character of another image. Built with PyTorch, a frozen VGG-19 feature encoder, an AdaIN-conditioned decoder, and two browser interfaces.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Web-Flask-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)

**[Open the live Streamlit demo](https://adain-neural-style-transfer-39tfl8u4t9pp9vbieaksv6.streamlit.app/)**

</div>

---

## Overview

**StyleForge AI** implements arbitrary neural style transfer using **Adaptive Instance Normalization (AdaIN)**. A content image supplies the scene and composition; a separate style image supplies feature statistics associated with its colours, textures, and brushwork. A trained decoder converts the blended feature representation back into an image.

The repository contains the trained model weights, a 65-image preset style library, example results, the training workflow, and two ways to use the model:

| Interface | Intended use | Main capabilities |
|---|---|---|
| **Flask web application** | Full application with accounts | Registration, login, password reset, preset/custom styles, face preservation, and per-user transfer history. |
| **Streamlit demo** | Quick public or local demo | No account required, preset/custom styles, adjustable processing size, adjustable face preservation, before/after comparison, and PNG download. |

The model checkpoints are included. You can run inference without downloading a dataset or training the decoder again.

## Try it online

**[🎨 Launch StyleForge AI](https://adain-neural-style-transfer-39tfl8u4t9pp9vbieaksv6.streamlit.app/)**

Upload a content image, choose a preset or upload a style reference, adjust the style strength, and generate an image directly in your browser.

## Example results

The repository includes example content images, style references, and generated outputs.

| Content | Style reference | StyleForge output |
|---|---|---|
| ![Portrait content image](examples/brad%20pitt.jpg) | ![Sketch style reference](examples/sketch.png) | ![Portrait rendered in sketch style](examples/stylized_brad_pitt.jpg) |
| ![Portrait content image](examples/brad%20pitt.jpg) | ![Painting style reference](examples/picasso%20seated%20nude%20hr.jpg) | ![Portrait rendered with a painting style](examples/stylized_brad%20pitt%20(1).jpg) |

## Features

### Neural style transfer

- Transfer the appearance of an arbitrary reference image; no per-style training is needed.
- Choose from **65 bundled preset styles** across multiple categories or upload your own PNG/JPG/JPEG reference.
- Adjust style strength (`alpha`) from `0.0` to `1.0`.
- Preserve facial details with a face-mask-guided blend after the neural style-transfer pass.

### Style library

- Preset metadata is stored in `static/styles/styles.json`.
- Full-size style images are stored in `static/styles/`; matching gallery thumbnails are in `static/styles/thumbs/`.
- The Flask and Streamlit interfaces load the manifest and can discover additional image files added to the styles directory.
- `tools/generate_styles.py` can regenerate the bundled illustrated style images and thumbnails.

### Flask application

- Account registration and login with hashed passwords.
- Password reset using time-limited, signed tokens; SMTP delivery is configurable.
- CSRF protection and rate limits on registration, login, and password-reset POST requests.
- User-specific upload directories and access-controlled image routes.
- Saved transfer history with content/style/output previews, downloads, deletion, pagination, and browser-side search/filter/sort controls.
- SQLite for local development and PostgreSQL support through `DATABASE_URL`.
- Gunicorn startup configuration in `Procfile`.

### Streamlit demo

- No login or database required.
- Preset gallery with category selection and custom style uploads.
- Adjustable processing size: 256, 384, or 512 pixels on the longest side.
- Optional face-preservation control and adjustable preservation strength.
- Before/after comparison slider, original/result tabs, and PNG download.
- Uploaded images are processed for the active session and are not added to the Flask history database.

## Architecture

### Inference pipeline

Both interfaces use the same model architecture and the same bundled inference checkpoint: `experiment/experiment200/decoder_192.pth`.

```mermaid
flowchart TD
    C[Content image] --> CP[Image validation and resize]
    S[Style reference] --> SP[Image validation and resize]
    CP --> E[Frozen VGG-19 feature encoder]
    SP --> E
    E --> CF[Content features]
    E --> SF[Style features]
    CF --> A[AdaIN: match channel-wise style statistics]
    SF --> A
    CF --> B[Blend with original content features using alpha]
    A --> B
    B --> D[Trained decoder]
    D --> ST[Stylized image]
    CP --> FM[Face-mask detection]
    FM --> FB[Feathered face-preservation blend]
    ST --> FB
    CP --> FB
    FB --> O[Final image]
```

1. **Prepare the images.** Each interface validates the image and converts it to RGB. Flask caps the inference copy at 256 pixels on its longest side. Streamlit lets the user select 256, 384, or 512 pixels. Aspect ratio is maintained.
2. **Extract features.** The same frozen VGG-19-based encoder processes the content and style images. The model exposes four feature levels; inference uses the deepest feature representation (`relu4_1`).
3. **Apply AdaIN.** The content feature map is normalized channel by channel, then scaled and shifted using the style feature map's channel-wise mean and standard deviation.
4. **Control style strength.** `alpha` blends the AdaIN-transformed features with the original content features.
5. **Decode.** The trained decoder reconstructs an RGB image from the blended feature map.
6. **Preserve the face when available.** A face mask is generated from the resized content image. A blurred mask controls how much of the original face is blended back into the stylized result.

### AdaIN and style strength

For content features \(x\) and style features \(y\), AdaIN is computed as:

\[
\operatorname{AdaIN}(x,y)=\sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\mu(y)
\]

The interface then blends the AdaIN result with the original content features:

\[
t=\alpha\,\operatorname{AdaIN}(x,y)+(1-\alpha)x
\]

The means and standard deviations are calculated per feature channel. A small epsilon is used when calculating standard deviations to avoid division by zero.

| `alpha` | Expected effect |
|---:|---|
| `0.0` | Keeps the original content feature representation; little or no style transfer. |
| `0.5` | Balances original content features and the AdaIN-transformed features. |
| `1.0` | Applies the full AdaIN style transformation. |

The control changes feature blending, not the amount of face preservation. Face preservation is a separate post-processing step.

### Face-preservation pipeline

The `face_preservation/` package contains two stages:

- **Mask generation — `face_parser.py`:** MediaPipe Face Mesh is tried first to create a polygon mask around the face oval. If MediaPipe is unavailable or does not return landmarks, the code falls back to OpenCV's Haar-cascade face detector and builds an elliptical mask around a detected face.
- **Blending — `blending.py`:** The mask is blurred to soften its boundary. The original content pixels and stylized pixels are blended inside the masked region according to `preserve_strength`.

The interfaces expose this behaviour differently:

| Interface | Behaviour in the current code |
|---|---|
| Flask | Face preservation runs automatically when the face-preservation modules import successfully and a face is detected. Strength is fixed at `0.7`, with a feather radius of `15` pixels. |
| Streamlit | The **Preserve facial features** checkbox is enabled by default when the modules are available. The user can turn it off and adjust preservation strength from `0.0` to `1.0` (default `0.7`). |

If the face detector finds no face, the neural style-transfer result is used without the additional blend. This is a lightweight face-region blending method, not a dedicated face-restoration or semantic face-parsing network; it can retain some original colour and texture in the face area.

### Training architecture

`train.py` trains the **decoder**, while keeping the VGG feature encoder frozen. For each training step it:

- extracts content and style features at four encoder levels;
- applies AdaIN to the deepest content/style feature maps;
- decodes the transformed features;
- compares the re-encoded generated image with the AdaIN target at the deepest feature level for the content loss;
- calculates style loss from feature means and standard deviations across the four feature levels;
- updates the decoder with Adam and steps a learning-rate scheduler.

The script writes periodic decoder checkpoints, optimizer states, preview images, and the run arguments. It does not run validation or automatically select a best checkpoint.

## Model files and active checkpoint

| File | Role |
|---|---|
| `vgg_normalised.pth` | Pretrained VGG-19 feature-encoder weights. |
| `experiment/experiment200/decoder_192.pth` | **Active inference decoder** loaded by both Flask and Streamlit. |
| `experiment/experiment200/args.txt` | Training arguments recorded for the `experiment200` run. |
| `experiment/experiment1/decoder_126.pth` and `decoder_128.pth` | Earlier decoder checkpoints. |
| `experiment/experiment1/optimizer_126.pth` and `optimizer_128.pth` | Matching optimizer states for the earlier checkpoints. |

**Checkpoint naming note:** the active file is `decoder_192.pth`. The accompanying `experiment200/args.txt` records a resumed run configured with `epochs: 200` and `start_epoch: 124`; the archive does not contain a `decoder_200.pth`. Both interfaces currently point to the available `decoder_192.pth` file.

You do not need to retrain for each new style reference. The model calculates the style features at inference time and reuses the same decoder.

## Repository structure

```text
.
├── app.py                          # Flask application, accounts, inference and history
├── train.py                        # Decoder training and resume workflow
├── requirements.txt                # Flask, training and model dependencies
├── Procfile                        # Gunicorn startup command
├── .python-version                 # Python 3.11.11
├── vgg_normalised.pth              # VGG feature-encoder weights
├── experiment/
│   ├── experiment1/                # Earlier checkpoints and training arguments
│   └── experiment200/              # Active decoder_192.pth and run arguments
├── face_preservation/
│   ├── face_parser.py              # MediaPipe/OpenCV face-mask generation
│   └── blending.py                 # Feathered original/stylized image blending
├── utils/
│   ├── models.py                   # VGGEncoder and Decoder definitions
│   └── utils.py                    # Dataset, transforms, AdaIN and statistics
├── templates/                      # Flask pages: login, register, transfer, history, reset
├── static/
│   └── styles/
│       ├── styles.json             # Preset style metadata
│       ├── *.jpg                   # Full-size preset style images
│       └── thumbs/                 # Gallery thumbnails
├── examples/                       # Sample content, style and output images
├── streamlit_demo/
│   ├── app.py                      # Streamlit interface
│   ├── requirements.txt            # Streamlit demo dependencies
│   └── README.md                   # Streamlit-specific setup
├── tools/
│   └── generate_styles.py          # Preset-style and thumbnail generation
└── code.ipynb                      # Notebook / experimentation workflow
```

## Requirements

- **Python 3.11.11** (the version specified in `.python-version`).
- pip and `venv`.
- The root environment uses **PyTorch 2.2.2** and **torchvision 0.17.2**.
- OpenCV and MediaPipe are included for face-mask generation. If face-preservation dependencies cannot be imported, the core style-transfer model can still run without that post-processing stage.
- An NVIDIA GPU is optional. CPU inference is supported but may be slower. Training is substantially more practical on a GPU.

Use Python 3.11 for this pinned environment. Newer Python versions may not have compatible wheels for the pinned PyTorch release.

## Run the Flask application

Run these commands from the repository root (the folder containing `app.py` and `requirements.txt`).

### 1. Create a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use a permitted shell or activate the environment using the appropriate command for your shell. You do not need to change a system-wide execution policy just to install the project.

**Linux / macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

For a CPU-only environment, install the CPU wheels explicitly and then install the remaining project requirements:

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the matching CUDA-enabled PyTorch pair first, then install the remaining project requirements:

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The CUDA 12.1 wheel command is one available build for this pinned PyTorch version. Choose a compatible official PyTorch wheel for your operating system and driver. The CUDA version printed by `nvidia-smi` describes driver compatibility; it does not mean the full CUDA Toolkit must necessarily be installed separately.

Check your installation:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')"
```

### 3. Configure Flask

The Flask app requires `SECRET_KEY`. Generate one with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Create a `.env` file in the project root:

```dotenv
SECRET_KEY=replace_with_the_generated_random_value
APP_BASE_URL=http://127.0.0.1:5000

# Leave DATABASE_URL unset to use local SQLite at instance/app.db.
# For PostgreSQL, use a connection URL such as:
# DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME

# Optional SMTP configuration for password-reset emails:
# MAIL_SERVER=smtp.example.com
# MAIL_PORT=587
# MAIL_USE_TLS=1
# MAIL_USE_SSL=0
# MAIL_USERNAME=your-smtp-username
# MAIL_PASSWORD=your-smtp-password
# MAIL_FROM=StyleForge AI <no-reply@example.com>
```

Keep `.env` private. It is excluded by `.gitignore`; never commit real secrets or database credentials.

**Database behaviour:** the app reads `DATABASE_URL`. If it is absent, the app creates/uses `instance/app.db` with SQLite and logs an informational warning. For PostgreSQL, set `DATABASE_URL`; common `postgres://` and `postgresql://` URLs are normalized to the `psycopg` v3 driver. The app configures `SQLALCHEMY_DATABASE_URI` internally, so set the environment variable `DATABASE_URL` rather than trying to set the Flask config key directly.

Database tables are created automatically on startup unless `AUTO_CREATE_DB=0`. Flask-Migrate is initialized in the application, but a production deployment should use an intentionally managed migration process and persistent database storage.

**Password-reset email:** without `MAIL_SERVER`, reset messages are written to the application log and are not sent. Configure the SMTP values and set `APP_BASE_URL` to the public application URL for deployed reset links. Use either STARTTLS or SSL, not both.

### 4. Start the application

```bash
python app.py
```

Open **http://127.0.0.1:5000**, register an account, then select a preset or upload a style reference and provide a content image.

The Flask interface limits the inference copy of each image to **256 pixels on its longest side** to reduce memory use. The original uploaded files are retained for the user's history/downloads, but the generated inference result is based on the resized image. The request upload limit is 10 MB.

## Run the Streamlit demo

The Streamlit interface is independent of Flask. It does not use Flask accounts, the Flask database, or the persistent Flask history workflow.

For a clean setup, create a separate environment from the project root.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv-streamlit
.\.venv-streamlit\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app.py
```

**Linux / macOS**

```bash
python3.11 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python -m pip install --upgrade pip
python -m pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app.py
```

The Streamlit requirements file configures CPU-only PyTorch wheels by default. For CUDA, install a compatible CUDA-enabled PyTorch/torchvision pair in this environment before launching the app.

The demo allows processing sizes of **256, 384, or 512 pixels** on the longest side, preserving aspect ratio. The combined content and style upload limit is 10 MB. Larger processing sizes can retain more detail but require more memory and processing time.

### Deploy the Streamlit demo

The repository is structured for Streamlit Community Cloud deployment:

1. Push the project, including the model weights and `static/styles/` assets, to GitHub.
2. Create a Streamlit app from the repository.
3. Set the main file to `streamlit_demo/app.py`.
4. Use Python 3.11 and the dependency file `streamlit_demo/requirements.txt`.

The application expects the model weights at the paths already defined in `streamlit_demo/app.py`, relative to the repository root. Do not remove `vgg_normalised.pth`, `experiment/experiment200/decoder_192.pth`, `static/styles/`, or `examples/` from the deployed project.

Live app: **[StyleForge AI on Streamlit](https://adain-neural-style-transfer-39tfl8u4t9pp9vbieaksv6.streamlit.app/)**

## Flask routes

The Flask application is server-rendered with an AJAX-enhanced transfer form; it is not a standalone public REST API.

| Route | Access | Purpose |
|---|---|---|
| `/register` | Public | Create an account. |
| `/login` | Public | Sign in. |
| `/forgot-password` | Public | Request a password-reset link. |
| `/reset-password/<token>` | Public; valid signed token required | Set a new password. |
| `/` | Login required | Upload/select content and style images and run inference. |
| `/uploads/<filename>` | Login required; current-user file only | Serve an image from the signed-in user's upload directory. |
| `/history` | Login required | Browse, search, sort, and filter saved transfers. |
| `/history/<transfer_id>/delete` | Login required; POST | Delete a transfer and images no longer referenced by another transfer. |
| `/logout` | Login required; POST | End the session. |
| `/examples/<path:filename>` | Public | Serve bundled example images. |

Authentication and password-reset POST routes have rate limits. Forms use Flask-WTF CSRF protection. See `app.py` for the precise limits and implementation.

## Add or regenerate preset styles

The style library is driven by `static/styles/styles.json`. Each entry contains an `id`, `name`, `category`, `description`, and image `file`.

To add a style manually:

1. Add a `.jpg`, `.jpeg`, or `.png` image to `static/styles/`.
2. Add its metadata to `styles.json` if you want a custom display name, category, and description.
3. Optionally add a thumbnail with the same filename to `static/styles/thumbs/`.

Images that are not listed in the manifest are discovered automatically and placed in the **More** category. To regenerate the bundled illustrated presets and thumbnails, run:

```bash
python tools/generate_styles.py
```

Run the script from the project root in an environment with the project's Pillow and NumPy dependencies installed.

## Train or resume the decoder (optional)

Training is **not required** to run either interface. Use `train.py` only when you want to train a new decoder or continue a compatible training run. Training datasets are not included in this repository.

### Dataset layout

`ImageFolderdataset` reads supported image files directly from each folder; it does not recursively scan subfolders.

```text
 data/
 ├── content/
 │   ├── content_001.jpg
 │   ├── content_002.jpg
 │   └── ...
 └── style/
     ├── style_001.jpg
     ├── style_002.jpg
     └── ...
```

Supported formats are `.jpg`, `.jpeg`, and `.png`. Each dataset must contain at least one complete batch because the data loaders use `drop_last=True`.

### Start a new experiment

Some defaults in `train.py` are developer-specific Windows paths. Always pass your own dataset and VGG paths.

```bash
python train.py \
  --content_dir "data/content" \
  --style_dir "data/style" \
  --vgg "vgg_normalised.pth" \
  --experiment "experiment2" \
  --batch_size 4 \
  --epochs 200
```

This example writes to `experiment/experiment2/`. Batch size `4` is a conservative starting point for a GPU with limited VRAM; increase it only when memory allows. The script defaults to batch size `16`, final crop size `256`, content/style resize size `512`, learning rate `1e-4`, content weight `1.0`, style weight `5.0`, and checkpoint saves every two epochs.

### Resume from a checkpoint

Use a matching decoder and optimizer checkpoint from the same epoch. For example, the archive includes `decoder_128.pth` and `optimizer_128.pth` in `experiment/experiment1/`:

```bash
python train.py \
  --content_dir "data/content" \
  --style_dir "data/style" \
  --vgg "vgg_normalised.pth" \
  --experiment "experiment2" \
  --resume \
  --start_epoch 128 \
  --epochs 200 \
  --decoder_path "experiment/experiment1/decoder_128.pth" \
  --optimizer_path "experiment/experiment1/optimizer_128.pth"
```

`--start_epoch` is the number of epochs already completed; set it to match the checkpoint filename. `--epochs` is the target total epoch count, not the number of additional epochs. The training code restores the decoder and optimizer state, but it does not restore a saved scheduler state, so the learning-rate schedule is not an exact continuation of the original run.

### Checkpoint outputs and using a newly trained decoder

At the configured save interval, the script writes:

- `decoder_<epoch>.pth` — decoder weights;
- `optimizer_<epoch>.pth` — optimizer state;
- `output_<epoch>.png` — a training preview;
- `args.txt` — the run configuration.

The script saves periodic checkpoints; it does not evaluate a validation set or automatically select the best checkpoint.

To use a new decoder in the web apps, update the path in **both** places and restart the application:

- Flask: model-loading section in `app.py`.
- Streamlit: `DECODER_PATH` near the top of `streamlit_demo/app.py`.

Both currently load `experiment/experiment200/decoder_192.pth`. Keep the matching `vgg_normalised.pth` available as well.

## Deployment notes for Flask

The root `Procfile` starts Flask using Gunicorn:

```text
web: gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 180 app:app
```

Configure these values in your hosting provider's environment settings:

- `SECRET_KEY` — unique, random secret; required.
- `DATABASE_URL` — recommended for a managed PostgreSQL database in production.
- SMTP variables — required only if password-reset emails should be delivered.
- Include both model weight files and the `static/styles/` assets in the deployed build.

Operational considerations:

- **Persistent storage:** user uploads and generated images are stored under `instance/user_uploads/`; the SQLite database is under `instance/app.db`. An ephemeral filesystem can lose these files on redeploy or restart. Use persistent storage or external object storage for user files, and a persistent database for account/history records.
- **Memory:** each Gunicorn worker loads the VGG encoder and decoder into its own process. Keep the single-worker setup unless the host has enough memory for additional model copies. The 256-pixel Flask inference cap reduces processing memory but does not eliminate model-loading overhead.
- **Rate-limit storage:** the default limiter store is in memory. Use a shared store such as Redis through `RATELIMIT_STORAGE_URI` when running multiple workers or instances.
- **Reverse proxy:** set `TRUSTED_PROXY=1` only when the app is behind exactly one trusted proxy, as expected by the `ProxyFix` configuration.
- **Secrets and HTTPS:** store credentials in the hosting provider's secret manager, terminate HTTPS at the provider or a trusted reverse proxy, and never commit secrets.

## Environment variables

| Variable | Default / requirement | Purpose |
|---|---|---|
| `SECRET_KEY` | **Required** | Flask session signing and password-reset tokens. |
| `DATABASE_URL` | Local SQLite if unset | SQLAlchemy database URL; PostgreSQL is recommended for persistent production use. |
| `APP_BASE_URL` | Empty | Public base URL used to construct password-reset links. |
| `RESET_TOKEN_MAX_AGE` | `3600` seconds | Password-reset token lifetime. |
| `MAIL_SERVER` | Unset | SMTP server; without it, reset messages are logged and not sent. |
| `MAIL_PORT` | `587` | SMTP port. |
| `MAIL_USERNAME` / `MAIL_PASSWORD` | Unset | Optional SMTP authentication credentials. |
| `MAIL_USE_TLS` | `1` | Enables STARTTLS unless SSL mode is selected. |
| `MAIL_USE_SSL` | `0` | Enables implicit SMTP-over-SSL when set to `1`. |
| `MAIL_FROM` | Username or local fallback | Sender address for application emails. |
| `RATELIMIT_STORAGE_URI` | `memory://` | Flask-Limiter storage; use a shared store for multiple instances. |
| `TRUSTED_PROXY` | `0` | Trust forwarding headers from one configured proxy when set to `1`. |
| `AUTO_CREATE_DB` | `1` | Calls `db.create_all()` during startup when enabled. |
| `FLASK_DEBUG` | `0` | Enables Flask debug mode when running `python app.py`; keep disabled in production. |

## Troubleshooting

### `RuntimeError: Set SECRET_KEY in your .env file.`

Create `.env` beside `app.py`, set a strong `SECRET_KEY`, and restart the process. Confirm the file is named `.env`, not `.env.txt`.

### `DATABASE_URL not set; using local SQLite database.`

This is an informational warning. The application is using `instance/app.db`. Set `DATABASE_URL` if you intended to connect to PostgreSQL.

### `torch.cuda.is_available()` is `False`

The application can run on CPU. If you expected GPU acceleration, verify that the active environment has a CUDA-enabled PyTorch build and a compatible NVIDIA driver. Check the PyTorch and torchvision versions together.

### `torchvision::nms does not exist` or native DLL/import errors

This often indicates incompatible or mixed PyTorch and torchvision installations. Recreate the virtual environment and install the matching pair specified for this project (`torch==2.2.2` and `torchvision==0.17.2`). On managed Windows machines, application-control policies may separately block native DLLs; changing package versions may not resolve that policy restriction.

### The application cannot find a model checkpoint

Check that both of these files exist relative to the repository root:

```text
vgg_normalised.pth
experiment/experiment200/decoder_192.pth
```

The current paths are defined in `app.py` and `streamlit_demo/app.py`. Update those paths if you move the checkpoint files.

### Password-reset email does not arrive

Without `MAIL_SERVER`, the app logs the reset message but does not send it. Configure SMTP values and `APP_BASE_URL`, then inspect application logs and your email provider's SMTP requirements.

### The face is not preserved

Face blending only runs when the face-preservation modules load and a face mask is detected. Ensure OpenCV is installed; MediaPipe is attempted first, with an OpenCV Haar-cascade fallback. Use a clear, front-facing image with the face visible. If the detector returns no mask, the app uses the unblended stylized output.

### The output looks too abstract or loses detail

Try reducing `alpha`, using a less busy style reference, or increasing the Streamlit processing size. The Flask app is intentionally capped at 256 pixels on the longest side. AdaIN transfers feature statistics and is not a semantic image editor or face-restoration model.

## Limitations, privacy, and licensing

- Neural style transfer does not understand object boundaries or guarantee preservation of every facial feature.
- Face preservation blends pixels from the resized original face region; it is not a learned identity-preservation model.
- Flask stores uploaded images and outputs on the server in per-user directories. Configure storage, access controls, backups, and retention appropriately before exposing the application publicly.
- Streamlit does not provide accounts or persistent history; uploaded content is used for the active inference session and is not written to the Flask database.
- Training datasets are not bundled. Supply your own content and style folders if you train a decoder.
- No project-level `LICENSE` file is included in this archive. Add a license before inviting external reuse or contributions. Separately check the applicable licenses and rights for the VGG weights, decoder checkpoint, bundled sample images, preset style images, and third-party assets. A source-code license does not automatically cover model weights or image assets.

## Reference

The method is based on:

> Xun Huang and Serge Belongie. **“Arbitrary Style Transfer in Real-Time With Adaptive Instance Normalization.”** Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.

- Paper: <https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html>
- arXiv: <https://arxiv.org/abs/1703.06868>

Please cite the original paper when using this method in academic work.

---

<div align="center">
Built with PyTorch, VGG-19, Flask, Streamlit, and Adaptive Instance Normalization.
</div>
