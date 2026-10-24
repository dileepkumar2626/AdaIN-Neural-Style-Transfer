<div align="center">

# StyleForge AI
### Arbitrary Neural Style Transfer with Adaptive Instance Normalization

Turn a photograph into a new visual style using PyTorch, a frozen VGG feature encoder, and a trained decoder. Use the included browser apps to try preset styles or upload your own reference image.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Web-Flask-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)

</div>

---

## Overview

**StyleForge AI** is an image-to-image application based on **Adaptive Instance Normalization (AdaIN)**. It separates the *content* of an image (the objects, scene, and broad composition) from the *style* of a reference image (such as its colour palette, textures, and brushwork), then combines them in a stylized output.

The repository contains the inference weights and two interfaces:

- **Flask web application** — account registration and login, password-reset flow, an interactive style-transfer page, a preset style gallery, and per-account transfer history.
- **Streamlit demo** — a lighter interface for quickly trying the model, comparing original and stylized images, and downloading results. It does not use the Flask accounts or database.

The pretrained VGG weights and decoder checkpoint are included in the repository archive, so **you can run inference without training the model again**.

## Example results

The repository includes sample content images, style references, and stylized outputs. These examples are local files, so they also render when the repository is viewed offline in an editor that supports Markdown previews.

| Content image | Style reference | Example output |
|---|---|---|
| ![Portrait content image](examples/brad%20pitt.jpg) | ![Sketch reference](examples/sketch.png) | ![Portrait rendered in sketch style](examples/stylized_brad_pitt.jpg) |
| ![Portrait content image](examples/brad%20pitt.jpg) | ![Painting reference](examples/picasso%20seated%20nude%20hr.jpg) | ![Portrait with transferred painting style](examples/stylized_brad%20pitt%20(1).jpg) |

## Features

### Image stylization

- Transform a content photo using a separate style-reference image.
- Choose from the bundled preset style gallery or upload your own PNG/JPG/JPEG style image.
- Adjust style strength with the `alpha` value (also displayed as **Style Strength** in the UI).
- Preview the generated image, compare it with the original, and download the result.
- Reuse the last uploaded images in the Flask interface to try another preset or strength without selecting both files again.

### Flask application

- Registration and login, with password hashes rather than plaintext passwords.
- Password-reset tokens with an expiration time; SMTP delivery can be configured through environment variables.
- CSRF protection on form submissions and rate limits for authentication-related endpoints.
- Per-user image storage and protected image routes.
- A transfer-history page with search, sorting, strength filters, downloads, and deletion.
- SQLite for a local setup, with PostgreSQL support through `DATABASE_URL`.
- A Gunicorn `Procfile` suitable as a starting point for Linux-based hosting.

### Streamlit demo

- A simple interactive workflow without account setup or a database.
- Built-in preset style selection and custom style uploads.
- A before/after comparison slider and PNG download.
- Selectable processing size: 256, 384, or 512 pixels on the longest side.

## Flask routes at a glance

The Flask app is a server-rendered web application with an AJAX-enhanced transfer form; it does not expose a separate public REST API. The main transfer POST can return JSON for the browser interface.

| Route | Access | Purpose |
|---|---|---|
| `/register` | Public | Create an account. |
| `/login` | Public | Sign in. |
| `/forgot-password` | Public | Request a password-reset email. |
| `/reset-password/<token>` | Public, signed token required | Set a new password through a valid reset link. |
| `/` | Login required | Upload/select content and style images and run inference. |
| `/uploads/<filename>` | Login required; file ownership checked | Serve the current user's saved images and results. |
| `/history` | Login required | Browse, filter, download, and delete previous transfers. |
| `/history/<transfer_id>/delete` | Login required; POST | Delete a transfer record and images no longer used by another transfer. |
| `/logout` | Login required; POST | End the session. |
| `/examples/<path:filename>` | Public | Serve the bundled demonstration images. |

Forms are protected by Flask-WTF CSRF protection. Authentication and password-reset POST endpoints have rate limits; see `app.py` for the exact values.

## How the model works

The app uses **Adaptive Instance Normalization (AdaIN)** to transfer the visual style of one image onto another. It uses a frozen VGG-19 encoder to extract features, applies AdaIN to transfer the style-image statistics, and then uses the trained decoder to turn those features back into an image.

### Processing pipeline

The diagram below uses GitHub-supported Mermaid syntax. The explanation and formulas underneath it remain readable even in Markdown viewers that do not render Mermaid.

```mermaid
flowchart TD
    C["Content image"] --> CE["VGG-19 encoder: content features"]
    S["Style reference image"] --> SE["VGG-19 encoder: style features"]
    CE --> A["AdaIN: transfer style statistics"]
    SE --> A
    CE --> B["Blend features using alpha"]
    A --> B
    B --> D["Trained decoder"]
    D --> O["Stylized output image"]
```

### Step by step

1. **Encode both images.** The pretrained VGG-19 encoder extracts feature maps from the content photograph and the style reference. The encoder is frozen; inference does not update its weights.
2. **Transfer the style statistics.** AdaIN normalizes the content features using the content image's channel-wise mean and standard deviation, then scales them using the style features' statistics.
3. **Set the style strength.** The `alpha` value blends the style-transformed features with the original content features.
4. **Decode the result.** The trained decoder converts the blended feature maps into the final stylized image.

The core computation can be expressed without special equation formatting:

```python
content_normalized = (content_features - content_mean) / content_std
adain_features = content_normalized * style_std + style_mean
blended_features = alpha * adain_features + (1 - alpha) * content_features
stylized_image = decoder(blended_features)
```

The means and standard deviations are calculated per feature channel. A small epsilon is added when calculating standard deviations to avoid division by zero.

### What the style-strength control means

| `alpha` value | Effect |
|---|---|
| `0.0` | Uses the original content features, so the output should remain close to the content image. |
| `0.5` | Blends the original content features with the AdaIN-transformed features. |
| `1.0` | Uses the full AdaIN style transformation. |

Values between `0.0` and `1.0` provide a continuous trade-off between the original content appearance and the transferred style. The Flask app validates that `alpha` stays within this range.

**You do not need to retrain the model for a new style image.** At inference time, the encoder extracts the new style image's feature statistics and the existing decoder generates the result. Training is only needed when you want to train or replace the decoder weights.

### Included model files

| File | Purpose |
|---|---|
| `vgg_normalised.pth` | Weights loaded by the VGG feature encoder. |
| `experiment/experiment1/decoder_128.pth` | Decoder checkpoint used by both web interfaces. |
| `experiment/experiment1/optimizer_128.pth` | Optimizer state available if continuing a compatible training run. |
| `experiment/experiment1/args.txt` | Recorded arguments for the bundled experiment checkpoint. |

The Flask interface loads the VGG encoder and decoder when `app.py` is imported. The Streamlit interface loads and caches the same model resources when its app starts.

## Repository layout

```text
.
├── app.py                         # Main Flask application
├── train.py                       # Decoder training script
├── requirements.txt               # Flask app / training dependencies
├── Procfile                       # Gunicorn startup command
├── .python-version                # Python 3.11.11
├── vgg_normalised.pth             # VGG encoder weights
├── experiment/
│   └── experiment1/
│       ├── decoder_128.pth        # Inference decoder checkpoint
│       ├── optimizer_128.pth      # Optimizer checkpoint
│       └── args.txt               # Recorded training parameters
├── utils/
│   ├── models.py                  # VGGEncoder and Decoder modules
│   └── utils.py                   # Dataset, transforms, AdaIN, feature statistics
├── templates/                     # Flask HTML templates
├── static/styles/                 # Preset styles, manifest, and thumbnails
├── examples/                      # Sample content/style/output images
├── streamlit_demo/
│   ├── app.py                     # Streamlit interface
│   ├── requirements.txt           # Streamlit demo dependencies
│   └── README.md                  # Streamlit-specific notes
├── tools/
│   └── generate_styles.py         # Generates the illustrated preset styles
└── code.ipynb                     # Notebook / experimentation workflow
```

## Requirements

- Python **3.11** (the repository specifies `3.11.11` in `.python-version`).
- pip and a virtual environment.
- PyTorch and torchvision versions compatible with each other. The main requirements file pins `torch==2.2.2` and `torchvision==0.17.2`.
- Optional: a supported NVIDIA GPU and compatible NVIDIA driver for CUDA acceleration. CPU inference is also supported, but can be slower.

## Run the Flask web application

The following commands are run from the project root—the directory containing `app.py` and `requirements.txt`.

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation because of the local execution policy, use a permitted shell or activate the environment with the appropriate command for your machine. Do not change a system-wide policy unless you understand the implications.

**Linux / macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch for your machine

Run **one** of the following before installing the remaining requirements.

**CPU-only**

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1 wheel build)**

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

Use a PyTorch build compatible with your Python version and NVIDIA driver. The CUDA version shown by `nvidia-smi` is the maximum CUDA runtime supported by the installed driver; it does not by itself mean that the matching CUDA toolkit must be installed system-wide.

### 3. Install project dependencies

```bash
python -m pip install -r requirements.txt
```

Confirm that PyTorch imports and check whether CUDA is available:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')"
```

### 4. Configure the environment

The Flask app requires `SECRET_KEY`. It loads a `.env` file from the project root when present. Generate a strong key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Create a file named `.env` in the project root:

```dotenv
SECRET_KEY=replace_this_with_the_generated_random_value
APP_BASE_URL=http://127.0.0.1:5000

# Leave DATABASE_URL unset for local SQLite at instance/app.db.
# For PostgreSQL, use a URL such as:
# DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME

# Optional SMTP settings enable real password-reset email delivery:
# MAIL_SERVER=smtp.example.com
# MAIL_PORT=587
# MAIL_USE_TLS=1
# MAIL_USE_SSL=0
# MAIL_USERNAME=your-smtp-username
# MAIL_PASSWORD=your-smtp-password
# MAIL_FROM=StyleForge AI <no-reply@example.com>
```

Keep `.env` private and do not commit secrets. The `.gitignore` already excludes `.env` files.

**Database note:** the app reads `DATABASE_URL`. If it is not set, it creates/uses a local SQLite database at `instance/app.db` and logs a warning. For local development, that warning is informational. When using PostgreSQL, set `DATABASE_URL` to your database connection string; this app normalizes common `postgres://` and `postgresql://` URLs to use the `psycopg` v3 driver. Do not use `SQLALCHEMY_DATABASE_URI` as a substitute for the environment variable—the code configures that Flask-SQLAlchemy setting internally.

On startup, the application creates its tables automatically while `AUTO_CREATE_DB` is unset or set to `1` (the default). For a production deployment, plan a controlled database schema migration strategy instead of relying indefinitely on automatic table creation.

**Password reset note:** without `MAIL_SERVER`, the app will log the reset email contents for development and will not actually send email. To enable email delivery, configure the SMTP variables above. Use `APP_BASE_URL` to set the public base URL used in links sent by email. Configure either TLS or SSL according to your mail provider; do not enable both at the same time.

### 5. Start Flask

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, register an account, then upload a content image and select or upload a style reference. In this app, both uploaded images together must be **under 10 MB**. Supported extensions are PNG, JPG, and JPEG.

The Flask app resizes each inference image so its **longest side is at most 256 pixels**, preserving the aspect ratio to constrain memory use on smaller hosts. This keeps processing lightweight, but the generated output is not full camera resolution.

## Run the Streamlit demo

The Streamlit interface is an alternative front end; it does not use Flask login, Flask history, or the Flask database. It processes the images for the current session and does not persist uploaded images in the application database.

For a clean environment, create a separate virtual environment from the repository root.

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

Open the local URL printed by Streamlit in the terminal. The Streamlit requirements file is configured for CPU-only PyTorch wheels by default. If you want to use CUDA in this separate environment, install the appropriate CUDA-enabled PyTorch/torchvision pair for your platform and check the result before launching the app.

The demo supports a processing size of **256, 384, or 512 pixels on the longest side**, with a combined upload limit of 10 MB. A larger processing size can preserve more detail but takes more time and memory.

## Add or regenerate preset styles

The preset gallery is read from `static/styles/styles.json`. Image files placed directly in `static/styles/` with a `.jpg`, `.jpeg`, or `.png` extension are also discovered automatically and shown under the fallback **More** category if they are not listed in the manifest. Place optional thumbnails in `static/styles/thumbs/` using the same filename. To edit a preset's name, category, or description, update its entry in `styles.json`.

The bundled `tools/generate_styles.py` script can regenerate the illustrated presets and their thumbnails. Run it from the project root in an environment with the project's Pillow and NumPy dependencies installed:

```bash
python tools/generate_styles.py
```

## Training the decoder (optional)

Training is **not required** to run either interface with the included checkpoints. Use `train.py` only if you want to train a new decoder or continue an experiment.

### Dataset layout

`ImageFolderdataset` reads image files directly inside each directory; it does not recursively scan subdirectories. Arrange your images like this:

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

Use supported image files (`.jpg`, `.jpeg`, `.png`). Both folders must contain enough images to produce at least one complete batch because the training data loaders use `drop_last=True`.

### Start a new training run

Some defaults in `train.py` point to a developer-specific Windows path. Always provide your own dataset and VGG paths rather than relying on those defaults.

```bash
python train.py --content_dir "data/content" --style_dir "data/style" --vgg "vgg_normalised.pth" --experiment "experiment2" --batch_size 4 --epochs 200
```

This command uses the script's defaults for image transforms and loss weights, while saving the new experiment under `experiment/experiment2/`. A batch size of 4 is a cautious starting point for a GPU with limited VRAM; adjust it according to available memory. CPU training is supported but can be substantially slower.

Important training behaviour:

- The VGG encoder is frozen; the decoder is optimized with Adam.
- The script combines a content reconstruction loss with a style-statistics loss based on feature means and standard deviations.
- Checkpoints and training preview images are saved every `--save_interval` epochs (default: every 2 epochs).
- The script writes periodic files named `decoder_<epoch>.pth`, `optimizer_<epoch>.pth`, and `output_<epoch>.png`, along with `args.txt`.
- The script saves periodic checkpoints; it does not evaluate a validation set or automatically select a best checkpoint.

### Resume an experiment

Supply matching decoder and optimizer files from the experiment you want to continue:

```bash
python train.py --content_dir "data/content" --style_dir "data/style" --vgg "vgg_normalised.pth" --experiment "experiment2" --resume --decoder_path "experiment/experiment2/decoder_100.pth" --optimizer_path "experiment/experiment2/optimizer_100.pth"
```

The training script restores the decoder and optimizer weights, but it does not save/restore the learning-rate scheduler state, so a resumed run restarts the scheduler from its initial epoch state.

### Use a newly trained decoder in the app

Both interfaces currently point to `experiment/experiment1/decoder_128.pth`:

- **Flask:** update the `torch.load(...)` path near the model initialization in `app.py`.
- **Streamlit:** update `DECODER_PATH` near the top of `streamlit_demo/app.py`.

Point the relevant interface to the new experiment's decoder checkpoint and restart the application. Keep `vgg_normalised.pth` available at the path expected by the app.

## Deployment notes

The root `Procfile` starts the Flask application with Gunicorn:

```text
web: gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 180 app:app
```

For a Linux-based host that supports a `Procfile`, configure at least:

- `SECRET_KEY` — a unique, strong secret.
- `DATABASE_URL` — recommended for a managed PostgreSQL database in production.
- SMTP variables if password-reset emails should be delivered.
- The model weights and `static/styles/` assets must be included in the deployed build.

Operational considerations:

- **Persistent files:** Flask uploads and generated images are stored under `instance/user_uploads/`, while the SQLite database is under `instance/app.db`. Ephemeral hosting filesystems can lose these between deploys or restarts. Use persistent storage or move user files to an external object store before depending on long-term history.
- **Database:** the built-in SQLite fallback is convenient for local development, not a substitute for a managed/persistent production database.
- **Memory:** the model is loaded into each Python worker process. Start with the single-worker command in the `Procfile`; adding workers or replicas increases model memory use.
- **Rate limits:** the default limiter store is in memory. If running multiple workers or instances, configure a shared backend such as Redis with `RATELIMIT_STORAGE_URI`.
- **Proxy:** set `TRUSTED_PROXY=1` only when the deployment is actually behind exactly one trusted proxy, as described in the comments in `app.py`.
- **HTTPS and secrets:** terminate HTTPS at your hosting provider or trusted reverse proxy, keep secrets in the provider's secret/environment settings, and never put credentials in source control.

## Useful environment variables

| Variable | Default / required | Purpose |
|---|---|---|
| `SECRET_KEY` | **Required** | Flask session signing and password-reset tokens. |
| `DATABASE_URL` | Local SQLite if unset | SQLAlchemy database URL; use PostgreSQL for a persistent production database. |
| `APP_BASE_URL` | Empty | Public base URL used to construct password-reset links. |
| `RESET_TOKEN_MAX_AGE` | `3600` seconds | Password-reset token lifetime. |
| `MAIL_SERVER` | Unset | SMTP server; without it, reset email is logged rather than sent. |
| `MAIL_PORT` | `587` | SMTP port. |
| `MAIL_USERNAME` / `MAIL_PASSWORD` | Unset | Optional SMTP authentication credentials. |
| `MAIL_USE_TLS` | `1` | Enable STARTTLS unless SSL mode is used. |
| `MAIL_USE_SSL` | `0` | Use implicit SMTP-over-SSL when set to `1`. |
| `MAIL_FROM` | Username or a local fallback | Sender address for application emails. |
| `RATELIMIT_STORAGE_URI` | `memory://` | Flask-Limiter storage URI; use shared Redis for multiple instances. |
| `TRUSTED_PROXY` | `0` | Trust forwarding headers from exactly one configured proxy when set to `1`. |
| `AUTO_CREATE_DB` | `1` | Call `db.create_all()` during application startup when enabled. |
| `FLASK_DEBUG` | `0` | Enables Flask debug mode when running `python app.py`; keep off in production. |

## Troubleshooting

### `RuntimeError: Set SECRET_KEY in your .env file.`

Create `.env` in the same directory as `app.py`, set `SECRET_KEY`, and restart the process. Ensure the file is named `.env`, not `.env.txt`.

### `DATABASE_URL not set; using local SQLite database.`

This is an informational warning from the current application. It means the app is falling back to `instance/app.db`. Set `DATABASE_URL` if you intended to use PostgreSQL.

### `torch.cuda.is_available()` is `False`

The app can still run on CPU. If you expected CUDA, confirm that the correct NVIDIA GPU build of PyTorch is installed in the active virtual environment and that the driver supports it. Check the printed `torch` and `torchvision` versions together; mismatched versions can cause import/runtime errors.

### `torchvision::nms does not exist` or DLL/import errors

This commonly points to incompatible or mixed PyTorch/torchvision installations. In a clean virtual environment, reinstall the matching pair specified for this project (`torch==2.2.2` and `torchvision==0.17.2`) from the same official PyTorch wheel index. On managed Windows machines, an application-control policy may independently block native DLLs; changing PyTorch versions will not necessarily override such a policy.

### The app cannot find a model file

Check that both files exist at the paths used by the interface:

- `vgg_normalised.pth`
- `experiment/experiment1/decoder_128.pth`

Run the app from the repository's expected layout, or update the model paths in the Python files when you relocate the checkpoints.

### Password reset does not arrive by email

If `MAIL_SERVER` is unset, the app deliberately does not send email; it logs the message instead. Configure the SMTP values and `APP_BASE_URL`, then check the application logs and your provider's SMTP requirements.

### The output looks too abstract or facial details are lost

AdaIN transfers statistics across feature maps; it is not a face-aware restoration model. Try a lower style strength, a less busy style reference, and a sharp, well-lit content image. Lower processing sizes are faster but retain less fine detail.

## Limitations and privacy notes

- This is neural style transfer, not a semantic image editor. It does not understand object boundaries or preserve every facial feature.
- The Flask interface intentionally downsizes inference inputs for memory efficiency. The result is not a full-resolution export.
- The Streamlit demo does not provide user accounts or a persistent history. Its uploads/results are handled for the active session and are not used to train the model.
- The Flask app stores uploads and outputs on the server under per-user directories. Protect the host, configure persistent storage deliberately, and review your data-retention needs before exposing it publicly.
- Training data is not bundled as part of this application workflow. Provide your own content/style datasets if you choose to train.

## Reference

The method is based on the following paper:

> Xun Huang and Serge Belongie. **“Arbitrary Style Transfer in Real-Time With Adaptive Instance Normalization.”** Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
>
> Paper: https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html  
> arXiv: https://arxiv.org/abs/1703.06868

Please cite the original paper when using this method in academic work.

## License and asset attribution

No project-level `LICENSE` file was present in the supplied repository archive. Add an explicit license before inviting external reuse or contributions. Separately verify the applicable licenses and usage rights for the pretrained VGG weights, decoder checkpoint, sample images, preset style images, and any third-party assets. A license for the source code does not automatically grant rights to every bundled image or model weight.

---

<div align="center">
Built with PyTorch, Flask, Streamlit, and Adaptive Instance Normalization.
</div>
