
<div align="center">

# 🎨 StyleForge AI — AdaIN Neural Style Transfer

**Turn any photo into art in real time. Upload a content image, pick a style, and watch a neural network repaint it.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-ready-4169E1?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## ✨ Overview

This project is a full-stack web app built around **Adaptive Instance Normalization (AdaIN)**, the arbitrary style transfer method from
[*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868) (Huang & Belongie, ICCV 2017).

Unlike older style transfer methods that need a new model for every style, AdaIN works with **any** style image in a single forward pass, so results come back in seconds, and you can blend content and style with a strength slider.

On top of the model there is a complete, interactive Flask application with user accounts, a personal history, and a polished UI.

> 📸 *Add screenshots here: home page, before/after slider, history page.*
> Example: `![Home](docs/home.png)`

---

## 🚀 Features

### 🖼️ Style transfer
- **Any content + any style:** no retraining needed.
- **Style strength (alpha) slider (0 → 1):** smoothly blend between the original photo and full stylization.
- **Fast inference** on CPU, faster on GPU (CUDA is detected automatically).

### 🖱️ Interactive interface
- **Drag-and-drop uploads** with instant previews, validation (PNG/JPG, 10 MB total) and one-click removal.
- **No page reloads:** live upload progress bar, status messages and a cancel button.
- **Before/after comparison slider** (mouse, touch and keyboard) with Compare / Stylized / Original views.
- **Click-to-enlarge** lightbox, download button, toast notifications and smooth animations.
- **Reuse uploaded images:** tweak the strength and re-run without uploading again.

### 👤 Accounts and history
- Register / login with hashed passwords and CSRF protection.
- **Personal history** of every transfer with search, strength filter, sorting and pagination.
- **Gallery viewer** with previous/next (arrow keys and swipe) and Content / Style / Result tabs.
- **Delete with confirmation:** removes the database record and its files (shared input images are kept while another record still uses them).
- Each user can only see and delete their own images.

### 🛡️ Robust backend
- PostgreSQL or SQLite via SQLAlchemy 2.x, with connection pre-ping for hosted databases.
- Flask-Migrate (Alembic) ready for schema migrations.
- Safe uploads: extension checks, unique filenames, per-user private storage outside the public static folder.
- Clean JSON error responses for the interactive front-end, with a normal form-post fallback.

---

## 🧠 How it works

```
 Content image ──► VGG encoder ──► content features ─┐
                                                      ├─► AdaIN ──► (× alpha blend) ──► Decoder ──► Stylized image
 Style image   ──► VGG encoder ──► style features ───┘
```

AdaIN aligns the channel-wise **mean and variance** of the content features to those of the style features:

```
AdaIN(x, y) = σ(y) · ( (x − μ(x)) / σ(x) ) + μ(y)
```

The result is then blended with the original content features using `alpha` and passed through a trained decoder that maps the features back to an image:

```
t = alpha · AdaIN(f_content, f_style) + (1 − alpha) · f_content
```

- **Encoder:** pretrained, frozen VGG-19 (up to `relu4_1`).
- **Decoder:** mirrored architecture trained with a content loss and a style loss.

---

## 🗂️ Project structure

```
AdaIN-Neural-Style-Transfer/
├── app.py                  # Flask app: auth, routes, DB models, inference
├── train.py                # Train the decoder on your own datasets
├── requirements.txt
├── .env.example            # Environment variable template
├── code.ipynb              # Notebook experiments
├── utils/
│   ├── models.py           # VGG encoder + decoder
│   └── utils.py            # AdaIN operation and helpers
├── templates/
│   ├── index.html          # Style transfer page
│   ├── history.html        # User history / gallery
│   ├── login.html
│   └── register.html
├── experiment/
│   └── experiment1/decoder_10.pth   # Trained decoder weights (not tracked by git)
├── examples/               # Example images shown on the home page
├── vgg_normalised.pth      # Pretrained VGG weights (not tracked by git)
└── instance/               # Created at runtime: SQLite DB + per-user uploads
```

---

## ⚙️ Installation

### 1. Clone

```bash
git clone https://github.com/dileepkumar2626/AdaIN-Neural-Style-Transfer.git
cd AdaIN-Neural-Style-Transfer
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU acceleration, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first.

### 4. Add the model weights

Weight files are large, so they are excluded from git. Place them here:

| File | Location |
|------|----------|
| Pretrained normalised VGG-19 (`vgg_normalised.pth`) | project root |
| Trained decoder (`decoder_10.pth`) | `experiment/experiment1/` |

The normalised VGG weights are widely available from the original AdaIN implementations (for example the [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) project). To produce your own decoder, see [Training](#-training).

### 5. Configure environment variables

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edit `.env`:

```env
SECRET_KEY=change-me-to-a-long-random-string
# PostgreSQL (postgres:// and postgresql:// URLs are converted automatically).
# Leave empty to use a local SQLite database at instance/app.db
DATABASE_URL=postgresql://user:password@localhost:5432/adain
AUTO_CREATE_DB=1
FLASK_DEBUG=0
```

Generate a strong secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 6. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**, create an account and start styling.

---

## 🗄️ Database

| Setting | Behaviour |
|---------|-----------|
| `DATABASE_URL` empty | Uses SQLite at `instance/app.db` (great for development) |
| `DATABASE_URL` set | Uses PostgreSQL (psycopg 3) with connection pooling and pre-ping |
| `AUTO_CREATE_DB=1` | Creates tables on startup (development / first run) |
| `AUTO_CREATE_DB=0` | Use migrations instead (recommended in production) |

**Tables:** `users` and `style_transfers` (linked by a foreign key with `ON DELETE CASCADE`).

**Using migrations** (recommended for production):

```bash
export FLASK_APP=app.py        # Windows PowerShell: $env:FLASK_APP="app.py"
flask db init                  # once
flask db migrate -m "initial schema"
flask db upgrade
```

Then set `AUTO_CREATE_DB=0`.

---

## 🏋️ Training

Train the decoder on your own content and style datasets (for example MS-COCO for content and WikiArt for style):

```bash
python train.py --help         # lists every option (dataset paths, epochs, batch size, crop, ...)
```

- Random cropping is on by default; disable it with `--no-crop`.
- Checkpoints are saved per epoch; point `app.py` at the checkpoint you want to serve (default: `experiment/experiment1/decoder_10.pth`).

---

## 🌐 Deployment

Run with a production server instead of the Flask dev server:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Checklist:

- [ ] Set a strong `SECRET_KEY` and `FLASK_DEBUG=0`.
- [ ] Use PostgreSQL and run `flask db upgrade`; set `AUTO_CREATE_DB=0`.
- [ ] Serve behind HTTPS (Nginx, Caddy or your platform's proxy).
- [ ] **Persist `instance/user_uploads/`.** Many hosts (Render, Heroku, etc.) wipe the disk on every deploy. Mount a persistent volume or move uploads to object storage such as S3. History entries whose files are missing are shown with a "no longer available" notice.
- [ ] Make sure the model weight files are present on the server.
- [ ] Model loading uses PyTorch, so allow enough RAM (≈ 1–2 GB) per worker.

---

## 🔐 Security notes

- Passwords are hashed with Werkzeug (`generate_password_hash`).
- All forms and AJAX calls are CSRF-protected.
- Uploaded files are renamed with random UUIDs and stored per user, outside the public static folder, and served only to their owner.
- Database errors are logged server-side and never shown to users.

---

## 🗺️ Roadmap

- [ ] Cloud storage (S3 / Cloudinary) for uploads
- [ ] Preset style gallery and style interpolation (mix several styles)
- [ ] Video and webcam style transfer
- [ ] Public sharing links for results
- [ ] Docker image and one-click deployment

---

## 📚 References

- X. Huang and S. Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
- K. Simonyan and A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, ICLR 2015.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-idea`
3. Commit your changes: `git commit -m "Add amazing idea"`
4. Push the branch: `git push origin feature/amazing-idea`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. Add a `LICENSE` file to the repository to make this official.

---

<div align="center">

Built with ❤️ by [Dileep Kumar](https://github.com/dileepkumar2626)

⭐ If you like this project, give it a star!

</div>