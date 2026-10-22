# 🎨 StyleForge AI — Neural Style Transfer with AdaIN

**Transform ordinary photographs into artistic images using deep learning.** StyleForge AI combines Adaptive Instance Normalization (AdaIN), PyTorch, and a Flask web application to transfer the visual characteristics of an artwork onto a photograph — without retraining the model for every new style.

<p align="center">
  <img src="examples/brad%20pitt.jpg" width="30%" alt="Original content photograph">
  <img src="examples/sketch.png" width="30%" alt="Artistic style reference">
  <img src="examples/stylized_brad_pitt.jpg" width="30%" alt="Neural style transfer result">
</p>

<p align="center">
  <em>Content image → Style reference → Stylized result</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Computer_Vision-Deep_Learning-8A2BE2" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Inference-CPU%20%7C%20CUDA-76B900?logo=nvidia&logoColor=white" alt="CPU and CUDA">
</p>

---

## ✨ Overview

StyleForge AI is a deep learning application built around **Arbitrary Neural Style Transfer**. It separates the content of an image from its visual appearance and combines them to create a new artistic result.

For example, you can take a photograph of a person and apply the colors, textures, and visual patterns of a painting while preserving recognizable structures from the original photograph.

Unlike traditional optimization-based neural style transfer, which may require repeatedly optimizing an image, AdaIN enables efficient stylization using a pretrained encoder and a trained decoder.

The project brings the model into a web application where users can register, upload images, choose their desired style, adjust stylization strength, generate results, and revisit previous transformations.

## 🚀 Features

### 🧠 AI-powered image stylization

- **Arbitrary style transfer:** Use different style-reference images without retraining the network for each style.
- **Adaptive Instance Normalization:** Align content-feature statistics with those of the style image.
- **Adjustable stylization strength:** Control the balance between original content features and stylized features using the alpha parameter.
- **Deep feature extraction:** Use a pretrained, normalized VGG-19 encoder.
- **Custom decoder:** Reconstruct stylized images from transformed feature representations.
- **CPU and NVIDIA CUDA support:** Run inference on compatible hardware with GPU acceleration when available.

### 🌐 Interactive web application

- Upload content and style images through a browser interface.
- Preview and compare original and generated images.
- Adjust style strength and generate different variations.
- Reuse images for additional experiments.
- Download generated results.
- Explore example images and artistic styles.

### 👤 User accounts and image history

- User registration and login.
- Hashed passwords and CSRF-protected forms.
- Password-reset workflow.
- Personal history of previous style transfers.
- View and delete saved transfer records.
- Per-user access controls for stored images.

### 🛠️ Model training and experimentation

- Train the decoder using your own content and style datasets.
- Configure training parameters, including epochs, batch size, learning rate, and loss weights.
- Save model checkpoints and resume experiments where supported by the training configuration.
- Explore model behavior and intermediate features using the included Jupyter notebook.

---

## 🧠 How It Works

StyleForge AI uses **Adaptive Instance Normalization (AdaIN)**, introduced by Huang and Belongie in their 2017 paper, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*.

### Architecture

```text
                     CONTENT IMAGE
                           |
                           v
                    +-------------+
                    |  VGG-19     |
                    |   Encoder   |
                    +-------------+
                           |
                     Content Features
                           |
                           +-------------------+
                                               |
                                               v
                                        +-------------+
                     Style Features --> |    AdaIN    |
                                               |
                                               v
                                        Alpha Blending
                                               |
                                               v
                                        +-------------+
                                        |   Decoder   |
                                        +-------------+
                                               |
                                               v
                                      STYLIZED IMAGE
```

The style image passes through the same encoder to extract its feature statistics. AdaIN then aligns the channel-wise mean and variance of the content features with those of the style features.

### 1. Feature extraction

A pretrained VGG-19 network extracts hierarchical visual representations from both images. Its convolutional features capture image structure, textures, colors, and other visual patterns.

### 2. Adaptive Instance Normalization

Given content features \(x\) and style features \(y\), AdaIN is defined as:

\[
\operatorname{AdaIN}(x,y)=
\sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\mu(y)
\]

Here, \(\mu\) and \(\sigma\) represent channel-wise means and standard deviations.

This operation transfers the feature statistics of the style image to the content representation.

### 3. Style-strength control

The transformed representation is blended with the original content features:

\[
t=\alpha\,\operatorname{AdaIN}(x,y)+(1-\alpha)x
\]

Where:

- \(\alpha=0\): preserve the original content features.
- \(\alpha=0.5\): use an intermediate level of stylization.
- \(\alpha=1\): use the fully transformed feature representation.

### 4. Image reconstruction

The trained decoder transforms the blended feature representation back into image space, producing the final stylized image.

**Important:** The decoder is trained beforehand. Generating an image is an inference operation, not a training operation. The model therefore does not need to be retrained every time a user changes the content image or style reference.

---

## 🖼️ Example Results

The repository includes sample photographs, style references, and generated outputs.

| Content image | Style reference | Generated result |
|---|---|---|
| `examples/brad pitt.jpg` | `examples/picasso seated nude hr.jpg` | `examples/stylized_brad_pitt.jpg` |
| `examples/sketch.png` | Your chosen artwork | Generate using the web application |

Use the included examples to explore the project's capabilities. Actual results depend on the chosen images, decoder checkpoint, input resolution, and alpha value.

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and model implementation |
| PyTorch | Deep learning and model inference |
| Torchvision | Vision-model utilities and image transformations |
| VGG-19 | Pretrained feature extraction |
| Adaptive Instance Normalization | Feature-statistics-based style transfer |
| Pillow | Image loading and processing |
| Flask | Web application and backend routes |
| Flask-Login | User authentication and session management |
| Flask-WTF / WTForms | Forms and CSRF protection |
| Flask-SQLAlchemy | Database integration |
| SQLAlchemy | Database ORM |
| Flask-Migrate | Database schema migrations |
| SQLite / PostgreSQL | Persistent application data |
| Jupyter Notebook | Experiments and model exploration |

---

## 📂 Project Structure

```text
AdaIN-Neural-Style-Transfer/
│
├── app.py
├── train.py
├── requirements.txt
├── code.ipynb
│
├── utils/
│   ├── models.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── forgot_password.html
│   ├── reset_password.html
│   └── history.html
│
├── static/
│   └── styles/
│       ├── styles.json
│       ├── thumbs/
│       └── [style images]
│
├── examples/
│   ├── brad pitt.jpg
│   ├── picasso seated nude hr.jpg
│   ├── sketch.png
│   └── [stylized examples]
│
├── experiment/
│   └── experiment1/
│       └── [trained decoder checkpoint]
│
└── tools/
    └── generate_styles.py
```

**Model weights and datasets:** The pretrained VGG weights, trained decoder checkpoints, and training datasets may need to be obtained separately. The exact files required depend on the paths configured in the application and training script.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or another compatible Python version.
- Git.
- Pip and virtual-environment support.
- The required pretrained VGG-19 weights and trained decoder checkpoint.
- An NVIDIA GPU with a compatible CUDA-enabled PyTorch build for GPU acceleration. CPU inference is also supported.

### 1. Clone the repository

```bash
git clone https://github.com/dileepkumar2626/AdaIN-Neural-Style-Transfer.git

cd AdaIN-Neural-Style-Transfer
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.10 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

If you have a different compatible Python version installed, replace `3.10` with that version.

### 3. Install PyTorch

PyTorch is installed separately because the correct build depends on your hardware.

**NVIDIA GPU with CUDA 12.1 wheels**

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only installation**

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2
```

For other hardware or software configurations, consult the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Install the remaining dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Add the model checkpoints

The application expects the pretrained VGG weights and a trained decoder at the paths configured in `app.py`.

The current application references:

```text
vgg_normalised.pth
experiment/
└── experiment1/
    └── decoder_100.pth
```

Place the required files at these locations, or update the corresponding paths in the code if your checkpoint filenames differ.

Do not assume that cloning the repository also downloads these model weights.

### 6. Configure environment variables

The application can use SQLite for local development or PostgreSQL for a database-backed deployment.

For a local PowerShell session, you can set a secret key and enable automatic database table creation:

```powershell
$env:SECRET_KEY = "replace-with-a-long-random-secret"
$env:AUTO_CREATE_DB = "1"
```

To use PostgreSQL, configure `DATABASE_URL` with your database connection string:

```powershell
$env:DATABASE_URL = "postgresql://username:password@localhost:5432/adain"
```

For local SQLite development, leave `DATABASE_URL` unset. The application uses a database under its instance directory.

For production, use a securely generated secret key, protect database credentials, and configure migrations appropriately. Avoid committing secrets to Git.

### 7. Start the application

```bash
python app.py
```

Open the local address in your browser:

**http://127.0.0.1:5000**

Register an account, upload a content image and a style reference, adjust the stylization strength, and generate your result.

---

## 🏋️ Training the Model

The project includes `train.py` for training the decoder used by the AdaIN pipeline.

Training is separate from using the web application. A compatible pretrained VGG checkpoint and suitable content and style datasets are required.

### Training workflow

1. Prepare content and style image directories.
2. Download or configure the pretrained VGG weights.
3. Configure the experiment name and training parameters.
4. Run the training script.
5. Evaluate the resulting decoder checkpoint.
6. Configure the web application to load the checkpoint you want to use.

Start by inspecting the supported training arguments:

```bash
python train.py --help
```

A training experiment can involve parameters such as:

| Parameter | Purpose |
|---|---|
| Content dataset | Images whose structures should be preserved |
| Style dataset | Images providing artistic appearance |
| Batch size | Number of image samples processed per batch |
| Learning rate | Controls optimizer update size |
| Epochs | Number of training passes through the dataset |
| Content weight | Controls the content reconstruction objective |
| Style weight | Controls the style reconstruction objective |
| Checkpoint interval | Controls how often model weights are saved |

The exact arguments and defaults are defined in `train.py`.

**Hardware note:** Training can require substantial GPU memory and processing time depending on image resolution, dataset size, batch size, and number of epochs. Start with a smaller configuration to verify that your environment works before launching a long experiment.

---

## 🔐 Security and Data Handling

The web application includes several security-oriented features:

- Password hashing rather than storing plaintext passwords.
- CSRF protection for supported forms.
- Login-protected personal history.
- Per-user access checks for stored transfer files.
- Unique filenames for uploads.
- Rate limiting on authentication-related routes.
- Time-limited password-reset tokens.
- Database-backed transfer records.

For production deployment, additionally configure HTTPS, secure environment variables, database backups, upload limits, persistent storage, and appropriate server-level protections.

Never deploy with a development secret key or enable Flask debug mode in production.

---

## 🧪 Experiments and Further Development

The repository provides a foundation for exploring neural style transfer beyond a single notebook experiment.

Potential directions include:

- Comparing different alpha values and input resolutions.
- Evaluating how style-reference selection affects generated images.
- Studying the trade-off between visual similarity and preservation of content structure.
- Testing the effects of different decoder checkpoints.
- Improving inference latency and GPU memory usage.
- Extending the application with additional style presets and comparison tools.
- Investigating stylization of faces, portraits, and highly textured images.

These are potential research and development directions, not claims that every feature has already been implemented.

---

## 📚 Research Reference

Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. Proceedings of the IEEE International Conference on Computer Vision (ICCV).

- [Read the paper on arXiv](https://arxiv.org/abs/1703.06868)
- [Original PyTorch AdaIN implementation](https://github.com/naoto0804/pytorch-AdaIN)

The AdaIN method forms the core theoretical foundation of this project.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Commit your changes with a descriptive message.
5. Push your branch and open a pull request.

When contributing, please avoid committing large model weights, private user uploads, datasets without appropriate permissions, or environment secrets.

---

## 👨‍💻 Author

**Dileep Kumar**

Computer Science student interested in artificial intelligence, machine learning, computer vision, and deep learning engineering.

- GitHub: [@dileepkumar2626](https://github.com/dileepkumar2626)
- Project: [AdaIN-Neural-Style-Transfer](https://github.com/dileepkumar2626/AdaIN-Neural-Style-Transfer)

---

## 📄 License

Before using or redistributing this project, add and verify an explicit repository license. Also check the licensing terms of the pretrained model weights, training datasets, and example images independently.

---

<p align="center">
  <strong>Turning images into art with deep learning.</strong>
  <br>
  If you find this project interesting, consider starring the repository ⭐
</p>
