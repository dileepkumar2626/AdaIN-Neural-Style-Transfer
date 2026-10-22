# Deploy the AdaIN demo with Streamlit Community Cloud

This is a separate public demo for the existing AdaIN Neural Style Transfer project. It does not replace or modify the Flask application.

## 1. Add the model weights

The uploaded repository ZIP does **not** include the model checkpoints. Before inference can work, add these files to the repository (or change the paths in `app.py`):

```text
AdaIN-Neural-Style-Transfer/
├── vgg_normalised.pth
├── experiment/
│   └── experiment1/
│       └── decoder_100.pth
└── streamlit_demo/
    ├── app.py
    └── requirements.txt
```

- `vgg_normalised.pth`: normalized VGG checkpoint compatible with `utils/models.py`.
- `experiment/experiment1/decoder_100.pth`: your trained decoder checkpoint.

Do not substitute standard torchvision VGG weights unless they match the custom architecture and key names used by this project. Confirm you have permission to redistribute both checkpoints before making them public. If the files are too large for GitHub, host them on a trusted model repository and add a verified download/cache step.

## 2. Push the demo to GitHub

From the project root:

```bash
git add streamlit_demo
git commit -m "Add Streamlit demo for AdaIN style transfer"
git push
```

The repository's `.gitignore` excludes `.pth` files by default. If you choose to commit model files, deliberately add them only if their size and licenses permit it.

## 3. Deploy

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with GitHub and choose **Create app**.
3. Select your `AdaIN-Neural-Style-Transfer` repository and branch.
4. Set **Main file path** to `streamlit_demo/app.py`.
5. Choose Python 3.11 or 3.12 to match the pinned PyTorch wheels.
6. Deploy the app.

Streamlit should install dependencies from `streamlit_demo/requirements.txt`. If prompted to select a dependency file, choose that file.

## 4. Test the public app

- Upload a content photograph and a style-reference image.
- Start with 256 or 384 px processing resolution.
- Adjust the style strength and generate.
- Download the result as PNG.

## Notes and limitations

- This app reuses the existing `utils/models.py` and `utils/utils.py`; the original Flask app remains intact.
- The host runs the model. Your local NVIDIA GPU is not used by Streamlit Cloud, and GPU access is not guaranteed.
- Large checkpoints can exceed GitHub or hosting limits. For large files, use a trusted model hosting location and download/cache the files at startup.
- The UI checks whether expected checkpoints exist and explains what is missing; inference will not work until valid weights are available.
- Never load model checkpoints from untrusted sources.
