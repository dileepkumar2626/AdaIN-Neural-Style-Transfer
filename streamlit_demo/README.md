# StyleForge AI – Streamlit version

Same model and UI as the Flask app (`app.py` + `templates/index.html`), minus login/database.

## Run locally (from the repo root)
```bash
pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app.py
```

## Deploy on Streamlit Community Cloud
1. Push the repo to GitHub (it must contain `vgg_normalised.pth`, `experiment/experiment1/decoder_128.pth`,
   `static/styles/` and `examples/`).
2. https://share.streamlit.io → **Create app** → pick the repo/branch.
3. **Main file path:** `streamlit_demo/app.py`  ·  Advanced settings → Python **3.12**.
4. Deploy.
