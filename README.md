# Ingredient Detector & Recipe Generator

This Streamlit app detects food ingredients from images and generates practical recipes using a fine-tuned GPT-2 model.

Important: large model files are not included in this repository by default. See below for deployment instructions.

## Features
- Upload images or take photos with your camera
- Automatic ingredient detection (Keras/TensorFlow)
- Recipe generation (Hugging Face transformers GPT-2, CPU)
- Sample gallery and run-final-tests mode

## Files of interest
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `best_model.keras` - (optional) trained Keras detection model (large; excluded from repo by default)
- `models/` - (optional) GPT-2 tokenizer/model folders (large; excluded from repo by default)

## How to run locally
1. Create a Python virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Place models in the project:
- Put your Keras detection model at `./best_model.keras` (or `./models/best_model.keras`).
- Put your fine-tuned GPT-2 model/tokenizer inside `./models/gpt2_recipe_final` (or separate tokenizer/model folders under `./models/`).

4. Run the app:

```powershell
$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE='none'; .\.venv\Scripts\streamlit.exe run app.py --server.port 8504
```

5. Open http://localhost:8504 in your browser.

## Notes
- The app is optimized to load the detection model on-demand and the GPT-2 model on user request.
- For production or sharing with others, host the model files externally (S3, Google Drive, or release assets) and update the app to download them at startup or on demand.
- For high-quality recipe generation, consider using a stronger instruction-tuned LLM or a hosted inference API (OpenAI, Hugging Face Inference).

## License
Add a license file if you want to make this repo public with a license.
