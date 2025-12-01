import os
import time
import pickle
import traceback
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import streamlit as st


IMG_SIZE = 224
MODEL_DIR = "./models"
DEFAULT_DETECTION_MODEL_PATHS = [
    "./best_model.keras",
    os.path.join(MODEL_DIR, "best_model.keras"),
    os.path.join(MODEL_DIR, "ingredient_detection_model.keras"),
]
GPT2_MODEL_DIRS = [
    os.path.join(MODEL_DIR, "gpt2_recipe_final"),
    os.path.join(MODEL_DIR, "gpt2_recipe_model"),
    os.path.join(MODEL_DIR, "gpt2_recipe_checkpoints"),
]

# Official hosted model URLs (Hugging Face) — convenient defaults for downloader
HF_DETECTION_MODEL_URL = "https://huggingface.co/yeremiaimanuels/food-recipe-assistant/resolve/main/best_model.keras"
HF_MODELS_ARCHIVE_URL = "https://huggingface.co/yeremiaimanuels/food-recipe-assistant/resolve/main/gpt2_recipe_model.zip"


@st.cache_resource
def load_detection_model() -> Optional[object]:
    """Try multiple locations and load a Keras model if available."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception:
        return None

    for p in DEFAULT_DETECTION_MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = load_model(p)
                return model
            except Exception:
                continue
    return None


@st.cache_data
def load_class_names() -> List[str]:
    # Try saved class names first
    pkl_path = os.path.join(MODEL_DIR, "class_names.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # Fallback to the list defined in the notebook
    return [
        "banana", "bread", "broccoli", "carrot", "chicken", "cucumber", "eggplant",
        "fish", "lamb", "mustard greens", "potato", "salmon", "sausage", "shrimp",
        "squid", "sweet potato", "tempeh", "tofu"
    ]


def try_load_gpt2():
    """Attempt to load a fine-tuned GPT-2 model + tokenizer if present.
    Returns (tokenizer, model) or (None, None) if unavailable.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except Exception:
        return None, None

    # Try to intelligently locate tokenizer and model directories inside MODEL_DIR
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"}
    model_files = {"pytorch_model.bin", "tf_model.h5", "model.safetensors"}

    # list subdirectories (guard if MODEL_DIR is missing)
    if not os.path.exists(MODEL_DIR) or not os.path.isdir(MODEL_DIR):
        return None, None
    try:
        subdirs = [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    except Exception:
        return None, None
    tokenizer_dirs = []
    model_dirs = []
    for sd in subdirs:
        try:
            files = set(os.listdir(sd))
        except Exception:
            files = set()
        if files & tokenizer_files:
            tokenizer_dirs.append(sd)
        if files & model_files:
            model_dirs.append(sd)

    # Prefer explicit tokenizer + model pair
    if tokenizer_dirs and model_dirs:
        for td in tokenizer_dirs:
            for md in model_dirs:
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(td)
                    model = GPT2LMHeadModel.from_pretrained(md)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    if getattr(model.config, 'pad_token_id', None) is None:
                        model.config.pad_token_id = tokenizer.eos_token_id
                    return tokenizer, model
                except Exception:
                    continue

    # Fallback: try known GPT2_MODEL_DIRS as single folder containing both tokenizer+model
    for d in GPT2_MODEL_DIRS:
        if os.path.exists(d) and os.path.isdir(d):
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(d)
                model = GPT2LMHeadModel.from_pretrained(d)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if getattr(model.config, 'pad_token_id', None) is None:
                    model.config.pad_token_id = tokenizer.eos_token_id
                return tokenizer, model
            except Exception:
                continue
    return None, None


def try_load_gpt2_verbose():
    """Verbose GPT-2 loader that returns (tokenizer, model, error_str).
    Useful in deployed environments to capture error tracebacks."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except Exception:
        return None, None, traceback.format_exc()

    tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"}
    model_files = {"pytorch_model.bin", "tf_model.h5", "model.safetensors"}

    if not os.path.exists(MODEL_DIR) or not os.path.isdir(MODEL_DIR):
        return None, None, f"MODEL_DIR '{MODEL_DIR}' not found"
    try:
        subdirs = [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    except Exception:
        return None, None, traceback.format_exc()

    tokenizer_dirs = []
    model_dirs = []
    for sd in subdirs:
        try:
            files = set(os.listdir(sd))
        except Exception:
            files = set()
        if files & tokenizer_files:
            tokenizer_dirs.append(sd)
        if files & model_files:
            model_dirs.append(sd)

    # Prefer explicit tokenizer + model pair
    if tokenizer_dirs and model_dirs:
        for td in tokenizer_dirs:
            for md in model_dirs:
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(td)
                    model = GPT2LMHeadModel.from_pretrained(md)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    if getattr(model.config, 'pad_token_id', None) is None:
                        model.config.pad_token_id = tokenizer.eos_token_id
                    return tokenizer, model, None
                except Exception:
                    err = traceback.format_exc()
                    return None, None, err

    for d in GPT2_MODEL_DIRS:
        if os.path.exists(d) and os.path.isdir(d):
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(d)
                model = GPT2LMHeadModel.from_pretrained(d)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if getattr(model.config, 'pad_token_id', None) is None:
                    model.config.pad_token_id = tokenizer.eos_token_id
                return tokenizer, model, None
            except Exception:
                return None, None, traceback.format_exc()
    return None, None, f"No tokenizer/model found under {MODEL_DIR}"


def load_detection_model_verbose():
    """Load detection model and return (model, error_str).
    This exposes tracebacks to help debugging in hosted deployments."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception:
        return None, traceback.format_exc()

    for p in DEFAULT_DETECTION_MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = load_model(p)
                return model, None
            except Exception:
                return None, traceback.format_exc()
    return None, f"No detection model found in paths: {DEFAULT_DETECTION_MODEL_PATHS}"


def load_models_on_start():
    """Attempt to load detection model, class names and GPT-2 at startup."""
    # Attempt auto-download in deployed environments if env vars provided
    try:
        model_archive_url = os.environ.get('MODEL_ARCHIVE_URL')
        detection_url = os.environ.get('DETECTION_MODEL_URL')
        need_models = not (os.path.exists('best_model.keras') or os.path.exists(MODEL_DIR))
        if (model_archive_url or detection_url) and need_models:
            try:
                download_and_install_models(model_archive_url or '', detection_url or '')
            except Exception:
                # swallow here; verbose loader will show errors
                pass
    except Exception:
        pass

    # Load only the detection model and class names at startup to avoid blocking UI
    class_names = load_class_names()
    detection_model, det_err = load_detection_model_verbose()

    # Save to session state with error details for diagnosis
    st.session_state['detection_model'] = detection_model
    st.session_state['detection_load_error'] = det_err
    st.session_state['class_names'] = class_names
    # GPT-2 remains unloaded until user requests it
    st.session_state['tokenizer'] = None
    st.session_state['gpt2_model'] = None
    st.session_state['gpt2_sample'] = None
    st.session_state['gpt2_load_error'] = None


def load_gpt2_on_demand():
    """Load GPT-2 model when requested by user and produce a short sample."""
    tokenizer, gpt2_model, err = try_load_gpt2_verbose()
    st.session_state['tokenizer'] = tokenizer
    st.session_state['gpt2_model'] = gpt2_model
    st.session_state['gpt2_load_error'] = err
    gpt2_sample = None

    if tokenizer is not None and gpt2_model is not None:
        try:
            import torch
            device = torch.device('cpu')
            gpt2_model.to(device)
            prompt = '[INGREDIENT: chicken]\nRecipe Name:'
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            outputs = gpt2_model.generate(
                input_ids,
                max_length=300,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in text:
                text = text.split(prompt, 1)[1].strip()
            if '[END]' in text:
                text = text.split('[END]')[0].strip()
            gpt2_sample = text
        except Exception:
            gpt2_sample = None
    st.session_state['gpt2_sample'] = gpt2_sample


def preprocess_image_for_model(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def download_file_stream(url: str, target_path: str, progress_st=None) -> bool:
    try:
        import requests
        from math import floor
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = resp.headers.get('content-length')
        if total is None:
            # unknown size
            with open(target_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if progress_st:
                progress_st.progress(100)
            return True
        total = int(total)
        dl = 0
        with open(target_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    dl += len(chunk)
                    if progress_st:
                        progress_st.progress(min(floor(dl * 100 / total), 100))
        return True
    except Exception:
        return False


def extract_archive_if_needed(archive_path: str, dest_dir: str) -> bool:
    try:
        import zipfile, tarfile
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(dest_dir)
            return True
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as t:
                t.extractall(dest_dir)
            return True
        # Not an archive; maybe a model file
        return False
    except Exception:
        return False


def download_and_install_models(models_url: str, detection_url: str) -> Tuple[bool, str]:
    """Download provided URLs and install into ./models/ and place detection model.
    Returns (success: bool, message: str). On failure, message contains an error traceback or reason.
    Supports local filesystem paths and file:// URLs in addition to HTTP/HTTPS.
    """
    import shutil

    MODEL_DIR_LOCAL = MODEL_DIR
    os.makedirs(MODEL_DIR_LOCAL, exist_ok=True)

    # If models archive URL provided, download/extract or copy from local path
    if models_url:
        try:
            # handle file:// or local absolute/relative paths without using requests
            if models_url.startswith('file://'):
                local_path = models_url[len('file://'):]
            else:
                local_path = models_url

            if os.path.exists(local_path):
                # direct local file — if archive, extract; otherwise copy into models dir
                if extract_archive_if_needed(local_path, MODEL_DIR_LOCAL):
                    pass
                else:
                    shutil.copy(local_path, os.path.join(MODEL_DIR_LOCAL, os.path.basename(local_path)))
            else:
                tmp_archive = os.path.join(MODEL_DIR_LOCAL, 'models_download.tmp')
                prog = None
                try:
                    prog = st.progress(0)
                    ok = download_file_stream(models_url, tmp_archive, prog)
                    if not ok:
                        return False, f"Failed to download models archive from {models_url}"
                    extracted = extract_archive_if_needed(tmp_archive, MODEL_DIR_LOCAL)
                    # if extraction failed, maybe the archive is a single model file
                    if not extracted:
                        # move file into models dir preserving name
                        shutil.move(tmp_archive, os.path.join(MODEL_DIR_LOCAL, os.path.basename(models_url)))
                    else:
                        os.remove(tmp_archive)
                finally:
                    if prog:
                        prog.empty()
        except Exception:
            return False, traceback.format_exc()

    # If detection model URL provided, download or copy it to repo root as best_model.keras
    if detection_url:
        try:
            if detection_url.startswith('file://'):
                local_det = detection_url[len('file://'):]
            else:
                local_det = detection_url

            target = os.path.join('.', 'best_model.keras')
            if os.path.exists(local_det):
                shutil.copy(local_det, target)
            else:
                prog2 = None
                try:
                    prog2 = st.progress(0)
                    ok2 = download_file_stream(detection_url, target, prog2)
                    if not ok2:
                        return False, f"Failed to download detection model from {detection_url}"
                finally:
                    if prog2:
                        prog2.empty()
        except Exception:
            return False, traceback.format_exc()
    return True, "Models installed — restart the app or click 'Load detection model now' and 'Load GPT-2 model now'"


def predict_image(model, img: Image.Image, class_names: List[str]) -> Tuple[str, float]:
    x = preprocess_image_for_model(img)
    preds = model.predict(x, verbose=0)
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = np.array(preds).ravel()
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])


def generate_recipe(tokenizer, gpt2_model, ingredient: str) -> str:
    try:
        import torch
        device = torch.device("cpu")
        gpt2_model.to(device)
        prompt = f"[INGREDIENT: {ingredient}]\nRecipe Name:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = gpt2_model.generate(
            input_ids,
            max_length=500,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # remove prompt
        if prompt in text:
            text = text.split(prompt, 1)[1].strip()
        # cut at END token if present
        if "[END]" in text:
            text = text.split("[END]")[0].strip()
        return text
    except Exception as e:
        return f"(recipe generation failed: {e})"


def test_multiple_samples(detection_model, tokenizer, gpt2_model, class_names, base_image_dir="./image_sample/", n_samples=20):
    results = []
    if not os.path.exists(base_image_dir):
        return results, "No sample images folder found. Place images in './image_sample/'"

    all_images = [os.path.join(base_image_dir, f) for f in os.listdir(base_image_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".webp"))]
    if not all_images:
        return results, "No images found in './image_sample/'"

    import random
    n_samples = min(n_samples, len(all_images))
    sample_images = random.sample(all_images, n_samples)

    for img_path in sample_images:
        try:
            img = Image.open(img_path)
        except Exception:
            continue

        detected, confidence = predict_image(detection_model, img, class_names)
        recipe = None
        if tokenizer is not None and gpt2_model is not None:
            recipe = generate_recipe(tokenizer, gpt2_model, detected)

        results.append({
            "image": img_path,
            "detected": detected,
            "confidence": confidence,
            "recipe": recipe,
        })

    return results, None


def main():
    st.set_page_config(page_title="Food Ingredient Detection + Recipe Generator", layout="wide")

    # Header
    header_cols = st.columns([0.12, 0.76, 0.12])
    with header_cols[1]:
        st.markdown("""
        <div style='text-align:center'>
            <h1 style='margin:0'>🍽️ Ingredient Detector & Recipe Generator</h1>
            <p style='color: #666; margin-top:4px'>Intelligent Cooking Assistant with Detection and Recipe Generation</p>
        </div>
        """, unsafe_allow_html=True)

    # Short descriptive hero text to support UI look and explain value
    st.markdown("""
    <div style='text-align:center; color:#444; margin-top:8px'>
        <strong>This app helps you detect food types from images.</strong>
        <p style='margin:6px 120px; color:#666'>You'll see accurate predictions and then get relevant recipe recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Ensure session keys exist
    if 'detection_model' not in st.session_state:
        st.session_state['detection_model'] = None
    if 'class_names' not in st.session_state:
        st.session_state['class_names'] = load_class_names()
    if 'tokenizer' not in st.session_state:
        st.session_state['tokenizer'] = None
    if 'gpt2_model' not in st.session_state:
        st.session_state['gpt2_model'] = None
    if 'gpt2_sample' not in st.session_state:
        st.session_state['gpt2_sample'] = None
    if 'demo_image_path' not in st.session_state:
        st.session_state['demo_image_path'] = None

    # If deployment provides model URLs via env vars, attempt to auto-install/load once
    try:
        env_models = os.environ.get('MODEL_ARCHIVE_URL') or os.environ.get('DETECTION_MODEL_URL')
        if env_models and st.session_state.get('detection_model') is None:
            # this will attempt download (if needed) and load, storing errors in session_state
            load_models_on_start()
    except Exception:
        pass

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        st.write("Model loading and demo settings")
        # small support blurb in sidebar for quick context
        st.markdown("""
        **This app helps you detect food types from images.**

        Core features:
        - Upload food images
        - Automatic detection
        - Recipe recommendations
        - Ingredient info & step-by-step guide
        """)
        if st.button("Load detection model now"):
            with st.spinner("Loading detection model (this may take a few seconds)..."):
                load_models_on_start()
            if st.session_state.get('detection_model') is not None:
                st.success("Detection model loaded")
            else:
                st.error("Detection model failed to load. Check `best_model.keras` or MODEL_DIR.")
                if st.session_state.get('detection_load_error'):
                    st.subheader('Detection load error (trace)')
                    st.code(st.session_state.get('detection_load_error'))
        if st.button("Load GPT-2 model now (on-demand)"):
            with st.spinner("Loading GPT-2 (this may take a while)..."):
                load_gpt2_on_demand()
            if st.session_state.get('gpt2_model') is not None:
                st.success("GPT-2 model loaded")
            else:
                st.error("GPT-2 failed to load. Check model folder or dependencies.")
                if st.session_state.get('gpt2_load_error'):
                    st.subheader('GPT-2 load error (trace)')
                    st.code(st.session_state.get('gpt2_load_error'))

        st.markdown("---")
        st.subheader("Demo options")
        sample_count = st.slider("Sample gallery size", 3, 30, 12)
        st.checkbox("Show sample gallery", value=True, key='show_gallery')
        st.markdown("---")
        st.subheader("Models downloader")
        st.write("If your deployment does not include model files, paste a direct download URL below (zip or tar) and click Install.")
        models_url = st.text_input("Models archive URL (zip/tar)", value="", help="Public direct link to a models archive containing tokenizer/model folders or best_model.keras")
        detection_url = st.text_input("Detection model file URL (optional)", value="", help="Direct link to best_model.keras or Keras .h5 file")
        st.markdown("---")
        st.write("Or install the hosted models from Hugging Face:")
        if st.button("Install official Hugging Face models"):
            with st.spinner("Downloading official models from Hugging Face..."):
                ok, msg = download_and_install_models(HF_MODELS_ARCHIVE_URL, HF_DETECTION_MODEL_URL)
                if ok:
                    st.success(msg)
                else:
                    st.error("Download/install failed — see error below")
                    st.code(msg)
        if st.button("Download & install models"):
            # perform download and extraction
            with st.spinner("Downloading and installing models..."):
                ok, msg = download_and_install_models(models_url.strip(), detection_url.strip())
                if ok:
                    st.success(msg)
                else:
                    st.error("Download/install failed — see error below")
                    st.code(msg)
        st.markdown("---")
        st.write("Server info")
        st.write(f"Detection model: {'loaded' if st.session_state['detection_model'] is not None else 'not loaded'}")
        st.write(f"GPT-2 model: {'loaded' if (st.session_state['tokenizer'] is not None and st.session_state['gpt2_model'] is not None) else 'not loaded'}")

    # Main content: tabs for Demo / Tests / Samples / About
    demo_tab, tests_tab, samples_tab, about_tab = st.tabs(["Demo", "Run Final Tests", "Samples", "About"])

    # Demo tab
    with demo_tab:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Try it — upload, take a photo, or pick a sample image")
            uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key='uploader')
            # Camera input (captures a photo from user's device camera)
            camera_photo = None
            try:
                camera_photo = st.camera_input("Or take a photo using your camera")
            except Exception:
                # camera_input may not be available in some environments
                camera_photo = None

            if camera_photo is not None:
                # camera_photo is an UploadedFile-like object
                demo_img = Image.open(camera_photo)
                st.session_state['demo_image_path'] = None
            elif uploaded is not None:
                demo_img = Image.open(uploaded)
                st.session_state['demo_image_path'] = None
            elif st.session_state.get('demo_image_path'):
                try:
                    demo_img = Image.open(st.session_state['demo_image_path'])
                except Exception:
                    demo_img = None
            else:
                demo_img = None

            if demo_img is not None:
                st.image(demo_img, use_container_width=True)
            else:
                st.info("No demo image selected. Upload an image or select one from Samples.")

            if demo_img is not None:
                if st.session_state['detection_model'] is not None:
                    if st.button("Detect ingredient"):
                        with st.spinner("Running detection..."):
                            detected, confidence = predict_image(st.session_state['detection_model'], demo_img, st.session_state['class_names'])
                            st.session_state['last_detect'] = (detected, confidence)
                            st.success(f"Detected: {detected} — {confidence:.2%}")
                else:
                    st.warning("Detection model not loaded. Use sidebar 'Load detection model now'.")

        with right:
            st.subheader("Result & Recipe")
            last = st.session_state.get('last_detect')
            if last:
                detected, confidence = last
                st.metric(label="Detected Ingredient", value=detected)
                st.progress(min(int(confidence * 100), 100))
                if st.session_state.get('tokenizer') is not None and st.session_state.get('gpt2_model') is not None:
                    if st.button("Generate recipe for last detection"):
                        with st.spinner("Generating recipe..."):
                            recipe = generate_recipe(st.session_state['tokenizer'], st.session_state['gpt2_model'], detected)
                            st.session_state['last_recipe'] = recipe
                else:
                    st.info("GPT-2 not loaded — use sidebar to load the recipe model.")

            if st.session_state.get('last_recipe'):
                with st.expander("Generated recipe (full)", expanded=True):
                    st.write(st.session_state['last_recipe'])

    # Run Final Tests tab
    with tests_tab:
        st.subheader("Run Final Tests — evaluate on sample images")
        n_samples = st.number_input("Number of samples to test", min_value=1, max_value=200, value=20)
        if st.button("Run Final Tests"):
            if st.session_state['detection_model'] is None:
                st.error("Load detection model first (use sidebar)")
            else:
                with st.spinner("Running tests..."):
                    results, err = test_multiple_samples(
                        detection_model=st.session_state['detection_model'],
                        tokenizer=st.session_state['tokenizer'],
                        gpt2_model=st.session_state['gpt2_model'],
                        class_names=st.session_state['class_names'],
                        base_image_dir="./image_sample/",
                        n_samples=int(n_samples)
                    )
                if err:
                    st.warning(err)
                if results:
                    avg_conf = np.mean([r['confidence'] for r in results])
                    st.success(f"Testing complete — {len(results)} samples, average confidence: {avg_conf:.2%}")
                    import pandas as pd
                    df = pd.DataFrame([{'image': os.path.basename(r['image']), 'detected': r['detected'], 'confidence': r['confidence']} for r in results])
                    st.dataframe(df)
                else:
                    st.info("No results produced.")

    # Samples tab
    with samples_tab:
        st.subheader("Sample images gallery")
        sample_dir = "./image_sample/"
        if not os.path.exists(sample_dir):
            st.info("No sample images found in './image_sample/'.")
        else:
            imgs = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            imgs = imgs[:sample_count]
            if imgs:
                cols = st.columns(4)
                for i, p in enumerate(imgs):
                    with cols[i % 4]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
                        if st.button(f"Use this image", key=f"use_{i}"):
                            st.session_state['demo_image_path'] = p
                            # Use experimental_rerun when available; otherwise trigger a rerun
                            try:
                                if hasattr(st, 'experimental_rerun'):
                                    st.experimental_rerun()
                                else:
                                    # Fallback: change a query param to force rerun using new API
                                    # `st.query_params` is the recommended way to set query params.
                                    try:
                                        st.query_params = {"_rerun": str(int(time.time()))}
                                    except Exception:
                                        # last-resort attempt using string assign
                                        st.query_params = {"_rerun": str(int(time.time()))}
                            except Exception:
                                try:
                                    st.query_params = {"_rerun": str(int(time.time()))}
                                except Exception:
                                    # Last resort: no-op (user can manually refresh)
                                    pass
            else:
                st.info("No sample images to show.")

    # About tab
    with about_tab:
        st.subheader("About this demo")
        st.markdown("""
        **This app helps you detect food types from images.** You'll see accurate predictions and then get relevant recipe recommendations.

        **Core Features**
        - Upload food images
        - Automatic detection
        - Recipe recommendations based on detection results
        - Ingredient information
        - Cooking step-by-step guide

        **Added Values**
        - Fast predictions
        - Simple and easy-to-use interface
        - Practical recipes that can be tried immediately

        **Usage:** Load models from the sidebar, upload or pick a sample image, then run detection and generation. Note: GPT-2 model load may take time and uses the CPU by default.
        """)

    st.markdown("---")
    st.caption("Built for demos — press 'Load detection model now' and 'Load GPT-2 model now' in the sidebar before using heavy operations.")

if __name__ == "__main__":
    main()

