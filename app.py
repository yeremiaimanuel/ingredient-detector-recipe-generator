import os
import time
import pickle
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

    # list subdirectories
    subdirs = [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
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


def load_models_on_start():
    """Attempt to load detection model, class names and GPT-2 at startup."""
    # Load only the detection model and class names at startup to avoid blocking UI
    detection_model = load_detection_model()
    class_names = load_class_names()

    # Save to session state
    st.session_state['detection_model'] = detection_model
    st.session_state['class_names'] = class_names
    # GPT-2 remains unloaded until user requests it
    st.session_state['tokenizer'] = None
    st.session_state['gpt2_model'] = None
    st.session_state['gpt2_sample'] = None


def load_gpt2_on_demand():
    """Load GPT-2 model when requested by user and produce a short sample."""
    tokenizer, gpt2_model = try_load_gpt2()
    st.session_state['tokenizer'] = tokenizer
    st.session_state['gpt2_model'] = gpt2_model
    gpt2_sample = None
    if tokenizer is not None and gpt2_model is not None:
        try:
            import torch
            device = torch.device('cpu')
            gpt2_model.to(device)
            prompt = '[INGREDIENT: chicken]\\nRecipe Name:'
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
        if st.button("Load GPT-2 model now (on-demand)"):
            with st.spinner("Loading GPT-2 (this may take a while)..."):
                load_gpt2_on_demand()
            if st.session_state.get('gpt2_model') is not None:
                st.success("GPT-2 model loaded")
            else:
                st.error("GPT-2 failed to load. Check model folder or dependencies.")

        st.markdown("---")
        st.subheader("Demo options")
        sample_count = st.slider("Sample gallery size", 3, 30, 12)
        st.checkbox("Show sample gallery", value=True, key='show_gallery')
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
                            st.experimental_rerun()
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

