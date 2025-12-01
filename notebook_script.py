# Converted from notebook
#!/usr/bin/env python
# coding: utf-8

# ## <div align="center"> LAB IS794 Deep Learning </div>
# ## <div align="center">  Food Ingredients Detection and Recipe Generation System Using Deep Learning-Based Image Recognition and Generative AI</div>
# #### <div align="center"> Odd Semester 2025/2026 </div>
# ---

# In[1]:


import datetime
import uuid


# In[2]:


import datetime
import uuid

studentName1 = "Delista Dwi Widyastuti (00000105174)"
studentName2 = "Yeremia Imanuel Susanto (00000095653)"
studentName3 = "Yohanes Brian Caesaryano Lala (00000098769)"
studentName4 = "Alexander Briant (00000099774)"
studentClass = "IS794-AL"


# In[3]:


myDate = datetime.datetime.now()
myDevice = str(uuid.uuid1())

print("Name Group C: \t{}".format(studentName1))
print(" \t\t{}".format(studentName2))
print(" \t\t{}".format(studentName3))
print(" \t\t{}".format(studentName4))
print("Class: \t\t{}".format(studentClass))
print("Start: \t\t{}".format(myDate))
print("Device ID: \t{}".format(myDevice))


# # 1. Import Library

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

# Web scraping 
import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import time
from bing_image_downloader import downloader


# We start by importing the necessary libraries to handle data, images, and deep learning tasks. These include tools for basic image processing, machine learning, and web scraping. With TensorFlow and Keras, we build and fine-tune pre-trained models like MobileNetV2, ResNet50, and EfficientNetB0 to classify images effectively. We also use web scraping to gather custom datasets for training and evaluation.
# 
# * **Data Handling & Image Processing**: Libraries like os, numpy, pandas, cv2, and PIL are used to manage and manipulate images.
# * **Machine Learning**: train_test_split helps split the data, while classification_report and accuracy_score evaluate the model’s performance.
# * **Deep Learning Setup**: We use pre-trained models (MobileNetV2, ResNet50, EfficientNetB0) for transfer learning, leveraging TensorFlow/Keras.
# * **Callbacks**: EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau are implemented to improve training and prevent overfitting.
# * **Web Scraping**: Using requests, BeautifulSoup, tqdm, and bing_image_downloader, we scrape images from the web to create a custom dataset.

# # 2. Set Random Seeds

# In[2]:


np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)


# We set the random seed for both NumPy and TensorFlow to ensure reproducibility in our experiments. Additionally, we print the current version of TensorFlow to verify the environment setup.

# # 3. config

# In[3]:


ingredient_classes = [
    "banana",
    "bread",
    "broccoli",
    "carrot",
    "chicken",
    "cucumber",
    "eggplant",
    "fish",
    "lamb",
    "mustard greens",
    "potato",
    "salmon",
    "sausage",
    "shrimp",
    "squid",
    "sweet potato",
    "tempeh",
    "tofu"
]


# We define a list of ingredient classes, which includes banana, bread, broccoli, carrot, chicken, cucumber, eggplant, fish, lamb, mustard greens, potato, salmon, sausage, shrimp, squid, sweet potato, tempeh, and tofu. These ingredients represent a variety of food types, ranging from fruits and vegetables to meats and plant-based alternatives.
# 

# ## 4. Model Configuration

# In[4]:


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


# The model configuration defines key hyperparameters that control the training process and the structure of the deep learning model:
# 
# * **IMG_SIZE = 224**: This sets the input image size to 224x224 pixels. It's a common size used for many pre-trained models (like ResNet, MobileNet, etc.), ensuring compatibility with these architectures.
# * **BATCH_SIZE = 32**: This defines the number of images to be processed in each batch during training. A batch size of 32 is often a good balance between memory usage and model performance.
# * **EPOCHS = 50**: This specifies that the model will train for 50 complete passes through the entire dataset. More epochs can lead to better training, but there’s also a risk of overfitting if too many epochs are used.
# * **LEARNING_RATE = 0.001**: This is the rate at which the model's weights are adjusted during training. A learning rate of 0.001 is a standard starting point, as it’s small enough to allow fine-tuned updates without overshooting the optimal solution.

# ## 5. Directory Configuration

# In[5]:


BASE_INGREDIENT_DIR = "./dataset_ingredients/"
PROCESSED_DIR = "./dataset_processed/"
MODEL_DIR = "./models/"
RECIPES_DIR = "./recipes/"
CSV_FILE = "./recipes/ingredient_recipes.csv"


# The directory configuration specifies paths for storing and accessing various datasets and models. The BASE_INGREDIENT_DIR holds raw ingredient data, PROCESSED_DIR stores preprocessed data, MODEL_DIR saves trained models, RECIPES_DIR contains recipe files, and CSV_FILE points to a CSV containing ingredient recipe mappings.

# ## 6. Scraping parameters

# In[6]:


recipe_data = []
num_recipes_per_ingredient = 50
delay_between_links = (3, 5)


# We used web scraping in this project to gather a diverse and extensive dataset of recipes associated with various food ingredients. By scraping recipe websites, we could automatically collect data on ingredient-recipes relationships, which is crucial for training the deep learning model to both recognize ingredients in images and generate relevant recipes. This approach enables us to build a large, custom dataset tailored to the specific ingredients and recipe context required for the system.
# 
# * **num_recipes_per_ingredient = 15**: This sets the number of recipes to scrape for each ingredient. For each ingredient, the script will collect 15 related recipes.
# * **delay_between_links = (3, 5)**: This defines a random delay (in seconds) between scraping each link, with a range between 3 to 5 seconds. This helps prevent the scraper from making requests too quickly, reducing the risk of getting blocked by the website.

# ## 6.1 Create directories

# In[7]:


for d in [BASE_INGREDIENT_DIR, PROCESSED_DIR, MODEL_DIR, RECIPES_DIR]:
    os.makedirs(d, exist_ok=True) 


# The code creates the necessary directories for the project, including those for raw ingredients, processed data, trained models, and recipes. Using os.makedirs(d, exist_ok=True), it ensures these directories are created if they don’t already exist, preventing any errors.

# # 7. Data Acquisition

# ## 7.1 Scraping Images

# In[11]:


def scrape_ingredient_images(num_images=200):
    """Scrape images for each ingredient class using Bing"""
    print("\n=== START SCRAPING INGREDIENTS IMAGES ===")
    for ingredient in ingredient_classes:
        query = f"raw {ingredient} ingredient"
        print(f"\nScraping: {query}")
        try:
            downloader.download(
                query,
                limit=num_images,
                output_dir=BASE_INGREDIENT_DIR,
                adult_filter_off=True,
                force_replace=False,
                timeout=60,
                verbose=False
            )
            # Rename folder
            raw_folder = os.path.join(BASE_INGREDIENT_DIR, query)
            save_dir = os.path.join(BASE_INGREDIENT_DIR, ingredient)
            if os.path.exists(raw_folder):
                if not os.path.exists(save_dir):
                    os.rename(raw_folder, save_dir)
            print(f"✅ Completed: {ingredient}")

        except Exception as e:
            print(f"❌ Error scraping {ingredient}: {e}")
            continue 
    print("\n✅ Scraping completed!")

# scrape_ingredient_images() 


# We define the function scrape_ingredient_images(num_images=200) to scrape images for each ingredient in the ingredient_classes list using Bing's image search. For each ingredient, we generate a query like "raw {ingredient} ingredient," download the specified number of images (default 200) using bing_image_downloader, and store them in the BASE_INGREDIENT_DIR directory, ensuring that adult content is filtered and no files are overwritten.
# 
# Once the images are downloaded, we rename the folder containing them to match the ingredient name for better organization. If any errors occur during the scraping process, we catch them and continue with the next ingredient. After all ingredients are processed, we print a message confirming that the scraping is complete. The scrape_ingredient_images function collects images for each ingredient by querying Bing, downloading up to 200 images per ingredient, and saving them in organized folders. After scraping, data cleaning involves removing irrelevant or inaccurate images to ensure only appropriate ingredient images are retained.
# 
# The scraping logs reveal several patterns and issues:
# 
# 1. Successful Downloads: We successfully download images for ingredients like chicken, tempeh, and potato, while others like corn and onion fail due to network errors (getaddrinfo failed).
# 2. Image Validity Issues: Some images were deemed invalid (e.g., from Shutterstock and Getty), likely due to access restrictions or premium content, causing download failures.
# 3. HTTP Errors: We encounter HTTP 403 Forbidden and HTTP 400 Bad Request errors, suggesting some websites block our scraper.
# 4. Ingredient Availability: Ingredients like chicken had more images available online, while others like spinach and cabbage had fewer, leading to more scraping failures.
# 
# #### Key Issues:
# * **Network problems** resulted in multiple failures, especially for specific ingredients.
# * **Access restrictions** from premium websites hindered the image collection process.
# * **Ingredient bias**: Some ingredients are more widely represented online, affecting the success of image downloads.
# 

# ## 7.2 Scraping Recipes

# In[12]:


import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


# We begin by setting up the environment for web scraping using Selenium and Chrome WebDriver, configuring it to avoid detection. Then, we iterate through a list of ingredients, search for related recipes on Allrecipes, and collect relevant links. For each recipe link, we extract the title, ingredients, cooking instructions, and image URL. This data is stored in a list called recipe_data, with auto-saves every 20 recipes to a CSV file.
# 
# The data shows a consistent number of recipes (15) for most ingredient classes like tofu, shrimp, and banana, but fewer recipes for ingredients like chicken (9) and sweet potato (9), suggesting some ingredients might be less popular in recipe databases. Biases could arise from the source (Allrecipes), where certain ingredients may be overrepresented, while others are underrepresented due to search algorithms or available data.
# 
# The process involves scraping recipes for each ingredient, cleaning up any irrelevant or incomplete data, and storing the results in a CSV file. Additionally, images are scraped for each ingredient, and irrelevant or incorrect ones are cleaned out for quality control.
# 
# To ensure smooth scraping, we implement error handling that skips any incomplete or problematic recipes, allowing us to continue without interruptions. After scraping, we save the collected data into a final CSV file and generate a summary, including the total number of recipes, ingredient-specific counts, and averages for ingredients and instructions per recipe. Finally, we close the WebDriver to clean up after the process.

# # 8. Data Loading and Visualization

# ## 8.1 Load data Images

# In[8]:


def load_and_visualize_ingredient_dataset():
    """Load ingredient dataset"""
    print("\n" + "="*60)
    print("LOADING AND VISUALIZING INGREDIENT DATASET")
    print("="*60)

    if not os.path.exists(BASE_INGREDIENT_DIR):
        print(f"❌ Folder '{BASE_INGREDIENT_DIR}' tidak ditemukan.")
        return []

    ingredient_classes = [f for f in os.listdir(BASE_INGREDIENT_DIR) 
                         if os.path.isdir(os.path.join(BASE_INGREDIENT_DIR, f))]

    if not ingredient_classes:
        print("⚠️ Tidak ada subfolder ingredients ditemukan.")
        return []

    ingredient_counts = {}
    for ingredient in ingredient_classes:
        ingredient_path = os.path.join(BASE_INGREDIENT_DIR, ingredient)
        image_files = [f for f in os.listdir(ingredient_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))]
        ingredient_counts[ingredient] = len(image_files)

    # Tampilkan statistik dataset
    df_counts = pd.DataFrame(list(ingredient_counts.items()), 
                            columns=['Ingredient', 'Image Count']).sort_values(by="Image Count", ascending=False)
    print("\n📊 Dataset Statistics:")
    print(df_counts.to_string(index=False))
    print(f"\n📦 Total Classes: {len(ingredient_counts)}")
    print(f"🖼️ Total Images: {sum(ingredient_counts.values())}")

    # Visualisasi contoh gambar
    n_classes = min(15, len(ingredient_counts))  # Max 15 samples
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    sample_ingredients = np.random.choice(list(ingredient_counts.keys()), n_classes, replace=False)

    plt.figure(figsize=(15, 3 * n_rows))
    for idx, ingredient in enumerate(sample_ingredients, 1):
        ingredient_path = os.path.join(BASE_INGREDIENT_DIR, ingredient)
        images = [f for f in os.listdir(ingredient_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))]
        plt.subplot(n_rows, n_cols, idx)
        if images:
            img_path = os.path.join(ingredient_path, np.random.choice(images))
            try:
                img = Image.open(img_path)
                plt.imshow(img)
                plt.title(f"{ingredient}\n({ingredient_counts[ingredient]} imgs)", fontsize=9)
                plt.axis('off')
            except Exception as e:
                print(f"⚠️ Gagal buka gambar di {ingredient}: {e}")
                plt.axis('off')
        else:
            plt.axis('off')
    plt.suptitle('Sample Images from Ingredient Dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

    return list(ingredient_counts.keys()) 

ingredient_classes_loaded = load_and_visualize_ingredient_dataset() 


# We begin by loading and visualizing the ingredient dataset, checking if the directory containing the images exists. We count the number of images for each ingredient class and display a summary of the dataset, including the total number of ingredient classes and images. To better understand the data, we randomly select up to 15 ingredients, display a sample image for each, and show the number of images available for each ingredient. The dataset's statistics are presented in a table, helping us assess the balance and diversity of ingredients.
# 
# The key insights from the dataset reveal several patterns and anomalies. Ingredients like cucumber, sweet potato, and bread have more images, showing they are more popular in recipes, while carrot and tofu are underrepresented with fewer images. The dataset is biased toward commonly used ingredients like shrimp and bread, which may affect fairness in model training. Additionally, more commonly used ingredients tend to have higher image counts, while niche ingredients are less represented, creating potential imbalances in the dataset.

# ## 8.2 Load Data Recipes

# In[9]:


import os
import pandas as pd

def load_ingredient_recipe_dataset(csv_path="recipes/ingredient_recipes.csv"):
    """Load dataset hasil scraping resep berdasarkan bahan (ingredient)."""
    print("\n" + "="*70)
    print("📂 LOADING INGREDIENT RECIPE DATASET")
    print("="*70)

    if not os.path.exists(csv_path):
        print(f"❌ File '{csv_path}' tidak ditemukan!")
        print(f"📁 Path dicari: {os.path.abspath(csv_path)}")
        print("\n💡 Pastikan:")
        print("   1. Script scraping sudah dijalankan")
        print("   2. Path file CSV sudah benar")
        print("   3. File CSV ada di folder 'recipes/'")
        return None

    try:
        # Load CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # 🧩 Penyesuaian nama kolom agar konsisten
        if "ingredients" in df.columns and "all_ingredients" not in df.columns:
            df.rename(columns={"ingredients": "all_ingredients"}, inplace=True)

        print(f"✅ Dataset berhasil di-load!")
        print(f"📦 Total recipes: {len(df)}")
        print(f"📊 Columns: {list(df.columns)}")

        # Validasi kolom penting
        required_cols = ["ingredient_class", "recipe_name", "all_ingredients", "instructions"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"\n⚠️ Warning: Kolom hilang: {missing_cols}")
            return df

        # Statistik per ingredient
        print(f"\n📈 Jumlah resep per ingredient class:")
        class_counts = df["ingredient_class"].value_counts()
        for ingredient, count in class_counts.items():
            print(f"   {ingredient}: {count} recipes")

        # Data quality check
        print(f"\n🔍 Data Quality Check:")
        if "image_url" in df.columns:
            img_count = df["image_url"].notna().sum()
            print(f"   Recipes with images: {img_count} ({img_count/len(df)*100:.1f}%)")
        else:
            print("   ⚠️ Kolom 'image_url' tidak ditemukan")

        # Missing values
        print("\n🚨 Missing Values:")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"   {col}: {missing} missing ({missing/len(df)*100:.1f}%)")

        # 🔍 Tampilkan contoh recipe pertama
        print(f"\n🔍 Sample Recipe:")
        sample = df.iloc[0]
        print("="*70)
        print(f"Ingredient Category: {sample['ingredient_class']}")
        print(f"Recipe Name: {sample['recipe_name']}")
        print("\nAll Ingredients:")

        ingredients = sample["all_ingredients"]
        if isinstance(ingredients, str):
            ing_list = ingredients.split("\n")
            for ing in ing_list[:5]:
                print(f"  - {ing}")
            if len(ing_list) > 5:
                print(f"  ... dan {len(ing_list)-5} bahan lainnya")

        print("\nInstructions:")
        instructions = sample["instructions"]
        if isinstance(instructions, str):
            inst_list = instructions.split("\n")
            for step in inst_list[:3]:
                print(f"  {step}")
            if len(inst_list) > 3:
                print(f"  ... dan {len(inst_list)-3} langkah lainnya")

        print(f"\nImage URL: {sample['image_url'] if 'image_url' in df.columns else 'N/A'}")
        print(f"Source URL: {sample['source_url'] if 'source_url' in df.columns else 'N/A'}")
        print("="*70)

        # 📊 Statistik tambahan
        print("\n📊 Additional Stats:")
        newline = "\n"
        avg_ingredients = df["all_ingredients"].str.split(newline).str.len().mean()
        avg_instructions = df["instructions"].str.split(newline).str.len().mean()
        print(f"   Rata-rata jumlah bahan per resep: {avg_ingredients:.1f}")
        print(f"   Rata-rata jumlah langkah memasak: {avg_instructions:.1f}")

        # Top dan bottom ingredients
        print("\n🏆 Top 5 Ingredients dengan resep terbanyak:")
        for i, (ingredient, count) in enumerate(class_counts.head(5).items(), 1):
            print(f"   {i}. {ingredient}: {count} recipes")

        if len(class_counts) > 5:
            print("\n⚠️ Ingredients dengan resep paling sedikit:")
            for ingredient, count in class_counts.tail(5).items():
                print(f"   {ingredient}: {count} recipes")

        return df

    except Exception as e:
        print(f"\n❌ Error saat loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


# In[10]:


df_recipes = load_ingredient_recipe_dataset("recipes/ingredient_recipes.csv")


# We begin by loading the dataset from a CSV file, ensuring that all necessary columns are present and renaming them for consistency. Next, we perform a data quality check to confirm there are no missing values and report the percentage of recipes that include images. We then provide a summary of the dataset, including the total number of recipes and a breakdown of recipes by ingredient class.
# 
# Following this, we display a sample recipe, showcasing key details like the ingredients, instructions, image URL, and source URL. We also calculate and share additional statistics, such as the average number of ingredients and instructions per recipe. We highlight the top 5 ingredients with the most recipes, as well as those with the fewest.
# 
# Our analysis reveals interesting patterns and anomalies. Ingredients like banana, fish, and shrimp appear frequently in recipes, while others like bread, cucumber, and sweet potato are less common. We observe a clear correlation between ingredient popularity and recipe count, with more common ingredients being better represented. This indicates a noticeable bias toward widely used ingredients in the dataset, which could influence the fairness of model training or analysis.

# # 9.Data Preprocessing

# ## 9.1 Image Preprocessing

# In[11]:


def preprocess_image_ingredient(image_path, target_size=(224, 224)):
    """Preprocess single ingredient image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0

        return img
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None 


# The function preprocess_image_ingredient is designed to prepare a single ingredient image for use in machine learning models by performing several standard preprocessing steps. It reads the image from the specified path, converts the color format from BGR (used by OpenCV) to RGB, resizes the image to a fixed size (default 224x224), and normalizes the pixel values to a range of 0 to 1 for better model performance. If any issues arise during processing, such as the image being unreadable or errors in resizing, it catches the exception and returns None. This process ensures that all images are consistently formatted and ready for input into a neural network, though it may introduce biases if the images are of varying quality or aspect ratios that get distorted during resizing.

# In[12]:


def preprocess_ingredient_dataset(ingredient_classes, target_size=(224, 224)):
    """Preprocess seluruh ingredient dataset"""
    print("\n" + "="*60)
    print("PREPROCESSING INGREDIENT DATASET")
    print("="*60)

    if not ingredient_classes:
        print("❌ Tidak ada kelas ingredient ditemukan untuk diproses.")
        return None, None

    processed_images, labels = [], []
    for ingredient in ingredient_classes:
        ingredient_path = os.path.join(BASE_INGREDIENT_DIR, ingredient)
        if not os.path.exists(ingredient_path):
            continue

        images = [f for f in os.listdir(ingredient_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))]

        print(f"Loading {ingredient}: {len(images)} images")
        for img_file in images:
            img_path = os.path.join(ingredient_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                img = img.astype(np.float32) / 255.0
                processed_images.append(img)
                labels.append(ingredient)
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

    X = np.array(processed_images)
    y = np.array(labels)

    print(f"\n✅ Preprocessing completed!")
    print(f"Total processed images: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print(f"Value range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Label count: {len(np.unique(y))}")
    print(f"Classes: {sorted(np.unique(y))}")

    visualize_preprocessed_ingredient_samples(X, y)
    return X, y


# The function preprocess_ingredient_dataset preprocesses an entire ingredient dataset by iterating through each ingredient class and processing all images within that class. It loads the images from the specified directory, applies standard preprocessing (color conversion to RGB, resizing to the target size, and normalization), and stores the processed images in a list along with their corresponding ingredient labels. If any errors occur during image loading or processing, they are caught and logged, while skipping the problematic images. After processing, the function converts the processed images and labels into numpy arrays and provides a summary of the total number of images, their shape, value range, and unique classes.
# 
# ### **Patterns and Insights**:
# 
# * **Patterns**: The dataset is processed by ingredient class, with each image being preprocessed to a consistent size and format for machine learning models.
# * **Anomalies**: Some images may fail to load or be corrupted, resulting in skipped files and potentially missing data.
# * **Biases**: If some ingredient classes have significantly more images than others, the model may be biased towards the more frequently represented classes. Additionally, the function assumes that all images are of sufficient quality, which may not always be the case.
# 
# ### **Data Check**:
# 
# * **Processed Images**: The function logs the total number of processed images, the shape of the images, and the range of pixel values.
# * **Label Distribution**: It also checks the label distribution to ensure a balanced dataset, listing the unique classes and their counts.
# 

# In[13]:


def visualize_preprocessed_ingredient_samples(X, y, n_samples=6):
    """Visualize preprocessed ingredient images"""
    print("\n" + "="*60)
    print("VISUALIZING PREPROCESSED INGREDIENT IMAGES")
    print("="*60)

    indices = np.random.choice(len(X), n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        img = X[indices[i]]
        img_display = np.clip(img, 0, 1)

        ax.imshow(img_display)
        ax.set_title(f"{y[indices[i]]}", fontsize=11)
        ax.axis('off')

        ax.text(0.02, 0.98, f"[{img.min():.2f}, {img.max():.2f}]",
                transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.suptitle("Preprocessed Ingredient Images", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show() 


# This code defines a function visualize_preprocessed_ingredient_samples(X, y, n_samples=6) to visualize a set of preprocessed ingredient images. The function first prints a header indicating the start of the visualization process. It then randomly selects n_samples images from the dataset X (using np.random.choice), ensuring no duplicates by setting replace=False. The selected images are displayed in a 2x3 grid using plt.subplots(), and for each image, the corresponding label is displayed as the title of the subplot. The pixel values of the images are clipped to the range [0, 1] for proper display, and the minimum and maximum values of the image are shown in the corner of each subplot to give insight into the image's value range.
# 
# The visualization allows us to inspect the appearance of preprocessed images and assess the consistency of the data. Potential patterns could be identified by looking for similarities in the preprocessed images, such as uniform brightness or certain color distributions. Anomalies might arise if any image looks overly dark, bright, or distorted, which could indicate issues in preprocessing. Correlations between the image content and labels can be observed if certain patterns (e.g., ingredient types or textures) are clearly visible in the images. Biases might emerge if certain ingredient classes are overrepresented in the visualization, potentially skewing the model’s understanding of less common classes. The display of image value ranges gives additional insight into the normalization or scaling applied to the images during preprocessing, which can be important for the model's performance.

# In[14]:


X_ingredients, y_ingredients = preprocess_ingredient_dataset(ingredient_classes_loaded) 


# This code begins by calling the visualize_preprocessed_ingredient_samples(X, y, n_samples=6) function to visualize a selection of preprocessed ingredient images. It first selects a random set of n_samples images from the preprocessed dataset (X), where y represents their corresponding labels. The images are displayed in a 2x3 grid using matplotlib, with each image's label shown as the title of the subplot. The pixel values of the images are clipped to the range [0, 1] for proper visualization. Additionally, the minimum and maximum pixel values of each image are displayed in the top-left corner of each subplot. The overall title of the grid is set as "Preprocessed Ingredient Images". This helps to visually assess how the preprocessing has been applied across different ingredient images.
# 
# The visualization offers insights into the appearance of the preprocessed images, allowing us to identify any potential patterns or anomalies. Since the images are preprocessed to have a uniform shape and pixel range, we can see how different ingredients, such as eggplant, shrimp, and bread, are represented. Patterns may include consistency in how textures and colors are preserved after preprocessing. Anomalies could be observed if certain images appear overly bright, dark, or distorted, which might suggest issues with the preprocessing steps. Correlations can be drawn between the ingredient types and their visual characteristics, such as the texture of bread versus shrimp. Biases might emerge if some classes, like bread or shrimp, are overrepresented, which could influence model performance during training. The consistency of preprocessing across different classes is crucial for ensuring fair and effective model learning.
# 
# **Key Points:**
# 
# * The images are randomly selected from the preprocessed dataset and displayed in a 2x3 grid.
# * Labels for each ingredient are shown alongside the images.
# * Patterns include consistency in texture and color across ingredients.
# * Anomalies could arise if any image appears overly bright, dark, or distorted.
# * Correlations are visible between ingredient type and visual features (e.g., texture differences).
# * Biases may occur if some ingredient classes are overrepresented.
# 

# ### 9.2 Data Augmentation

# In[15]:


def create_ingredient_data_augmentation():
    """Create data augmentation for ingredients"""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    return datagen 


# This code defines a function create_ingredient_data_augmentation() that sets up a series of data augmentation techniques for ingredient images using ImageDataGenerator. The augmentation includes a range of transformations such as rotation (up to 20 degrees), width and height shifts (up to 15%), zoom (up to 15%), shear (up to 10%), and horizontal flipping, which helps the model generalize better by introducing more variability in the data. Additionally, the brightness of images is adjusted within a range of 0.8 to 1.2, simulating different lighting conditions. The function returns the datagen object that will apply these augmentations to the images during training.
# 
# These augmentations help improve the model's robustness by simulating real-world variations that the model may encounter, such as changes in angle, size, and lighting of the ingredients. Patterns might emerge if certain transformations, like horizontal flipping, predominantly affect specific ingredient classes (e.g., some ingredients may not be symmetric). Anomalies could appear if augmentation parameters, like extreme rotations or brightness shifts, result in unrealistic images that don’t reflect actual variations in the ingredients, possibly hurting the model’s learning. Correlations between the augmented images and labels can be seen if specific transformations (like zoom or shear) help highlight ingredient features that are important for classification. Lastly, if certain ingredient classes are more likely to undergo particular transformations (e.g., symmetrical ingredients being flipped), there may be biases in how the model learns the features of different ingredients.

# In[16]:


def augment_ingredient_dataset(X, y, augmentation_factor=2):
    """Augment ingredient dataset"""
    print("\n" + "="*60)
    print("APPLYING DATA AUGMENTATION TO INGREDIENTS")
    print("="*60)
    print(f"Augmentation factor: {augmentation_factor}x")

    datagen = create_ingredient_data_augmentation()

    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        img = (X[i] * 255).astype(np.uint8)
        img = img.reshape((1,) + img.shape)
        aug_iter = datagen.flow(img, batch_size=1, shuffle=False)

        for _ in range(augmentation_factor):
            augmented_img = next(aug_iter)[0].astype(np.uint8)
            augmented_img = augmented_img.astype(np.float32) / 255.0
            X_augmented.append(augmented_img)
            y_augmented.append(y[i])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(X)} images")

    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    print(f"\n✅ Augmentation completed!")
    print(f"Original dataset size: {len(X)}")
    print(f"Augmented dataset size: {len(X_augmented)}")
    print(f"Value range: [{X_augmented.min():.3f}, {X_augmented.max():.3f}]")

    visualize_augmented_ingredient_grid(X_augmented, y_augmented)
    return X_augmented, y_augmented


# This code defines a function augment_ingredient_dataset(X, y, augmentation_factor=2) to augment the ingredient dataset by applying data augmentation techniques. First, it prints the augmentation details, including the specified factor (augmentation_factor). It uses a datagen object, created by create_ingredient_data_augmentation(), to apply augmentation transformations to each image in X. The function processes each image, applies augmentations, and stores both the augmented images (X_augmented) and their corresponding labels (y_augmented). The loop runs for the number of augmentations specified by the factor, adding the augmented images to the dataset. Every 100 images processed, a message is printed to track progress.
# 
# At the end, the augmented dataset is converted into NumPy arrays (X_augmented and y_augmented), and the function prints the original and augmented dataset sizes along with the value range of the augmented images. The augmented dataset is then visualized using visualize_augmented_ingredient_grid(X_augmented, y_augmented) to show a grid of the augmented images. This process enhances the model's robustness by generating variations of the original data. However, if the augmentation factor is too high, it may lead to overfitting on the augmented features, especially if some classes are underrepresented in the original dataset, potentially introducing biases. On the other hand, if the augmentation does not properly simulate real-world variations, it might not improve generalization effectively. The goal is to ensure that the augmentations maintain the diversity needed to train the model while avoiding overfitting to specific patterns.

# In[17]:


def visualize_augmented_ingredient_grid(X, y, n_samples=6):
    """Visualize augmented ingredient images"""
    print("\n" + "="*60)
    print("VISUALIZING AUGMENTED INGREDIENT IMAGES")
    print("="*60)

    indices = np.random.choice(len(X), n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        img = X[indices[i]]
        img_display = np.clip(img, 0, 1)

        ax.imshow(img_display)
        ax.set_title(f"{y[indices[i]]}", fontsize=11)
        ax.axis('off')

    plt.suptitle("Augmented Ingredient Images", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# In[18]:


X_aug_ingredients, y_aug_ingredients = augment_ingredient_dataset(X_ingredients, y_ingredients, augmentation_factor=2) 


# This code defines a function visualize_augmented_ingredient_grid(X, y, n_samples=6) to visualize a set of augmented ingredient images. It first randomly selects n_samples images from the augmented dataset (X) and their corresponding labels (y). The selected images are displayed in a 2x3 grid using matplotlib, with each image showing its label as the title. The pixel values are clipped to the range [0, 1] for proper visualization, and the images are displayed without axes to focus on the ingredients. A title is added to the grid, indicating that the images are augmented, and the grid layout is adjusted for a clear view.
# 
# The augmentation process, as indicated in the log, was applied with a factor of 2x, doubling the dataset size from 1666 to 3332 images. After the augmentations, the grid of images shows ingredient samples like lamb, shrimp, bread, potato, chicken, and squid. Patterns could emerge, such as common features like ingredient shapes and textures being preserved even after transformations like rotations or flips. Anomalies might appear if certain augmentations (e.g., excessive rotation or zoom) distort the images too much, making them unrecognizable. Correlations might be observed if the augmented images exhibit variations in appearance but maintain consistency in ingredient identification. Possible biases could occur if certain ingredients are more prone to specific augmentations (e.g., ingredients with symmetrical shapes may be overrepresented in flipped images), which could impact how the model generalizes across ingredient classes.
# 
# **Key Points:**
# 
# * The images are randomly selected from the augmented dataset and displayed in a 2x3 grid.
# * The augmentation includes transformations like rotation, zoom, brightness changes, and flipping.
# * Visual patterns such as consistent ingredient textures can be seen in the augmented images.
# * Anomalies may appear if extreme augmentations distort the images.
# * Augmentation biases may occur if some ingredient classes are more affected by certain transformations.

# In[19]:


def compare_original_vs_augmented(X, y, n_pairs=3):
    """Compare original ingredient images with their augmented versions"""
    print("\n" + "="*60)
    print("COMPARING ORIGINAL vs AUGMENTED INGREDIENT IMAGES")
    print("="*60)

    datagen = create_ingredient_data_augmentation()

    # Pastikan tipe data dan rentang nilai
    if X.dtype != np.float32:
        X = X.astype(np.float32) / 255.0

    # Batasi jumlah pasangan agar tidak lebih dari total data
    n_pairs = min(n_pairs, len(X))
    indices = np.random.choice(len(X), n_pairs, replace=False)

    fig, ax_grid = plt.subplots(n_pairs, 3, figsize=(12, 4 * n_pairs))

    # Jika hanya 1 baris
    if n_pairs == 1:
        ax_grid = ax_grid.reshape(1, -1)

    for row, idx in enumerate(indices):
        original_img = X[idx]
        label = str(y[idx])

        # --- Kolom 1: Gambar Asli ---
        ax_grid[row, 0].imshow(np.clip(original_img, 0, 1))
        ax_grid[row, 0].set_title(f"Original: {label}", fontsize=10)
        ax_grid[row, 0].axis('off')

        # --- Kolom 2 & 3: Augmented Versions ---
        img_uint8 = (original_img * 255).astype(np.uint8)
        img_batch = np.expand_dims(img_uint8, axis=0)
        aug_iter = datagen.flow(img_batch, batch_size=1)

        for col in range(1, 3):
            aug_img = next(aug_iter)[0].astype(np.float32) / 255.0
            ax_grid[row, col].imshow(np.clip(aug_img, 0, 1))
            ax_grid[row, col].set_title(f"Augmented {col}: {label}", fontsize=10)
            ax_grid[row, col].axis('off')

    plt.suptitle("Original vs Augmented Ingredient Comparison", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# The compare_original_vs_augmented function compares the original ingredient images with their augmented versions by visualizing them side-by-side. It randomly selects n_pairs (default is 3) of images from the dataset and displays them in a grid. For each selected image, the function first shows the original image, then generates two augmented versions using the ImageDataGenerator. The augmented images are created through transformations like rotation, zoom, or shifting. The original and augmented images are displayed in a 3-column grid where the first column contains the original image, and the next two columns show the augmented versions. This allows us to visually compare the effects of augmentation on the dataset.
# 
# ### **Analysis of Results**:
# 
# * **Patterns**:  The function effectively demonstrates how augmentations like rotation, zoom, and shifts can generate different variations of the same image, helping to visualize the variety added to the dataset. By comparing the original and augmented images, it’s clear that augmentations increase the diversity of the dataset, making the model more robust to real-world variations.
# * **Anomalies**: Some augmentations may distort the image in ways that could reduce their usefulness, such as extreme zoom or rotation that makes the ingredient unrecognizable. If the original dataset has low-quality or poorly labeled images, the augmented versions might also inherit those flaws, amplifying errors.
# * **Correlations**: There is a positive correlation between dataset size and diversity, as augmentations increase the number of unique variations, potentially improving model generalization and preventing overfitting.
# * **Biases**: Augmentation may not fully correct biases in the dataset. For instance, ingredients with fewer images may still be underrepresented, and the augmented images might not be diverse enough to fully represent all real-world scenarios.

# In[20]:


compare_original_vs_augmented(X_ingredients, y_ingredients, n_pairs=3)


# The visualization compares original ingredient images with two augmented versions for each sample (shrimp, mustard greens, and salmon). The augmentation introduces visual variations such as rotation, zoom, brightness shifts, and distortion, which expand dataset diversity and help improve model generalization. Despite these modifications, the core features of each ingredient remain recognizable, ensuring that the model learns essential characteristics like color, texture, and shape. However, some augmented images (especially for shrimp and mustard greens) show excessive stretching or blurring, which may introduce noise or unrealistic patterns that could mislead the model if overused. Overall, the augmentation process effectively increases training robustness and reduces overfitting risks, but maintaining a balance between realism and variation is crucial to prevent bias toward distorted or artificial image features.

# ## 9.3 Encode Label

# In[21]:


from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.utils import to_categorical

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_aug_ingredients).astype('int32')
y_encoded = to_categorical(y_encoded, num_classes=len(label_encoder.classes_))


class_names = label_encoder.classes_.tolist()
num_classes = len(class_names)

print(f"\n✅ Labels encoded successfully!")
print(f"Class names: {class_names}")
print(f"📦 Jumlah kelas ingredients: {num_classes}")
print(f"\nClass mapping:")
for idx, name in enumerate(class_names):
    print(f"  {idx} -> {name}")
print(f"\nEncoded labels sample: {y_encoded[:10]}") 

print("\n" + "="*70)
print("===== INGREDIENT DETECTION SYSTEM =====")
print("="*70)
print("Sumber Data: Bing Images & Nutrition Databases")
print(f"Dataset Images: '{BASE_INGREDIENT_DIR}'")
print(f"Nutrition Info: '{CSV_FILE}'")
print(f"Jumlah ingredient classes: {num_classes}")
print(f"Total images (after augmentation): {len(X_aug_ingredients)}")
print("="*70)


# In this step, we start by encoding the ingredient labels using the LabelEncoder from scikit-learn and then convert them into one-hot encoded vectors using to_categorical from Keras. This is essential for preparing the labels for classification tasks, as most machine learning models expect numerical values or one-hot encoded labels. We map each ingredient class to a unique index, ensuring consistency in label representation. After encoding, we print out the class names, the number of ingredient classes (18), and a sample of the encoded labels to verify the correctness of the encoding process. This step ensures that the dataset is correctly prepared for model training with proper label formatting.
# 
# Next, we provide an overview of the entire ingredient detection system, including the source of the dataset (Bing Images and Nutrition Databases), the dataset’s location, and the total number of images after augmentation (3,332). We also confirm the number of ingredient classes (18) to ensure the dataset has the expected diversity and coverage. This step is crucial for understanding the dataset's size, structure, and potential scope before starting any model training. By printing these details, we verify that the dataset is ready for the next steps in the development of the ingredient detection model.

# # 10. Preproccesing Recipes

# In[22]:


def preprocess_recipes_text(df_recipes):
    """Preprocessing text resep untuk model generative"""
    print("\n" + "="*70)
    print("🔤 PREPROCESSING RECIPE TEXT")
    print("="*70)

    recipes = []

    for idx, row in df_recipes.iterrows():
        # Format: <ingredient_class> | Ingredients: ... | Instructions: ...
        recipe_text = f"<{row['ingredient_class']}> Ingredients: {row['all_ingredients']} Instructions: {row['instructions']}"
        recipes.append(recipe_text)

    print(f"✅ Processed {len(recipes)} recipes")
    print(f"📝 Average length: {np.mean([len(r) for r in recipes]):.0f} characters")

    return recipes


# In[23]:


recipes_text = preprocess_recipes_text(df_recipes)


# In the function preprocess_recipes_text, we preprocess the recipe text data for use in a generative model by formatting each recipe into a structured text format. For each recipe in the dataset, we concatenate the ingredient class, ingredients, and instructions into a single string, following the format: <ingredient_class> Ingredients: ... Instructions: .... This structured format helps the generative model learn the relationship between ingredients and their respective instructions. After processing all recipes, we calculate and print the total number of recipes processed and the average length of the recipe text. This step prepares the recipe data in a format suitable for training a generative model to generate recipes based on ingredient classes.
# 
# The function preprocess_recipes_text successfully processed 242 recipes from the dataset, as indicated by the log message. Each recipe was formatted into a structured text string, combining the ingredient class, ingredients, and instructions. This structured format is suitable for input into a generative model, where the model can learn the relationships between ingredients and cooking instructions. The average recipe length of 1,167 characters suggests that the recipes are relatively detailed, containing both ingredients and multiple cooking instructions. This length provides enough context for the model to capture relevant information, while also allowing for the generation of diverse and coherent recipes. The preprocessing results indicate that the dataset is well-suited for use in training a model to generate recipes based on ingredient classes.
# 

# ## 10.1 Tokenizer

# In[24]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenizer Keras (bisa atur num_words sesuai kebutuhan vocab)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(recipes_text)

# Konversi teks jadi sequence angka
sequences = tokenizer.texts_to_sequences(recipes_text)

# Padding supaya semua sequence panjangnya sama
padded = pad_sequences(sequences, padding="post", truncating="post", maxlen=200)

print("Jumlah vocab:", len(tokenizer.word_index))
print("Contoh sequence:", sequences[0][:15])
print("Contoh padded:", padded[0][:15])


# In this code, we first initialize a Keras Tokenizer to process the recipe text. We specify a vocabulary size of 10,000 words and include an out-of-vocabulary (<OOV>) token to handle any words not seen during training. The tokenizer then fits on the entire recipes_text dataset, learning the unique words and assigning them integer indices. After this, we convert each recipe text into a sequence of integers, where each integer corresponds to a word's index in the tokenizer’s word index. This step helps transform the text data into a numerical format, which is required for training machine learning models.
# 
# To ensure that all input sequences are of the same length, we apply padding using the pad_sequences function. We specify a maximum sequence length of 200, meaning any sequence longer than that is truncated, and shorter sequences are padded with zeros at the end. This ensures uniformity in the input data, which is necessary for feeding into neural networks. We then print out the vocabulary size (1,844 unique words in this case) and show examples of both the original integer sequences and the padded sequences. This preprocessing step prepares the recipe text data for model input by standardizing the format and handling varyinHg sequence lengths.

# # 11. EDA 

# ## 11.1 Data Sources & Acquisition Overview

# In[25]:


print("===== Data Sources & Acquisition =====")
print("Sumber Data: Bing Images & AllRecipes")
print("Data disimpan di folder: 'dataset_ingredient' & 'ingredient_recipes.csv'")
print("Jumlah kelas (ingedient categories):", len(os.listdir("dataset_ingredients")))


# This code prints information about the data sources and the dataset's structure. It first informs the user that the data comes from Bing Images (for ingredient images) and AllRecipes (for recipe information). The data is stored in two locations: the 'dataset_ingredients' folder (for images) and the 'ingredient_recipes.csv' file (for recipe data). Finally, it prints the total number of ingredient categories by counting the subdirectories in the 'dataset_ingredients' folder, which corresponds to the number of ingredient classes (18). This step provides a clear overview of where the data is stored and the dataset's size, giving us a sense of the scope of the project and the structure of the data before proceeding with analysis or model training.

# ## 12. Image Dataset Analysis

# In[26]:


BASE_INGREDIENT_DIR = "dataset_ingredients"
ingredient_classes = os.listdir(BASE_INGREDIENT_DIR)
ingredient_counts = {cls: len([f for f in os.listdir(os.path.join(BASE_INGREDIENT_DIR, cls)) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
               for cls in ingredient_classes}


# This code defines the base directory BASE_INGREDIENT_DIR as "dataset_ingredients", where ingredient images are stored. It then retrieves the list of ingredient classes (subdirectories) within this base directory using os.listdir(). For each ingredient class, it counts the number of image files with specific extensions (.jpg, .jpeg, .png) by iterating through the files in each class’s directory. This count is stored in a dictionary ingredient_counts, where the keys are ingredient class names and the values are the number of images in each class. This step helps us understand the distribution of images across different ingredient categories and ensures we have an adequate number of images for each class before further processing or training a model.

# ## 12.1 Statistik dasar

# In[27]:


print("Jumlah gambar per kelas:")
for cls, count in ingredient_counts.items():
    print(f"{cls}: {count}")


# This code iterates over the ingredient_counts dictionary and prints the number of images for each ingredient class. The result shows the distribution of images across 18 ingredient classes, with the number of images per class ranging from 56 (carrot) to 117 (sausage), indicating some class imbalances that may require attention during model training to prevent biases toward more frequent classes.

# ## 12.2 Visualisasi distribusi gambar per kelas

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
sns.barplot(x=list(ingredient_counts.keys()), y=list(ingredient_counts.values()))
plt.xticks(rotation=90)
plt.title("Distribusi Gambar per Kelas")
plt.ylabel("Jumlah Gambar")
plt.show()


# The total number of images collected is 1654, with an average of 91 images per class. Based on the visualizations above however, we can observe that the images are not equally distributed between each class. Some classes contain significantly more images, and some have significantly less images than average, with most ingredients having counts between 70 and 110 images. The class with the most images is “mustard greens”, with a total of 152 images. Meanwhile, the class with the least amount of images is “carrots” with only 56 images in its class, followed by “tofu” with only 60 images.
# 
# The differences between the amount of images in each class shows that there is a class imbalance in the data. This imbalance has the potential to impact how the model would perform, as these images are used to train the model for the classification task. With some classes having more or less images than the average, the model might perform better on classes that have a large set of images (like mustard greens or chicken), while smaller classes (like carrot or tofu) might be underrepresented, leading to potential bias in prediction accuracy.

# ## 12.3 Pie chart proporsi dataset

# In[29]:


plt.figure(figsize=(8,8))
plt.pie(list(ingredient_counts.values()), labels=list(ingredient_counts.keys()), autopct='%1.1f%%')
plt.title("Proporsi Dataset Gambar")
plt.show()


# This pie chart represents the distribution of images across various ingredient categories in the dataset. Each slice represents the proportion of images belonging to a specific class, revealing that mustard greens (9.2%), sausage (7.1%), and chicken (7.0%) are the most frequent categories, while carrot (3.4%), tofu (3.6%), and broccoli (4.2%) have the fewest samples. This suggests a mild class imbalance, where some ingredients are overrepresented compared to others. Such imbalance may introduce bias in model training, causing the model to predict more accurately for dominant classes while underperforming on less frequent ones. Overall, the dataset shows moderate diversity but could benefit from rebalancing techniques, such as data augmentation or resampling, to ensure fairer and more reliable model performance.

# ## 12.4 Dimensi rata-rata & ukuran file

# In[30]:


dims, sizes = [], []
for cls in ingredient_classes:
    for f in os.listdir(os.path.join(BASE_INGREDIENT_DIR, cls)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_INGREDIENT_DIR, cls, f)
            img = Image.open(img_path)
            dims.append(img.size)  # (width, height)
            sizes.append(os.path.getsize(img_path)/1024)  # KB

dims = np.array(dims)
print(f"Dimensi rata-rata (WxH): {dims[:,0].mean():.1f} x {dims[:,1].mean():.1f}")
print(f"Ukuran file rata-rata: {np.mean(sizes):.1f} KB")


# Based on the information above, the average image dimension of all images is 1122 x 896 pixels and the average file size is 213.5 KB.

# ## 12.5 Scatter plot dimensi

# In[31]:


plt.figure(figsize=(8,6))
plt.scatter(dims[:,0], dims[:,1], alpha=0.5)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Scatter Plot Dimensi Gambar")
plt.show()


# The scatterplot above shows us the distribution of image dimensions between the images within the dataset. We can observe that the majority of images have a resolution between 100-2000 pixels for both width and height, with the remaining having more than 2000 pixels and some being as big as 5000 x 6000 pixels in resolution. This means that the images that are represented in the data are mostly within the 100 to 2000 pixels resolution range.
# 

# ## 12.6 Bar chart ukuran file

# In[32]:


plt.figure(figsize=(10,5))
sns.histplot(sizes, bins=30)
plt.title("Distribusi Ukuran File Gambar (KB)")
plt.show()


# Figure 16 presents a histogram illustrating the distribution of image file sizes in kilobytes across the dataset. Most images are concentrated on the left side of the graph, under 500 KB, showing that most of the image files are relatively small. The distribution is right-skewed, with a long tail extending toward larger file sizes, indicating a few outliers (Example : images that are significantly larger, likely due to higher resolution or different compression formats). This skew suggests potential inconsistencies in image preprocessing or data sourcing, which could impact model performance or loading times if not addressed. To ensure consistency and efficiency during training, it may be necessary to standardize image sizes or compress the larger outliers to reduce variability. 

# # 13. Recippe Dataset Analysis

# In[33]:


print("Jumlah resep:", len(df_recipes))
print("\nMissing values per kolom:\n", df_recipes.isna().sum())


# This code prints the total number of recipes in the dataset (len(df_recipes)) and checks for missing values in each column using isna().sum(). The result shows that there are 242 recipes, and no missing values in any of the columns (ingredient class, recipe name, all ingredients, instructions, image URL, and source URL), indicating that the dataset is complete and ready for further processing.
# 

# ## 13.1 Distribusi resep per kelas

# In[34]:


plt.figure(figsize=(12,5))
sns.countplot(data=df_recipes, x='ingredient_class')
plt.xticks(rotation=90)
plt.title("Jumlah Resep per Kelas")
plt.show()


# The bar chart shows about the recipe distribution by class. This shows the number of recipes associated with each ingredient class in the dataset. Each bar represents how many recipes use a particular ingredient, such as banana, chicken, tofu, and others. The visualization reveals that most ingredient classes have a fairly similar number of recipes, typically around 13–15, indicating a generally balanced dataset in terms of recipe distribution. However a few ingredients such as chicken, squid, and sweet potato, therefore suggesting a minor imbalance that could influence model generalization if used for classification or recommendation tasks. Overall, the dataset appears diverse and well-distributed, minimizing the risk of strong bias toward specific ingredient types.

# ## 13.2 Text statistics: ingredients & instructions

# In[35]:


df_recipes['num_ingredients'] = df_recipes['all_ingredients'].apply(lambda x: len(str(x).split(',')))
df_recipes['num_instructions'] = df_recipes['instructions'].apply(lambda x: len(str(x).split('.')))

plt.figure(figsize=(12,5))
sns.histplot(df_recipes['num_ingredients'], bins=20, kde=True)
plt.title("Distribusi Jumlah Bahan per Resep")
plt.show()

plt.figure(figsize=(12,5))
sns.histplot(df_recipes['num_instructions'], bins=20, kde=True)
plt.title("Distribusi Jumlah Instruksi per Resep")
plt.show()


# The histogram (above) shows the distribution of ingredient amounts. This reveals how many ingredients are typically used in each recipe within the dataset. The majority of recipes contain between 2 and 5 ingredients, with the highest frequency occurring around 3 ingredients, indicating that most recipes are relatively simple. The distribution is right-skewed, meaning there are fewer recipes with a large number of ingredients, and only a small number of recipes use more than 8 ingredients. This pattern suggests that the dataset mainly consists of concise recipes, possibly designed for quick or basic meal preparation. The lack of recipes with extensive ingredient lists may introduce a bias toward simpler dishes, potentially limiting the model’s ability to learn if the goal is to predict or generate complex recipes.
# 
# The histogram (below) shows the distribution of ingredient amounts. This reveals how many ingredients are typically used in each recipe within the dataset. The majority of recipes contain between 2 and 5 ingredients, with the highest frequency occurring around 3 ingredients, indicating that most recipes are relatively simple. The distribution is right-skewed, meaning there are fewer recipes with a large number of ingredients, and only a small number of recipes use more than 8 ingredients. This pattern suggests that the dataset mainly consists of concise recipes, possibly designed for quick or basic meal preparation. The lack of recipes with extensive ingredient lists may introduce a bias toward simpler dishes, potentially limiting the model’s ability to learn if the goal is to predict or generate complex recipes.

# ## 13.3 Boxplot bahan per kelas

# In[36]:


plt.figure(figsize=(12,5))
sns.boxplot(data=df_recipes, x='ingredient_class', y='num_ingredients')
plt.xticks(rotation=90)
plt.title("Boxplot Jumlah Bahan per Kelas")
plt.show()


# The  boxplot visualizing the number of ingredients per class. Each box represents the median, quartiles, and potential outliers for each ingredient class. Most classes have a median ingredient count between 3 and 5, indicating relatively consistent recipe complexity. However, ingredient classes like chicken, mustard greens, and tempeh exhibit higher variability, suggesting that recipes with these ingredients tend to have a wider range of ingredient counts. In contrast, classes such as shrimp, sausage, and tofu show exceptionally complex recipes with a large number of ingredients. This variation highlights differences in recipe composition across ingredient types and suggests that certain ingredients are linked to more intricate or customizable dishes, which could influence model training if recipe complexity is considered a predictive factor.
# 

# # 14. Split Data Images

# In[37]:


from sklearn.model_selection import train_test_split
import numpy as np

def split_data(X, y, test_size=0.1, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets (approximate multi-label stratification)"""

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    X, y = X[indices], y[indices]

    # Split test
    test_count = int(len(X) * test_size)
    X_test, y_test = X[:test_count], y[:test_count]
    X_temp, y_temp = X[test_count:], y[test_count:]

    # Split validation
    val_count = int(len(X_temp) * val_size)
    X_val, y_val = X_temp[:val_count], y_temp[:val_count]
    X_train, y_train = X_temp[val_count:], y_temp[val_count:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Total: {len(X_train) + len(X_val) + len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# In[38]:


X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X_aug_ingredients, y_encoded
)


# In this code, we split the dataset to divide the augmented ingredient images (X_aug_ingredients) and their corresponding encoded labels (y_encoded) into three parts: training, validation, and testing. This ensures the model can learn effectively while also being evaluated on new, unseen data to prevent overfitting. First, we shuffle the image indices randomly using a fixed seed (random_state=42) to ensure reproducibility and eliminate any ordering bias. Then, we allocate 10% of the data for testing and split another 20% of the remaining data for validation. The largest portion is used for training, where the model learns to identify visual patterns.
# 
# This process creates a balanced and fair evaluation framework, with each subset serving a distinct purpose in the machine learning workflow. It allows us to evaluate both the model's learning and its ability to generalize. However, if the dataset is small or imbalanced, there could be biases, such as certain ingredient classes being overrepresented in one subset. To address this, methods like stratified sampling or data augmentation can help mitigate these issues.
# 
# * **Training set (2400 images):** used for model learning.
# * **Validation set (599 images):** used to tune model parameters and detect overfitting.
# * **Test set (333 images):** used for final performance evaluation on unseen data.
# 

# ## 14.1 Split Data Recipes

# In[39]:


def split_recipe_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Pisahkan dataset resep menjadi train, validation, dan test sets.
    """
    print("\n" + "="*60)
    print("SPLITTING RECIPE DATASET")
    print("="*60)

    if df is None or df.empty:
        print("Data kosong! Pastikan sudah melalui tahap preprocessing.")
        return None, None, None

    # Split test set first
    df_temp, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Split validation set from remaining data
    val_fraction = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_temp, test_size=val_fraction, random_state=random_state
    )

    print(f"Train set: {len(df_train)} recipes")
    print(f"Validation set: {len(df_val)} recipes")
    print(f"Test set: {len(df_test)} recipes")
    print(f"Total: {len(df_train) + len(df_val) + len(df_test)} recipes")

    # Contoh hasil
    print("\nContoh data train:")
    print("Nama Resep:", df_train.iloc[0]['recipe_name'])
    print("Ingredients:", df_train.iloc[0]['all_ingredients'][:150] + "...")
    print("Instructions:", df_train.iloc[0]['instructions'][:200] + "...")

    print("\nSplit completed!")

    return (
        df_train.reset_index(drop=True), 
        df_val.reset_index(drop=True), 
        df_test.reset_index(drop=True)
    )


# In[40]:


df_train, df_val, df_test = split_recipe_dataset(
    df_recipes,
    test_size=0.15, 
    val_size=0.15, 
    random_state=42
)


# In this code, we divide the recipe dataset into three parts: training, validation, and testing sets. We do this to ensure that our model learns effectively while also being evaluated fairly on unseen data. The training set is used for the learning process, the validation set helps tune parameters and prevent overfitting, and the test set measures the model’s final performance. By splitting the dataset randomly (using a fixed seed for reproducibility), we maintain a balanced representation of various recipe types, ensuring that no specific category dominates one subset and causes biased learning.
# 
# Through this process, we can also identify possible data-related patterns or issues.
# 
# * **Patterns:** We may notice trends such as popular ingredients or frequent cooking styles appearing across splits.
# * **Anomalies:** Some recipes might have missing, incomplete, or duplicated information.
# * **Correlations:** Ingredients or instructions might correlate strongly with specific cuisines or preparation methods.
# * **Biases:** If certain recipe types are overrepresented, the model might perform better on those and worse on underrepresented ones.
# 

# # 15. Model Building

# ## 15.1 Model Images

# In[41]:


def build_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Build single-label classification model (softmax output)"""

    # Use MobileNetV2 as base model 
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',  
        metrics=['accuracy']              
    )


    return model, base_model


# In this code, we create a deep learning model for image classification using MobileNetV2 as the base feature extractor. We begin by loading MobileNetV2 with pre-trained weights from ImageNet, taking advantage of the visual features it has already learned. By freezing its layers during the initial training phase, we ensure they remain unchanged, allowing the model to retain general visual knowledge such as edges, shapes, and textures. On top of the base model, we add new layers: global average pooling to reduce dimensions, dense layers with ReLU activation to capture complex patterns, and dropout layers to reduce overfitting. The final layer uses softmax activation to output probabilities for each recipe class. We compile the model with the Adam optimizer and categorical crossentropy loss, which are ideal for multi-class classification tasks.
# 
# Through this design, we can identify possible data or model-related patterns and issues.
# 
# * **Patterns:** The model may learn to associate certain colors or textures with specific dishes or ingredients.
# * **Anomalies:** Images with poor lighting, unusual angles, or incorrect labels may confuse the model.
# * **Correlations:** Visual similarities (e.g., soups with similar color) might cause overlapping predictions between classes.
# * **Biases:** If the dataset has more images of certain cuisines or cooking styles, the model might favor those categories.
# 

# In[42]:


def unfreeze_base_model(model, base_model, learning_rate=1e-5):
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# In[43]:


num_classes = len(ingredient_classes)-1
model, base_model = build_model(num_classes)


# In this code, we perform a fine-tuning process to enhance the performance of our pre-trained MobileNetV2 model. Initially, we freeze most of the base model layers to retain the general visual knowledge it learned from ImageNet. Then, within this function, we “unfreeze” only the last 20 layers of the base model, which capture more specific, high-level patterns relevant to our recipe dataset. By retraining just a small portion of the model, we allow it to learn detailed patterns specific to our data, such as food textures and ingredient shapes, while still maintaining the general features learned during its original training. The model is then recompiled with a lower learning rate (1e-5) to ensure gradual updates and prevent disrupting the pre-trained weights.
# 
# From this step, several insights and considerations may arise:
# 
# * **Patterns:** The fine-tuned layers may capture detailed distinctions between visually similar foods.
# * **Anomalies:** Overfitting can occur if the dataset is small or imbalanced during fine-tuning.
# * **Correlations:** The model might become overly sensitive to minor visual cues that correlate with certain labels.
# * **Biases:** If certain recipe types dominate the dataset, the model may skew predictions toward those categories.

# ## 15.2 Phase 1: Feature Extraction (Training Awal / “Frozen”)

# In[44]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

def train_model(model, X_train, y_train, X_val, y_val, initial_epochs=20):
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
            monitor='val_accuracy',   
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=initial_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history1


# In this code, we train our image classification model while using callbacks to optimize performance and prevent overfitting. The function defines three key mechanisms: EarlyStopping, which halts training if validation accuracy stops improving for five epochs; ModelCheckpoint, which automatically saves the best-performing model based on validation accuracy; and ReduceLROnPlateau, which lowers the learning rate when the validation loss stagnates to help the model escape local minima. These callbacks ensure that training is efficient, stable, and avoids unnecessary epochs that could lead to overfitting or wasted computation. The model is trained on the training dataset and validated on unseen validation data to monitor learning progress and generalization ability.
# 
# Through this training process, we can observe and analyze several important patterns and potential issues:
# * **Patterns:** Gradual improvement in accuracy and decreasing validation loss indicate effective learning.
# * **Anomalies:** Sudden drops or fluctuations in accuracy may suggest noisy data or poor learning rate tuning.
# * **Correlations:** Consistent gaps between training and validation accuracy may reveal overfitting or underfitting.
# * **Biases:** If certain classes dominate the training set, the model may become biased toward those, achieving high accuracy on common classes but poor generalization on rare ones.

# In[45]:


history1 = train_model(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    initial_epochs=20
)


# In this training process, the model is trained using the train_model() function, which includes optimization callbacks to ensure stable learning. During each epoch, the model adjusts its parameters to minimize categorical crossentropy loss and improve classification accuracy. The results show that the model starts with an accuracy of 33.29% in Epoch 1 and quickly reaches 94.66% validation accuracy by Epoch 18, with the loss consistently decreasing. The EarlyStopping and ModelCheckpoint callbacks ensure that training stops when improvements plateau, and the best-performing model is saved. Additionally, the ReduceLROnPlateau callback reduces the learning rate when validation loss stagnates, allowing smoother fine-tuning and preventing the model from overshooting optimal values in later epochs.
# 
# Analyzing the results reveals several key insights and implications:
# 
# * **Initial accuracy:** 33% → 94.7% (rapid improvement across epochs)
# * **Validation accuracy:** Increases steadily, reaching around 94.7%, then levels off
# * **Validation loss:** Drops until around Epoch 10, then oscillates slightly
# * **Learning rate:** Decays over time with **ReduceLROnPlateau**
# * **Training loss:** Continues to decrease even after validation loss stabilizes
# 
# The base (frozen) model quickly learned general features, as it:
# 
# * Started with pre-trained convolutional features (likely from ImageNet),
# * Learned dataset-specific decision boundaries through the top dense layers,
# * And reached high performance quickly.
# 
# However, between Epochs 10 and 20:
# 
# * Training loss kept decreasing,
# * Validation loss and accuracy stagnated.
# 
# This pattern indicates mild overfitting, where the model starts memorizing details from the training data that don't generalize well to the validation set, but the overfitting isn't severe.
# 

# ## 15.3 Phase 2: Training Fine-Tuning (Unfreeze)

# In[46]:


def fine_tune_model(model, base_model, X_train, y_train, X_val, y_val, fine_tune_epochs=30):
    """Fine-tune the model with unfrozen base layers"""
    print("\n" + "="*60)
    print("FINE-TUNING MODEL")
    print("="*60)

    # Unfreeze base model
    model = unfreeze_base_model(model, base_model)

    # Callbacks for fine-tuning
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]

    # Fine-tune model
    print(f"\nPhase 2: Fine-tuning for {fine_tune_epochs} epochs...")
    history2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=fine_tune_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return model, history2


# In[47]:


model, history2 = fine_tune_model(
    model, 
    base_model, 
    X_train, y_train, 
    X_val, y_val, 
    fine_tune_epochs=30
)


# In Phase 2: Fine-Tuning (Unfreeze), we unfreeze some of the lower layers of the pre-trained MobileNetV2 model to allow it to make more targeted adjustments based on our dataset. This follows Phase 1: Feature Extraction (Frozen), where only the newly added top layers were trained while the base model remained frozen. During fine-tuning, the model refines its previously learned features using a lower learning rate (1e-5) to prevent overwriting the general patterns learned during pretraining. To stabilize the process and reduce the risk of overfitting, we use the EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks.
# 
# The fine-tuning process stopped early at epoch 6, triggered by the EarlyStopping callback, which detected no improvement in validation accuracy for five consecutive epochs. Compared to Phase 1, the accuracy gains in Phase 2 were smaller because the model was already highly optimized (≈94–95% validation accuracy). Fine-tuning generally leads to modest improvements, refining smaller features rather than making large gains. This is expected since the model had already reached a strong performance during Phase 1, and further training could risk overfitting.
# 
# The analysis of the results reveals several key insights:
# 
# * **Training accuracy:** Starts high (~86%) and reaches 94.5% in 5 epochs.
# * **Validation accuracy:** Starts at ~94.5% and does not show significant improvement.
# * **Validation loss:** Remains steady around 0.25.
# * **Training stopped early (epoch 6)** due to early stopping.
# 
# This behavior aligns with expectations for a fine-tuning phase on a well-converged base model:
# 
# * When unfreezing and fine-tuning, most weights (particularly in convolutional layers) are already optimized.
# * The model doesn't have much new information to learn, only subtle adjustments to its feature representations.
# * Validation metrics plateau early as the model reaches its generalization limit.
# 
# The small learning rate (1e-5 → 2e-6) meant that updates were gradual, helping prevent overfitting while refining high-level features. The fact that early stopping triggered after just 6 epochs further highlights the training stability.
# 

# # 16. Visualisasi Hasil Training

# ## 16.1 Hasil Training (Frozen)

# In[48]:


plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history1.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Food Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label='Train Loss', color='blue')
plt.plot(history1.history['val_loss'], label='Val Loss', color='orange')
plt.title('Food Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In this code, we visualize the performance of our food classification model during Phase 1: Feature Extraction, where certain layers are frozen to focus on feature extraction rather than learning new weights. The first subplot tracks the accuracy trends, where the training accuracy (blue line) increases rapidly, indicating that the model quickly learns from the training data. However, the validation accuracy (orange line) increases more slowly and eventually plateaus, suggesting that while the model learns well from the training set, it struggles to generalize to new, unseen data. This indicates potential overfitting, as the model might not effectively apply the learned features to the validation set. In the second subplot, the loss curves show that the training loss decreases quickly at first but then flattens, while the validation loss drops more slowly, another sign of overfitting, as the model becomes too specialized in the training data.
# 
# From this visualization, we can conclude several things about Phase 1: Feature Extraction. The model shows fast improvement in both training accuracy and loss, but the validation accuracy and loss indicate that the model’s ability to generalize is somewhat limited. The main observation is that the validation accuracy stagnates while the training accuracy continues to rise, suggesting that the model could benefit from methods to prevent overfitting, such as regularization or unfreezing layers for further fine-tuning in subsequent phases.
# 
# 1. **Accuracy trend:**
# - **Training accuracy (blue)** steadily increases and almost reaches 1.0 (or 99%).
# - **Validation accuracy (orange)** rises quickly but levels off around 95%.
# - The validation curve remains stable, while training accuracy continues to rise, with minimal decrease.
# 
# 2. **Loss trend:**
# - **Training loss (blue)** keeps decreasing and eventually stabilizes near 0.05.
# - **Validation loss (orange)** decreases sharply initially and then stabilizes around 0.4–0.5 without increasing.
# 
# 3. **Gap analysis:**
# - A small gap exists between training and validation loss/accuracy.
# - This gap does not widen significantly after around epoch 10, indicating that the model is generalizing reasonably well and not overfitting extensively.

# ## 16.2 Hasil Training Fine-Tuning (Unfreeze)

# In[49]:


plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history2.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history2.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Food Ingredient Model Accuracy (Fine-Tune)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history2.history['loss'], label='Train Loss', color='blue')
plt.plot(history2.history['val_loss'], label='Val Loss', color='orange')
plt.title('Food Ingredient Model Loss (Fine-Tune)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In this code, we visualize the performance of the food ingredient model during the **fine-tuning phase (Phase 2: Unfreeze)**, where additional layers are unfrozen to allow for more flexibility in adapting the model to the specific task. The first subplot shows the accuracy of both the training and validation sets over the epochs. We observe that the training accuracy (blue line) jumps significantly in the first few epochs, reaching nearly 94%, while the validation accuracy (orange line) also rises, but at a slower pace. This suggests that after fine-tuning, the model fits both the training and validation data well, and the validation accuracy is steadily improving. In the second subplot, the loss curves show a sharp decrease in training loss (blue) during the first few epochs, while the validation loss (orange) decreases more gradually. Both loss curves stabilize after a few epochs, with the validation loss plateauing at a lower value compared to Phase 1 (Feature Extraction), indicating that fine-tuning has enhanced the model’s ability to generalize.
# 
# From this visualization, we can conclude that fine-tuning has successfully improved the model's generalization to the validation set, as evidenced by the consistent improvements in both accuracy and loss. The key takeaway is that unfreezing additional layers has allowed the model to continue learning and generalizing, with both accuracy and loss metrics showing positive trends. However, the slower rate of increase in validation accuracy, compared to training accuracy, still hints at some minor overfitting, though it is much less pronounced than in Phase 1.
# 
# 1. **Accuracy (left plot):**
# 
# - **Training accuracy (blue)** increases steadily from around 86% to 94.5%.
# - **Validation accuracy (orange)** starts high (~94.5%) and remains stable with only slight fluctuations.
# - The validation accuracy does not drop as training accuracy rises, which is a positive indicator.
# 
# 2. **Loss (right plot):**
# - **Training loss (blue)** decreases rapidly from ~0.45 to ~0.15, as expected during fine-tuning.
# - **Validation loss (orange)** starts around 0.24 and remains fairly stable, with slight increases between epochs 1–3, then stabilizes.
# - The validation loss curve is stable overall, with no sharp increases.
# 
# ### 3. **Gap between curves:**
# 
# - There is a small gap between the training and validation loss curves.
# - This gap slightly widens toward the end as the training loss continues to decrease, while the validation loss plateaus.

# In[50]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_model(model, X_test, y_test, class_names):
    """Evaluate model performance for single-label classification"""
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)

    # Predict probabilities
    y_pred_proba = model.predict(X_test)

    # Convert probabilities → class index
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)  # y_test is one-hot encoded

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Overall Test Accuracy: {acc:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Return detailed results
    df_results = pd.DataFrame({
        "True Label": [class_names[i] for i in y_true],
        "Predicted Label": [class_names[i] for i in y_pred]
    })

    return y_pred, y_pred_proba, df_results


# In[51]:


y_pred, y_pred_proba, df_results = test_model(model, X_test, y_test, class_names=class_names)


# In conclusion, the model performs exceptionally well with an overall test accuracy of 96.40%, indicating strong generalization across most classes. However, there are slight variations in performance across different food categories, particularly for less frequent classes like "carrot," "squid," and "tofu." These discrepancies may be attributed to the challenges of distinguishing certain food items or the class imbalance in the dataset. Despite these minor issues, the model's precision, recall, and F1-scores are generally very high, reflecting a solid ability to classify most food items correctly.
# 
# * **Overall Performance:** High overall test accuracy of 96.40%, indicating strong generalization.
# * **High Precision and Recall:** Many classes have precision, recall, and F1-scores close to 1.00, particularly for common items like "banana" and "broccoli."
# * **Lower Performance for Some Classes:** Categories like "carrot," "squid," and "tofu" show slightly lower metrics, hinting at difficulties in prediction.
# * **Class Imbalance Impact:** The lower performance in certain classes suggests a possible effect of class imbalance.
# * **Macro and Weighted Averages:** Both averages around 0.96, showing consistent model performance across different classes with minor improvement needed for difficult categories.
# 

# ## 5: RECIPE GENERATION WITH PRE-TRAINED GPT-2

# In[52]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Embedding, 
    Dropout, Attention, Concatenate, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random


# ## 5.1 PREPARE RECIPE TEXT DATA FOR TRAINING

# In[53]:


def prepare_recipe_training_data(df, max_len=200):
    """
    Prepare text data for sequence-to-sequence model
    Format: <ingredient_class> -> Recipe (ingredients + instructions)
    """
    print("\n" + "="*70)
    print("📝 PREPARING RECIPE TRAINING DATA")
    print("="*70)

    input_texts = []  # Input: ingredient class name
    target_texts = []  # Output: full recipe

    for idx, row in df.iterrows():
        # Input: ingredient class
        input_text = f"{row['ingredient_class']}"

        # Target: Recipe with special tokens
        recipe = f"<START> Recipe Name: {row['recipe_name']}. "
        recipe += f"Ingredients: {row['all_ingredients']}. "
        recipe += f"Instructions: {row['instructions']} <END>"

        input_texts.append(input_text)
        target_texts.append(recipe)

    print(f"✅ Prepared {len(input_texts)} training samples")
    print(f"\n📊 Sample Input-Output Pair:")
    print(f"Input: {input_texts[0]}")
    print(f"Target: {target_texts[0][:200]}...")

    return input_texts, target_texts


# ### Prepare data from train set

# In[54]:


input_texts_train, target_texts_train = prepare_recipe_training_data(df_train)
input_texts_val, target_texts_val = prepare_recipe_training_data(df_val)


# ## INSTALL AND IMPORT DEPENDENCIES

# In[55]:


try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers import TextDataset, DataCollatorForLanguageModeling
    import torch
except ImportError:
    print("📥 Installing transformers and torch...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
    from transformers import TextDataset, DataCollatorForLanguageModeling
    import torch


# ## PREPARE DATA FOR GPT-2 FINE-TUNING

# In[56]:


def prepare_gpt2_data(df, output_file):
    """
    Format recipes for GPT-2 training
    Each recipe is formatted as:
    [INGREDIENT: x] Recipe Name: y | Ingredients: z | Instructions: w [END]
    """
    print(f"\n📝 Preparing data for {output_file}...")

    training_texts = []

    for idx, row in df.iterrows():
        # Format each recipe
        text = f"[INGREDIENT: {row['ingredient_class']}]\n"
        text += f"Recipe Name: {row['recipe_name']}\n\n"
        text += f"Ingredients:\n{row['all_ingredients']}\n\n"
        text += f"Instructions:\n{row['instructions']}\n"
        text += "[END]\n\n"
        training_texts.append(text)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(training_texts))

    print(f"✅ Saved {len(training_texts)} recipes")
    print(f"📊 Total characters: {sum(len(t) for t in training_texts):,}")

    return training_texts

# Create recipe directory if not exists
RECIPE_TEXT_DIR = "./recipe_texts/"
os.makedirs(RECIPE_TEXT_DIR, exist_ok=True)

# Prepare train and validation data
train_file = os.path.join(RECIPE_TEXT_DIR, 'train_recipes.txt')
val_file = os.path.join(RECIPE_TEXT_DIR, 'val_recipes.txt')

train_texts = prepare_gpt2_data(df_train, train_file)
val_texts = prepare_gpt2_data(df_val, val_file)

print(f"\n📦 Data Summary:")
print(f"✅ Training samples: {len(train_texts)}")
print(f"✅ Validation samples: {len(val_texts)}")

# Show sample
print(f"\n📋 Sample Training Data:")
print("="*70)
print(train_texts[0])
print("="*70)


# ## LOAD PRE-TRAINED GPT-2 MODEL

# In[57]:


print("\n" + "="*70)
print("📥 LOADING PRE-TRAINED GPT-2 MODEL")
print("="*70)

# Model size options: 'gpt2' (124M), 'gpt2-medium' (355M), 'gpt2-large' (774M)
MODEL_NAME = 'gpt2'  

print(f"Loading {MODEL_NAME}...")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
gpt2_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Configure padding token
tokenizer.pad_token = tokenizer.eos_token
gpt2_model.config.pad_token_id = tokenizer.eos_token_id

print(f"✅ Model loaded successfully!")
print(f"📊 Model parameters: {gpt2_model.num_parameters():,}")
print(f"📊 Vocabulary size: {len(tokenizer):,}") 


# ## PREPARE DATASETS FOR TRAINING

# In[58]:


print("\n" + "="*70)
print("🔧 PREPARING DATASETS FOR TRAINING")
print("="*70)

def load_dataset(file_path, tokenizer, block_size=512):
    """Load and tokenize text dataset"""
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

# Load datasets
train_dataset = load_dataset(train_file, tokenizer, block_size=512)
val_dataset = load_dataset(val_file, tokenizer, block_size=512)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal Language Modeling (not Masked LM)
)

print(f"✅ Train dataset blocks: {len(train_dataset)}")
print(f"✅ Validation dataset blocks: {len(val_dataset)}")


# ## FINE-TUNE GPT-2 ON RECIPE DATA

# In[59]:


print("\n" + "="*70)
print("🚀 FINE-TUNING GPT-2 ON RECIPE DATA")
print("="*70)

# Training configuration
training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, 'gpt2_recipe_checkpoints'),
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",      
    eval_steps=50,              
    save_strategy="steps",       
    save_steps=100,
    warmup_steps=100,
    logging_steps=20,
    logging_dir=os.path.join(MODEL_DIR, 'logs'),
    save_total_limit=3,
    load_best_model_at_end=True,    
    metric_for_best_model="loss",    
    greater_is_better=False,
    fp16=False,
    report_to='none'
)


# Initialize Trainer
trainer = Trainer(
    model=gpt2_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train the model
training_result = trainer.train() 


# In[67]:


def generate_recipe_improved(ingredient, model, tokenizer, 
                            max_length=500, temperature=0.7, 
                            top_k=50, top_p=0.92):
    """Generate recipe dengan parameter yang lebih baik"""
    prompt = f"<|startoftext|>\nINGREDIENT: {ingredient}\n\nRECIPE NAME:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            min_length=100,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # ← PENTING!
            no_repeat_ngram_size=3,  # ← PENTING!
            early_stopping=True
        )

    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe


# In[60]:


eval_results = trainer.evaluate()
print(eval_results)


# ## SAVE FINE-TUNED MODEL

# In[61]:


print("\n" + "="*70)
print("💾 SAVING FINE-TUNED MODEL")
print("="*70)

final_model_dir = os.path.join(MODEL_DIR, 'gpt2_recipe_final')
gpt2_model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"✅ Model saved to: {final_model_dir}")


# ## PLOT TRAINING HISTORY

# In[62]:


print("\n" + "="*70)
print("📈 PLOTTING TRAINING HISTORY")
print("="*70)

# Extract training history
if hasattr(trainer.state, 'log_history'):
    train_losses = []
    eval_losses = []
    steps = []
    eval_steps = []

    for log in trainer.state.log_history:
        if 'loss' in log:
            train_losses.append(log['loss'])
            steps.append(log.get('step', len(train_losses)))
        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
            eval_steps.append(log.get('step', len(eval_losses)))

    if train_losses and eval_losses:
        plt.figure(figsize=(14, 5))

        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(steps, train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Validation loss
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, eval_losses, label='Eval Loss', color='orange', linewidth=2)
        plt.title('Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"✅ Best validation loss: {min(eval_losses):.4f}")
        print(f"✅ Final validation loss: {eval_losses[-1]:.4f}")


# ## RECIPE GENERATION FUNCTION

# In[74]:


def generate_recipe_gpt2(
    ingredient_name,
    model,
    tokenizer,
    max_length=500,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
):
    """
    Generate recipe using fine-tuned GPT-2 with cleaner formatting.
    FIXED: ensures model & tensors stay on CPU to avoid device mismatch.
    """

    import torch
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Prompt mengikuti format training:
    prompt = (
        f"[INGREDIENT: {ingredient_name}]\n"
        f"Recipe Name:"
    )

    # Encode → force CPU
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=3,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    recipes = []

    for i in range(num_return_sequences):
        text = tokenizer.decode(output[i], skip_special_tokens=True)

        # Remove the prompt part
        recipe = text[len(prompt):].strip()

        # Cut at END token
        if "[END]" in recipe:
            recipe = recipe.split("[END]")[0].strip()



        recipes.append(recipe)

    return recipes[0] if num_return_sequences == 1 else recipes


# ## TEST RECIPE GENERATION

# In[75]:


print("\n" + "="*70)
print("🧪 TESTING GPT-2 RECIPE GENERATION (CLEAN FORMAT)")
print("="*70)

test_ingredients = ['chicken', 'salmon', 'tofu', 'broccoli', 'potato']

for ing in test_ingredients:
    print(f"\n{'='*70}")
    print(f"🥘 INGREDIENT: {ing.upper()}")
    print(f"{'='*70}\n")

    recipe = generate_recipe_gpt2(
        ingredient_name=ing,
        model=gpt2_model,
        tokenizer=tokenizer,
        temperature=0.7,
        top_p=0.9
    )

    # Cetak hasil dengan format rapi
    print("📋 Generated Recipe:")
    print("-"*70)
    print(recipe)
    print("-"*70)

print("\n✅ Recipe generation complete!")


# ## 6. COMPLETE PIPELINE: IMAGE DETECTION + RECIPE GENERATION

# In[76]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

# ================== CONFIG ==================
IMG_SIZE = 224
BASE_IMAGE_DIR = "./image_sample/" 

# ================== COMPLETE PIPELINE ==================
def complete_pipeline_test_gpt2(
    image_path,
    detection_model,
    gpt2_model,
    tokenizer,
    class_names
):
    """
    Pipeline:
    1. Detect ingredient from image (MobileNetV2)
    2. Generate recipe (GPT-2 fine-tuned)
    """
    print("\n" + "="*70)
    print("🍳 COMPLETE PIPELINE: IMAGE → RECIPE")
    print("="*70)

    # ---------- STEP 1: DETECT INGREDIENT ----------
    print("\n[STEP 1] 🔍 Detecting ingredient from image...")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Cannot load image from {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    predictions = detection_model.predict(img_batch, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    detected_ingredient = class_names[predicted_class_idx]

    print(f"✅ Detected Ingredient: {detected_ingredient}")
    print(f"✅ Confidence: {confidence:.2%}")

    # ---------- STEP 2: GENERATE RECIPE ----------
    print(f"\n[STEP 2] 📝 Generating recipe for '{detected_ingredient}'...")

    generated_recipe = generate_recipe_gpt2(
        ingredient_name=detected_ingredient,
        model=gpt2_model,
        tokenizer=tokenizer,
        temperature=0.7,
        top_p=0.9
    )

    # ---------- DISPLAY RESULTS ----------
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Detected: {detected_ingredient}\nConfidence: {confidence:.2%}", fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.text(0.05, 0.95, f"Generated Recipe for: {detected_ingredient}", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.05, 0.05, generated_recipe[:800] + "...", fontsize=9, wrap=True, verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\n🎯 Detected Ingredient: {detected_ingredient}")
    print(f"📝 Generated Recipe:\n")
    print(generated_recipe)
    print("\n" + "="*70)

    return {
        'detected_ingredient': detected_ingredient,
        'confidence': confidence,
        'generated_recipe': generated_recipe
    } 


# ## 6.1 TEST WITH SAMPLE IMAGES

# In[77]:


def test_multiple_samples_gpt2(detection_model, gpt2_model, tokenizer, n_samples=20):
    """Test the complete pipeline with multiple images in folder"""
    print("\n" + "="*70)
    print("🧪 TESTING COMPLETE PIPELINE WITH MULTIPLE SAMPLES")
    print("="*70)

    all_images = [os.path.join(BASE_IMAGE_DIR, f) 
                  for f in os.listdir(BASE_IMAGE_DIR)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))]

    if len(all_images) < n_samples:
        n_samples = len(all_images)

    sample_images = random.sample(all_images, n_samples)
    results = []

    for idx, img_path in enumerate(sample_images, 1):
        print(f"\n{'='*70}")
        print(f"TEST {idx}/{n_samples}")
        print(f"{'='*70}")
        print(f"Image: {img_path}")

        result = complete_pipeline_test_gpt2(
            image_path=img_path,
            detection_model=detection_model,
            gpt2_model=gpt2_model,
            tokenizer=tokenizer,
            class_names=class_names
        )

        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("📊 TESTING SUMMARY")
    print("="*70)
    print(f"Total samples tested: {len(results)}")
    if results:
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.2%}")
        print("\nDetected Ingredients:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['detected_ingredient']} ({r['confidence']:.2%})")

    return results


# ## RUN FINAL TESTS

# In[81]:


print("\n" + "="*70)
print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
print("="*70)
print("✅ Ingredient Detection Model: Ready")
print("✅ Recipe Generation Model (GPT-2): Ready")
print("✅ Complete Pipeline: Ready")
print("="*70)

results = test_multiple_samples_gpt2(
    detection_model=model,
    gpt2_model=gpt2_model,
    tokenizer=tokenizer,
    n_samples=20
)


# ## SAVE ALL MODELS

# In[82]:


print("\n" + "="*70)
print("💾 SAVING MODELS AND TOKENIZERS")
print("="*70)
import os
import pickle

# Ingredient Detection Model
model.save(os.path.join(MODEL_DIR, 'ingredient_detection_model.keras'))

# GPT-2 Recipe Generation Model
gpt2_model.save_pretrained(os.path.join(MODEL_DIR, 'gpt2_recipe_model'))
tokenizer.save_pretrained(os.path.join(MODEL_DIR, 'gpt2_recipe_tokenizer'))

# Save class names for detection
with open(os.path.join(MODEL_DIR, 'class_names.pkl'), 'wb') as f:
    pickle.dump(class_names, f)

print("✅ All models and tokenizers saved successfully!")
print(f"📁 Location: {MODEL_DIR}")
print("="*70)


# ----

# ## <div align="center">  Reflection </div>

# In[ ]:





# In[ ]:





# In[ ]:




