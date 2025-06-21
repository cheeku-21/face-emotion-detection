import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tarfile
import requests
import zipfile

# Download FER2013 dataset
os.makedirs('data/fer2013', exist_ok=True)
os.system('kaggle datasets download -d msambare/fer2013 -p data/fer2013')
with zipfile.ZipFile('data/fer2013/fer2013.zip', 'r') as zip_ref:
    zip_ref.extractall('data/fer2013')

# Download CK+ dataset (example via a public source, replace with your access method)
os.makedirs('data/ckplus', exist_ok=True)
# Note: CK+ often requires academic access. If available, download manually or use:
# url = "http://example.com/CK+48.tar.gz"  # Replace with actual URL or local path
# response = requests.get(url)
# with open('data/ckplus/CK+48.tar.gz', 'wb') as f:
#     f.write(response.content)
# with tarfile.open('data/ckplus/CK+48.tar.gz', 'r:gz') as tar:
#     tar.extractall('data/ckplus')

# Define emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # FER2013 includes neutral

# Preprocess FER2013
def load_fer2013():
    df = pd.read_csv('data/fer2013/fer2013.csv')
    X, y = [], []
    for index, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
        img = cv2.resize(pixels, (224, 224))  # Resize to MobileNetV2 input
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        X.append(img)
        y.append(row['emotion'])
    return np.array(X) / 255.0, np.array(y)

# Preprocess CK+ (assuming folder structure: CK+48/emotion/image.png)
def load_ckplus():
    X, y = [], []
    for emotion_idx, emotion in enumerate(emotions[:-1]):  # CK+ lacks 'neutral'
        folder = f'data/ckplus/CK+48/{emotion}'
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                X.append(img / 255.0)
                y.append(emotion_idx)
    return np.array(X), np.array(y)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load and combine datasets
X_fer, y_fer = load_fer2013()
X_ck, y_ck = load_ckplus()
X = np.concatenate([X_fer, X_ck], axis=0)
y = np.concatenate([y_fer, y_ck], axis=0)

# Save preprocessed data
np.save('data/X_preprocessed.npy', X)
np.save('data/y_preprocessed.npy', y)

# Example: Generate augmented images
# for X_batch, y_batch in datagen.flow(X, y, batch_size=32):
#     # Save or use augmented data
#     break