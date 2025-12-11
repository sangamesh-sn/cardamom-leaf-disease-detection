import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# Import Keras MobileNetV2 and its preprocess_input from the correct module
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# === CONFIGURATION ===
SOURCE_FOLDER = 'dataset/train/phyllostica'  # Folder with input images
OUTPUT_FOLDER = 'cluster/train/phyllostica'      # Folder where clustered images will be saved
NUM_CLUSTERS = 7                              # Change as needed

# === Load Pretrained MobileNetV2 Model (Feature Extractor) ===
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# === Helper Function: Extract Feature Vector from Image ===
def extract_feature(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

# === Load and Extract Features from All Images ===
image_paths = [os.path.join(SOURCE_FOLDER, fname)
               for fname in os.listdir(SOURCE_FOLDER)
               if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

feature_list = []
valid_image_paths = []

print("🔍 Extracting features from images...")
for img_path in tqdm(image_paths):
    feat = extract_feature(img_path)
    if feat is not None:
        feature_list.append(feat)
        valid_image_paths.append(img_path)

if not feature_list:
    print("❌ No valid features found in the dataset. Exiting.")
    exit(1)

features_np = np.array(feature_list)
if features_np.ndim == 1:
    # Only one image processed successfully, reshape to (1, -1)
    features_np = features_np.reshape(1, -1)

if len(features_np) < NUM_CLUSTERS:
    print(f"❌ Not enough images to form {NUM_CLUSTERS} clusters. Found only {len(features_np)}.")
    exit(1)

# === Perform KMeans Clustering ===
print("🔗 Performing KMeans clustering...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features_np)

# === Save Images to Respective Cluster Folders ===
print("📁 Saving clustered images...")
for cluster_idx in range(NUM_CLUSTERS):
    cluster_dir = os.path.join(OUTPUT_FOLDER, f"cluster_{cluster_idx}")
    os.makedirs(cluster_dir, exist_ok=True)

for img_path, label in zip(valid_image_paths, labels):
    file_name = os.path.basename(img_path)
    target_path = os.path.join(OUTPUT_FOLDER, f"cluster_{label}", file_name)
    shutil.copy2(img_path, target_path)

print(f"✅ Done! {NUM_CLUSTERS} clusters created in '{OUTPUT_FOLDER}'.")
