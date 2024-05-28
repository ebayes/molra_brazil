import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

# Set the path to the crops folder
crops_folder = "./output/crops"
n_clusters = 10

# Load images and convert them to feature vectors
image_files = [f for f in os.listdir(crops_folder) if f.endswith(".jpg") or f.endswith(".png")]
image_features = []

for image_file in image_files:
    image_path = os.path.join(crops_folder, image_file)
    image = Image.open(image_path)
    image = image.resize((256, 256))  # Resize images to a fixed size
    feature_vector = np.array(image).flatten()
    image_features.append(feature_vector)

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_features)

# Create subfolders for each cluster
for i in range(n_clusters):
    subfolder_path = os.path.join(crops_folder, f"cluster_{i}")
    os.makedirs(subfolder_path, exist_ok=True)

# Move images to their respective cluster subfolders
for image_file, cluster_label in zip(image_files, cluster_labels):
    source_path = os.path.join(crops_folder, image_file)
    destination_folder = os.path.join(crops_folder, f"cluster_{cluster_label}")
    destination_path = os.path.join(destination_folder, image_file)
    shutil.move(source_path, destination_path)

# Perform PCA for visualization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(image_features)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Plot PCA distribution
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(pca_features[cluster_labels == i, 0], pca_features[cluster_labels == i, 1], label=f"Cluster {i}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("PCA Distribution of Image Clusters")
plt.tight_layout()
plt.savefig("pca_plot.png")
plt.show()