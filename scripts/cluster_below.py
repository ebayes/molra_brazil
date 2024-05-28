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
n_clusters = 5

# Get all subfolders in the crops folder
subfolders = [os.path.join(crops_folder, f) for f in os.listdir(crops_folder) if os.path.isdir(os.path.join(crops_folder, f))]

for subfolder in subfolders:
    # Load images and convert them to feature vectors
    image_files = [f for f in os.listdir(subfolder) if f.endswith(".jpg") or f.endswith(".png")]
    image_features = []

    for image_file in image_files:
        image_path = os.path.join(subfolder, image_file)
        image = Image.open(image_path)
        image = image.resize((256, 256))  # Resize images to a fixed size
        feature_vector = np.array(image).flatten()
        image_features.append(feature_vector)

    # Check if the number of images is less than the number of clusters
    if len(image_features) < n_clusters:
        print(f"Skipping clustering for {subfolder} as the number of images ({len(image_features)}) is less than the number of clusters ({n_clusters}).")
        continue

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(image_features)

    # Create subfolders for each cluster within the current subfolder
    for i in range(n_clusters):
        cluster_subfolder_path = os.path.join(subfolder, f"cluster_{i}")
        os.makedirs(cluster_subfolder_path, exist_ok=True)

    # Move images to their respective cluster subfolders
    for image_file, cluster_label in zip(image_files, cluster_labels):
        source_path = os.path.join(subfolder, image_file)
        destination_folder = os.path.join(subfolder, f"cluster_{cluster_label}")
        destination_path = os.path.join(destination_folder, image_file)
        try:
            shutil.move(source_path, destination_path)
        except Exception as e:
            print(f"Error moving file {source_path} to {destination_path}: {e}")

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
    plt.title(f"PCA Distribution of Image Clusters in {os.path.basename(subfolder)}")
    plt.tight_layout()
    # Save the plot with the same name as the subfolder
    pca_plot_filename = os.path.basename(subfolder) + "_pca_plot.png"
    plt.savefig(os.path.join(subfolder, pca_plot_filename))
    plt.close()  # Close the plot instead of showing it