from numpy import array
import cv2
from os import listdir
from os.path import join
from pandas import DataFrame

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
from matplotlib.image import imread


width = 224
height = 224
IMAGE_DIM = (width, height)


def setup_model():
    vgg = VGG16(
        input_shape=(width, height, 3),
        weights="imagenet",
        include_top=False
    )

    for layer in vgg.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    return model

def flatten_images(images):
    model = setup_model()
    return model.predict(images, verbose=0)

def load_image(image_path):
    return preprocess_input(
        cv2.resize(
            cv2.imread(image_path, 1),
            IMAGE_DIM,
            interpolation=cv2.INTER_AREA
        )
    )

def load_image_folder(folder_path):
    images = []
    for image_path in listdir(folder_path):
        images.append(load_image(join(folder_path,image_path)))

    return array(images)

def plot_dendrogram(flat_images, title):
    Z = hierarchy.linkage(flat_images, method="ward")
    hierarchy.dendrogram(Z)
    plt.title(title)
    plt.savefig(f"output/{title}__dendrogram.png")

def cluster_images(flat_images, folder, n_cluster):

    kmeans_engine = KMeans(n_clusters=n_cluster)
    clustering_result = DataFrame({
        "image": [join(folder, value) for value in listdir(folder)],
        "cluster": kmeans_engine.fit_predict(flat_images)
    })

    return clustering_result

def plot_examples(clustering_result, title, n_examples=3):

    n_cluster = len(clustering_result.cluster.value_counts())

    # take n example images from the clustered dataset
    for cluster in range(n_cluster):

        # take a sample of images
        example_subset = clustering_result.loc[clustering_result.cluster==cluster].sample(n_examples)

        fig, ax = plt.subplots(ncols=1, nrows=n_examples, figsize=(4, 4*n_examples))

        for image_index, image_path in enumerate(example_subset.image.values):
            ax[image_index].imshow(imread(image_path))
            ax[image_index].set_axis_off()
            ax[image_index].set_title(f"Example image {image_index+1}")

        fig.suptitle(f"{title}\nCluster: {cluster+1}")
        plt.savefig(f"output/{title}__cluster{cluster+1}.png")