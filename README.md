# Processing for Microscopic images

## Cropping

Images are cropped using `opencv-python`. The image is (1) loaded and transformed into an hsv image. After that (2), orange sections are identified and black-white-mask is generated. The black-white-mask is (3) then used to find contours. If a contour is (4) wider than a certain amount, the image is (5) cropped to the contour and saved as a new image. The cropping tool will loop a set of image folders within a root-folder.

```python
from image_cropper import crop_images_to_circle

crop_images_to_circle(
    root_folder="C:\Data\Images",
    output_root_folder="C:\Data\Images_Cropped"
)
```

## Clustering

The tool can also be used to cluster images based on image content. Clustering is done with the help of an image vector (all lines of an image put together). To enhance the image before clustering, images  are loaded by `cv2`, resized to `(224, 224, 3)`, fed through an `VGG16` model (without top) and are finally flattened by a `Flatten()` layer.

The clustering is afterwards done useing a `KMeans` model (as this is capable of clustering large quantities of images). Before doing so, you can plot a dendrogram to find the optimal number of clusters.

```python
title = "Organoide_10000, 5000, 2000, 1000 Zellen"
folder = join("input_data", "Images_Cropped", "d")

images = load_image_folder(folder)
flat_images = flatten_images(images)

clustering_result = cluster_images(flat_images=flat_images, folder=folder, n_cluster=3)
plot_dendrogram(flat_images=flat_images, title=title)
plot_examples(clustering_result=clustering_result, n_examples=4, title=title)
```

## Results

### Dendrogram

![Dendrogram](output/BT20%20-I,%20+I%20von%20Anfang%20an%201x__dendrogram.png)

### Cluster Examples

![Examples Cluster 1](output/BT20%20-I,%20+I%20von%20Anfang%20an%201x__cluster1.png)

![Examples Cluster 2](output/BT20%20-I,%20+I%20von%20Anfang%20an%201x__cluster2.png)

![Examples Cluster 3](output/BT20%20-I,%20+I%20von%20Anfang%20an%201x__cluster3.png)

![Examples Cluster 4](output/BT20%20-I,%20+I%20von%20Anfang%20an%201x__cluster4.png)

![Examples Cluster 5](output/BT20%20-I,%20+I%20von%20Anfang%20an%201x__cluster5.png)
