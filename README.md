Here’s a detailed and polished README for your code:

---

# Image Similarity Search using ResNet50 and Annoy

This project uses a pre-trained ResNet50 model to extract feature embeddings from images and applies the Annoy library to find similar images efficiently based on those embeddings. It helps in searching for visually similar images from a dataset using angular distance.

## Features

- **Feature Extraction**: Extracts image embeddings using ResNet50.
- **Nearest Neighbor Search**: Uses the Annoy library for efficient similarity search.
- **Image Similarity**: Finds and copies the top N most similar images to the target image.
- **Scalability**: Efficient even with large datasets, thanks to Annoy's indexing technique.

## How it Works

1. **ResNet50 for Feature Extraction**:
    - A pre-trained ResNet50 model is loaded (without the classification layer).
    - The model extracts 2048-dimensional feature embeddings for each image.

2. **Annoy for Fast Similarity Search**:
    - An Annoy index is built for the image embeddings.
    - It finds the top N most similar images to the target image using angular distance.
  
3. **Usage**:
    - The embeddings are generated, indexed, and saved.
    - Given a target image, the model retrieves the most similar images from the dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Bahrombekk/image-search.git
    cd image-search
    ```

2. Install the necessary dependencies:
    ```bash
    pip install tensorflow keras annoy numpy pillow
    ```

3. Download the ResNet50 pre-trained weights if not available locally.

## Usage

1. **Building the Index**:
    - Place your images in a folder (e.g., `uzum data`).
    - Build the Annoy index using the images in the folder:
      ```python
      build_annoy_index('path_to_your_images')
      ```

2. **Finding Similar Images**:
    - Set the path of the target image and the image folder.
    - Run the script to find the top N similar images:
      ```python
      similar_images = find_similar_images_with_annoy('path_to_target_image', 'path_to_index_file', 'path_to_image_folder')
      ```

3. **Copying Similar Images**:
    - Similar images are copied to the `similar_images1` folder by default.

## Example

To find similar images in the `uzum data` folder:

```bash
python image.py
```

## Output

- The script outputs the top 5 similar images to the target image.
- The images are saved to the `similar_images1` folder.

## Acknowledgments

- **ResNet50**: Pre-trained model used for feature extraction.
- **Annoy**: Library for efficient similarity search.

---

This README outlines the purpose, installation steps, and usage of your code, making it clear for users to get started quickly.
