import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from annoy import AnnoyIndex
import os
import shutil  # Rasmlarni nusxalash uchun kerak

# ResNet50 modelini yuklash, tasniflash qatlamini olib tashlash
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_image_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array)
    return embedding.flatten()

def build_annoy_index(folder_path, n_trees=10):
    images = os.listdir(folder_path)
    f = 2048  # Embedding vektorlarining o'lchami (ResNet50 uchun)
    t = AnnoyIndex(f, 'angular')  # Annoy indeksini yaratish (angular metrikasi)

    for i, img_name in enumerate(images):
        img_path = os.path.join(folder_path, img_name)
        img_embedding = get_image_embedding(img_path)
        t.add_item(i, img_embedding)  # Har bir rasm uchun embeddingni qo'shish

    t.build(n_trees)
    t.save('image_embeddings.ann')  # Indeksni saqlash

def find_similar_images_with_annoy(target_image_path, index_path, folder_path, top_n=5):
    target_embedding = get_image_embedding(target_image_path)
    f = 2048
    u = AnnoyIndex(f, 'angular')
    u.load(index_path)  # Saqlangan indeksni yuklash

    similar_indices = u.get_nns_by_vector(target_embedding, top_n)
    images = os.listdir(folder_path)

    similar_images = [images[i] for i in similar_indices]
    return similar_images

# Annoy indeksini yaratish va saqlash
folder_path = 'uzum data'
build_annoy_index(folder_path)

# Maqsadli rasm va saqlangan indeks bo'yicha o'xshash rasmlarni topish
target_image_path = 'uzum data/2.webp'
similar_images = find_similar_images_with_annoy(target_image_path, 'image_embeddings.ann', folder_path)

# O'xshash rasmlarni saqlash uchun papka yaratish
output_folder = 'similar_images1'
os.makedirs(output_folder, exist_ok=True)

print("O'xshash rasmlar:")
for img_name in similar_images:
    print(f"Rasm: {img_name}")
    source_path = os.path.join(folder_path, img_name)
    destination_path = os.path.join(output_folder, img_name)
    shutil.copy(source_path, destination_path)  # Rasmni yangi papkaga nusxalash

print(f"O'xshash rasmlar '{output_folder}' papkasiga saqlandi.")
