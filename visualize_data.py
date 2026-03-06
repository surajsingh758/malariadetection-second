# visualize_data.py

import os
import matplotlib.pyplot as plt
import cv2
import random

DATASET_PATH = "cell_images"

parasitized_path = os.path.join(DATASET_PATH, "Parasitized")
uninfected_path = os.path.join(DATASET_PATH, "Uninfected")

# -------------------------------
# Class Count
# -------------------------------
print("Parasitized images:", len(os.listdir(parasitized_path)))
print("Uninfected images:", len(os.listdir(uninfected_path)))

# -------------------------------
# Class Distribution Plot (Step 2)
# -------------------------------
labels = ["Parasitized", "Uninfected"]
counts = [len(os.listdir(parasitized_path)), len(os.listdir(uninfected_path))]

plt.figure(figsize=(6,4))
plt.bar(labels, counts)
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.show()

# -------------------------------
# Sample Image Grid (Step 1)
# -------------------------------
plt.figure(figsize=(10,5))

for i in range(4):
    img_path = os.path.join(parasitized_path, random.choice(os.listdir(parasitized_path)))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.title("Parasitized")
    plt.axis("off")

for i in range(4):
    img_path = os.path.join(uninfected_path, random.choice(os.listdir(uninfected_path)))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,i+5)
    plt.imshow(img)
    plt.title("Uninfected")
    plt.axis("off")

plt.tight_layout()
plt.show()