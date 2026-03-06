import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 6   # smaller to save time

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    "cell_images",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    "cell_images",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

def plot_history(history, model_name):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.savefig(f"{model_name}_accuracy.png")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.savefig(f"{model_name}_loss.png")
    plt.close()

results = {}

# ======================
# 1. Custom CNN
# ======================

print("\n==============================")
print("Training Custom CNN")
print("==============================")

custom_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

custom_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = custom_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop]
)

plot_history(history, "CustomCNN")
results["Custom CNN"] = max(history.history['val_accuracy'])


# ======================
# 2. MobileNetV2
# ======================

print("\n==============================")
print("Training MobileNetV2")
print("==============================")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop]
)

plot_history(history, "MobileNetV2")
results["MobileNetV2"] = max(history.history['val_accuracy'])


# ======================
# 3. EfficientNetB0
# ======================

print("\n==============================")
print("Training EfficientNetB0")
print("==============================")

base_model = EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop]
)

plot_history(history, "EfficientNetB0")
results["EfficientNetB0"] = max(history.history['val_accuracy'])


# ======================
# Comparison
# ======================

comparison_df = pd.DataFrame({
    "Model": results.keys(),
    "Best Validation Accuracy": results.values()
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(comparison_df)

best_model = comparison_df.loc[
    comparison_df["Best Validation Accuracy"].idxmax()
]

print("\nBest Model:", best_model["Model"])
print("Validation Accuracy:", best_model["Best Validation Accuracy"])