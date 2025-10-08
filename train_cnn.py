# train_cnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -------------------------
# 1. Dataset Path
# -------------------------
dataset_dir = "dataset_small"  # Folder containing subfolders like 01_palm, 02_index...
img_height, img_width = 64, 64
batch_size = 16

# -------------------------
# 2. Image Data Generator
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)
print(f"Classes found: {train_generator.class_indices}")

# -------------------------
# 3. Build CNN Model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 4. Train the Model
# -------------------------
epochs = 15
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# -------------------------
# 5. Save the Model
# -------------------------
model.save("hand_gesture_cnn.h5")
print("Model saved as hand_gesture_cnn.h5")
