# app_cnn.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.title("🖐 Hand Gesture Recognition")
st.write("Upload an image of a hand gesture, and the model will predict the gesture.")

# -------------------------
# 1. Load Trained Model
# -------------------------
model = load_model("hand_gesture_cnn.h5")

# -------------------------
# 2. Define Gesture Classes
# -------------------------
# Make sure these match your dataset folder names
gesture_classes = {
    0: "palm ✋",
    1: "index ☝️",
    2: "fist ✊",
    3: "thumbs_up 👍",
    4: "ok 👌"
    # Add more classes if you include more folders
}

# -------------------------
# 3. Upload Image
# -------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # -------------------------
    # 4. Preprocess Image
    # -------------------------
    img = img.resize((64,64))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)
    
    # -------------------------
    # 5. Make Prediction
    # -------------------------
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]*100
    
    st.write(f"**Predicted Gesture:** {gesture_classes.get(class_idx,'Unknown')}")
    st.write(f"**Confidence:** {confidence:.2f}%")
